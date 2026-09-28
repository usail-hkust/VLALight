"""Stage 2 structured-text prompt and response protocol.

Stage 1 perception is already supervised by SFT.  Stage 2 receives the two
perception JSON objects as context and trains only the decision suffix.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PHASES = ("ETWT", "NTST", "ELWL", "NLSL")


def _formal_prompt() -> str:
    source = Path(__file__).resolve().with_name(
        "stage2_cooperative_decision_prompt_formal.txt"
    )
    if not source.is_file():
        raise FileNotFoundError(f"formal Stage 2 prompt is missing: {source}")
    text = source.read_text(encoding="utf-8").strip()
    if "{{LOCAL_PERCEPTION_JSON}}" not in text or "{{COOPERATIVE_PERCEPTION_JSON}}" not in text:
        raise ValueError(f"formal Stage 2 prompt has missing placeholders: {source}")
    return text


DEFAULT_PROMPT = _formal_prompt()


@dataclass(frozen=True)
class DecisionResponse:
    mode: str | None
    reasoning: str
    signal: str | None
    format_penalty: float
    signal_valid: bool
    format_valid: bool
    raw: str


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def build_stage2_prompt(
    local_perception: Any,
    cooperative_perception: Any,
    *,
    template: str | None = None,
    forced_mode: str | None = None,
) -> str:
    """Render one independent intersection prompt without media placeholders."""
    if forced_mode is not None and forced_mode not in {"fast", "slow"}:
        raise ValueError("forced_mode must be fast, slow, or None")
    local_json = _compact_json(local_perception)
    cooperative_json = _compact_json(cooperative_perception)
    prompt = template or DEFAULT_PROMPT
    # Support the repository's formal template names without interpreting
    # arbitrary JSON braces as ``str.format`` fields.
    prompt = prompt.replace("{{LOCAL_PERCEPTION_JSON}}", local_json)
    prompt = prompt.replace("{{COOPERATIVE_PERCEPTION_JSON}}", cooperative_json)
    if "{local_json}" in prompt or "{cooperative_json}" in prompt:
        prompt = prompt.replace("{local_json}", local_json)
        prompt = prompt.replace("{cooperative_json}", cooperative_json)
    # ``forced_mode`` is rollout metadata, never prompt content.  Training
    # enforces it in vLLM's structured-output decoder so the formal prompt is
    # byte-for-byte identical across fast/slow candidates and deployment.
    return prompt


def build_stage2_prompt_from_cooperative_json(
    observation: Mapping[str, Any],
    *,
    template: str | None = None,
    forced_mode: str | None = None,
) -> str:
    """Build the formal Stage 2 prompt from one ``{local, neighbors}`` object."""
    if not isinstance(observation, Mapping):
        raise TypeError("cooperative observation must be a mapping")
    if "local" not in observation or "neighbors" not in observation:
        raise ValueError("cooperative observation requires local and neighbors")
    return build_stage2_prompt(
        observation["local"],
        {"neighbors": observation["neighbors"]},
        template=template,
        forced_mode=forced_mode,
    )


def route_stage2_observation(
    local_perception: Any,
    neighbors: Any,
) -> dict[str, Any]:
    """Assemble the canonical Stage 2 object from local and neighbor data.

    Stage 1 is intentionally limited to one intersection's ``<perception>``
    block.  This router-owned helper is the only place that creates the
    cross-intersection ``{local, neighbors}`` structure consumed by Stage 2.
    """
    if isinstance(neighbors, Mapping):
        # observation_builder stores map-routed neighbors internally as
        # north/east/south/west keys.  Expose the Stage 2 contract as the
        # ordered array used by the cooperative prompt dataset.
        direction_names = {"north": "N", "east": "E", "south": "S", "west": "W"}
        routed = []
        for side, item in neighbors.items():
            if not isinstance(item, Mapping):
                continue
            movements = item.get("upstream_movements", item.get("movements", {}))
            routed.append({
                "intersection_id": item.get("intersection_id", item.get("source_intersection")),
                "direction": item.get("direction", direction_names.get(str(side).lower(), side)),
                **({"distance_m": item["distance_m"]} if "distance_m" in item else {}),
                "movements": movements,
            })
        neighbors = routed
    elif isinstance(neighbors, tuple):
        neighbors = list(neighbors)
    elif isinstance(neighbors, list):
        neighbors = list(neighbors)
    return {"local": local_perception, "neighbors": neighbors}


def assistant_prefix(mode: str) -> str:
    if mode not in {"fast", "slow"}:
        raise ValueError("mode must be fast or slow")
    return f"<mode>{mode}</mode>\n"


def _tags(text: str, name: str) -> list[str]:
    return re.findall(rf"<{name}>\s*(.*?)\s*</{name}>", text or "", flags=re.I | re.S)


def parse_decision_response(response: str, *, forced_mode: str | None = None) -> DecisionResponse:
    """Parse and classify a Stage 2 response with the documented penalties."""
    raw = response or ""
    mode_values = _tags(raw, "mode")
    reasoning_values = _tags(raw, "reasoning")
    signal_values = _tags(raw, "signal")
    mode = mode_values[0].strip().lower() if len(mode_values) == 1 else None
    signal = signal_values[0].strip().upper() if len(signal_values) == 1 else None
    reasoning = reasoning_values[0].strip() if len(reasoning_values) == 1 else ""
    signal_valid = len(signal_values) == 1 and signal in PHASES
    tag_counts_ok = all(
        len(re.findall(rf"</?{name}>", raw, flags=re.I)) == 2
        for name in ("mode", "signal")
    )
    if mode == "fast":
        branch_ok = not reasoning_values and not re.search(r"</?reasoning>", raw, re.I)
        expected = rf"\s*<mode>fast</mode>\s*<signal>{signal or '.*?'}</signal>\s*"
    elif mode == "slow":
        branch_ok = len(reasoning_values) == 1 and bool(reasoning)
        expected = rf"\s*<mode>slow</mode>\s*<reasoning>.*?</reasoning>\s*<signal>{signal or '.*?'}</signal>\s*"
    else:
        branch_ok = False
        expected = r"a^"
    exact = re.fullmatch(expected, raw, flags=re.I | re.S) is not None
    if forced_mode is not None and mode != forced_mode:
        branch_ok = False
    format_valid = bool(mode in {"fast", "slow"} and signal_valid and tag_counts_ok and branch_ok and exact)
    if not signal_valid:
        penalty = -1.0
    elif not format_valid:
        penalty = -0.5
    else:
        penalty = 0.0
    return DecisionResponse(mode, reasoning, signal, penalty, signal_valid, format_valid, raw)


def executable_signal(parsed: DecisionResponse, current_phase: str) -> str:
    """Return a SUMO-safe phase while preserving invalid output for reward."""
    if parsed.signal_valid and parsed.signal in PHASES:
        return parsed.signal
    if current_phase not in PHASES:
        raise ValueError(f"invalid fallback current phase: {current_phase!r}")
    return current_phase
