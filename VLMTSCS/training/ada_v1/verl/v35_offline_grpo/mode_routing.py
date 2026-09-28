"""Adaptive, utility-calibrated routing for the V35 FAST/SLOW mode token."""

from __future__ import annotations

import datetime
import json
import math
import os
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def binary_slow_probability(fast_logprob: float, slow_logprob: float) -> float:
    """Normalize the two candidate sequence log-probabilities."""
    delta = max(-60.0, min(60.0, float(slow_logprob) - float(fast_logprob)))
    return 1.0 / (1.0 + math.exp(-delta))


def mode_candidate_token_ids(tokenizer) -> dict[str, list[int]]:
    """Return the exact token sequences scored by the two-stage router."""
    candidates = {
        "fast": tokenizer.encode("<mode>fast</mode>", add_special_tokens=False),
        "slow": tokenizer.encode("<mode>slow</mode>", add_special_tokens=False),
    }
    if not candidates["fast"] or not candidates["slow"]:
        raise ValueError("mode candidates must tokenize to non-empty sequences")
    return candidates


def mode_token_layout(tokenizer) -> dict[str, Any]:
    """Token layout used to score and then inject the selected mode."""
    opening = tokenizer.encode("\n<mode>", add_special_tokens=False)
    closing = tokenizer.encode("</mode>", add_special_tokens=False)
    words = {
        "fast": tokenizer.encode("fast", add_special_tokens=False),
        "slow": tokenizer.encode("slow", add_special_tokens=False),
    }
    if not opening or not closing or not all(words.values()):
        raise ValueError("mode layout must tokenize to non-empty sequences")
    full: dict[str, list[int]] = {}
    for mode, word_ids in words.items():
        expected = tokenizer.encode(f"\n<mode>{mode}</mode>", add_special_tokens=False)
        composed = [*opening, *word_ids, *closing]
        if expected != composed:
            raise ValueError(
                "two-stage routing requires composable tokenization for "
                f"\\n<mode>{mode}</mode>"
            )
        full[mode] = composed
    return {"opening": opening, "closing": closing, "words": words, "full": full}


def extract_prompt_sequence_token_logprobs(
    extra_fields: dict[str, Any], candidate_ids: list[int]
) -> list[float] | None:
    """Extract a candidate sequence logprob from verl's prompt-logprob fields.

    ``extract_prompt_logprobs`` stores one token-id/logprob row per prompt
    position and appends a dummy row for the final prompt token.  The helper
    searches by token ids instead of relying on an offset, which keeps it
    compatible with vLLM's multimodal prefix handling.
    """
    prompt_ids = extra_fields.get("prompt_ids")
    prompt_logprobs = extra_fields.get("prompt_logprobs")
    if not isinstance(prompt_ids, list) or not isinstance(prompt_logprobs, list):
        return None
    flattened_ids: list[int | None] = []
    flattened_logprobs: list[float | None] = []
    for ids, values in zip(prompt_ids, prompt_logprobs):
        if not isinstance(ids, list) or not isinstance(values, list) or not ids or not values:
            flattened_ids.append(None)
            flattened_logprobs.append(None)
            continue
        try:
            flattened_ids.append(int(ids[0]))
            flattened_logprobs.append(float(values[0]))
        except (TypeError, ValueError):
            flattened_ids.append(None)
            flattened_logprobs.append(None)
    target = [int(value) for value in candidate_ids]
    for start in range(len(flattened_ids) - len(target), -1, -1):
        if flattened_ids[start : start + len(target)] != target:
            continue
        values = flattened_logprobs[start : start + len(target)]
        if all(value is not None and math.isfinite(value) for value in values):
            return [float(value) for value in values]
    return None


def extract_prompt_sequence_logprob(extra_fields: dict[str, Any], candidate_ids: list[int]) -> float | None:
    values = extract_prompt_sequence_token_logprobs(extra_fields, candidate_ids)
    return float(sum(values)) if values is not None else None


def write_mode_probability_diagnostic(path: str, record: dict[str, Any]) -> None:
    """Append a JSON diagnostic for the two-stage Val/deployment router."""
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **record,
    }
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n")


@dataclass(frozen=True)
class CalibrationObservation:
    p_slow: float
    fast_utility: float
    slow_utility: float

    def as_dict(self) -> dict[str, float]:
        return {
            "p_slow": float(self.p_slow),
            "fast_utility": float(self.fast_utility),
            "slow_utility": float(self.slow_utility),
        }


class AdaptiveModeThreshold:
    """Choose a deterministic threshold that maximizes observed utility.

    The window stores paired FAST/SLOW outcomes from forced training exploration.
    Mode counts never enter the objective, so the calibrator cannot manufacture a
    requested FAST/SLOW ratio.
    """

    def __init__(
        self,
        *,
        path: str,
        window_size: int = 1200,
        ema_decay: float = 0.9,
        min_threshold: float = 0.1,
        max_threshold: float = 0.9,
        grid_step: float = 0.01,
        initial_threshold: float = 0.5,
    ) -> None:
        if window_size <= 0:
            raise ValueError("mode threshold window_size must be positive")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("mode threshold ema_decay must be in [0, 1)")
        if not 0.0 < min_threshold < max_threshold < 1.0:
            raise ValueError("mode threshold bounds must lie strictly inside (0, 1)")
        if grid_step <= 0.0:
            raise ValueError("mode threshold grid_step must be positive")
        self.path = str(path)
        self.window_size = int(window_size)
        self.ema_decay = float(ema_decay)
        self.min_threshold = float(min_threshold)
        self.max_threshold = float(max_threshold)
        self.grid_step = float(grid_step)
        self.threshold = float(initial_threshold)
        self.observations: deque[CalibrationObservation] = deque(maxlen=self.window_size)
        self.last_step = -1
        self._load()

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as handle:
            payload = json.load(handle)
        threshold = float(payload.get("threshold", self.threshold))
        if self.min_threshold <= threshold <= self.max_threshold:
            self.threshold = threshold
        self.last_step = int(payload.get("global_step", -1))
        for raw in payload.get("observations", [])[-self.window_size :]:
            try:
                observation = CalibrationObservation(
                    p_slow=float(raw["p_slow"]),
                    fast_utility=float(raw["fast_utility"]),
                    slow_utility=float(raw["slow_utility"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if self._valid(observation):
                self.observations.append(observation)

    @staticmethod
    def _valid(observation: CalibrationObservation) -> bool:
        values = (observation.p_slow, observation.fast_utility, observation.slow_utility)
        return all(math.isfinite(value) for value in values) and 0.0 <= observation.p_slow <= 1.0

    def _candidates(self) -> list[float]:
        count = int(round((self.max_threshold - self.min_threshold) / self.grid_step))
        return [min(self.max_threshold, self.min_threshold + index * self.grid_step) for index in range(count + 1)]

    @staticmethod
    def _mean_utility(observations: Iterable[CalibrationObservation], threshold: float) -> float:
        values = [
            row.slow_utility if row.p_slow >= threshold else row.fast_utility
            for row in observations
        ]
        return sum(values) / len(values) if values else 0.0

    def update(self, observations: Iterable[CalibrationObservation], *, global_step: int) -> dict[str, float]:
        accepted = [row for row in observations if self._valid(row)]
        self.observations.extend(accepted)
        if not self.observations:
            return self.metrics()

        candidates = self._candidates()
        objectives = [(self._mean_utility(self.observations, value), value) for value in candidates]
        best_objective = max(value for value, _ in objectives)
        tied = [
            value for objective, value in objectives
            if math.isclose(objective, best_objective, rel_tol=0.0, abs_tol=1e-12)
        ]
        raw_threshold = min(tied, key=lambda value: (abs(value - self.threshold), abs(value - 0.5)))
        self.threshold = (
            self.ema_decay * self.threshold
            + (1.0 - self.ema_decay) * raw_threshold
        )
        self.threshold = min(self.max_threshold, max(self.min_threshold, self.threshold))
        self.last_step = int(global_step)
        metrics = self.metrics()
        metrics.update(
            raw_threshold=float(raw_threshold),
            best_window_utility=float(best_objective),
            accepted_observations=float(len(accepted)),
        )
        self._save(metrics)
        return metrics

    def metrics(self) -> dict[str, float]:
        rows = list(self.observations)
        if not rows:
            return {
                "threshold": self.threshold,
                "window_size": 0.0,
                "routed_slow_ratio": 0.0,
                "mean_p_slow_model": 0.0,
                "mean_fast_utility": 0.0,
                "mean_slow_utility": 0.0,
                "chosen_window_utility": 0.0,
                "oracle_window_utility": 0.0,
            }
        return {
            "threshold": self.threshold,
            "window_size": float(len(rows)),
            "routed_slow_ratio": sum(row.p_slow >= self.threshold for row in rows) / len(rows),
            "mean_p_slow_model": sum(row.p_slow for row in rows) / len(rows),
            "mean_fast_utility": sum(row.fast_utility for row in rows) / len(rows),
            "mean_slow_utility": sum(row.slow_utility for row in rows) / len(rows),
            "chosen_window_utility": self._mean_utility(rows, self.threshold),
            "oracle_window_utility": sum(max(row.fast_utility, row.slow_utility) for row in rows) / len(rows),
        }

    def _save(self, metrics: dict[str, float]) -> None:
        if not self.path:
            return
        target = Path(self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "global_step": self.last_step,
            "threshold": self.threshold,
            "objective": "mean_paired_decision_utility",
            "metrics": metrics,
            "observations": [row.as_dict() for row in self.observations],
        }
        fd, temporary_path = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, separators=(",", ":"))
                handle.write("\n")
            os.replace(temporary_path, target)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)


def load_threshold(path: str, default: float = 0.5) -> float:
    try:
        with open(path, encoding="utf-8") as handle:
            value = float(json.load(handle)["threshold"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        value = float(default)
    return min(1.0, max(0.0, value))


class ModeThresholdLogitsProcessor:
    """Force only the FAST/SLOW token using a calibrated deterministic threshold."""

    def __init__(
        self,
        *,
        mode_open_token_ids: list[int],
        fast_token_id: int,
        slow_token_id: int,
        threshold_path: str,
        default_threshold: float = 0.5,
        diagnostic_path: str = "",
        sample_id: str = "",
        global_step: int = -1,
    ) -> None:
        self.mode_open_token_ids = tuple(int(value) for value in mode_open_token_ids)
        self.fast_token_id = int(fast_token_id)
        self.slow_token_id = int(slow_token_id)
        self.threshold_path = str(threshold_path)
        self.default_threshold = float(default_threshold)
        self.diagnostic_path = str(diagnostic_path)
        self.sample_id = str(sample_id)
        self.global_step = int(global_step)
        self._routed = False

    def __call__(self, *args):
        if len(args) < 2:
            raise TypeError("mode threshold logits processor requires token ids and logits")
        logits = args[-1]
        token_ids = args[-2]
        if self._routed:
            return logits
        generated = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
        width = len(self.mode_open_token_ids)
        if width == 0 or tuple(generated[-width:]) != self.mode_open_token_ids:
            return logits

        fast_logit = float(logits[self.fast_token_id].item())
        slow_logit = float(logits[self.slow_token_id].item())
        p_slow = binary_slow_probability(fast_logit, slow_logit)
        threshold = load_threshold(self.threshold_path, self.default_threshold)
        selected_mode = "slow" if p_slow >= threshold else "fast"
        selected_id = self.slow_token_id if selected_mode == "slow" else self.fast_token_id
        selected_logit = logits[selected_id].clone()
        logits.fill_(float("-inf"))
        logits[selected_id] = selected_logit
        self._routed = True
        self._write_diagnostic(p_slow, threshold, selected_mode, fast_logit, slow_logit)
        return logits

    def _write_diagnostic(
        self,
        p_slow: float,
        threshold: float,
        selected_mode: str,
        fast_logit: float,
        slow_logit: float,
    ) -> None:
        if not self.diagnostic_path:
            return
        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "pid": os.getpid(),
            "global_step": self.global_step,
            "sample_id": self.sample_id,
            "p_slow_model": p_slow,
            "p_fast_model": 1.0 - p_slow,
            "threshold": threshold,
            "selected_mode": selected_mode,
            "fast_logit": fast_logit,
            "slow_logit": slow_logit,
            "decode": "adaptive_threshold_greedy",
        }
        path = Path(self.diagnostic_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")


def build_mode_threshold_processor(tokenizer, **kwargs: Any) -> ModeThresholdLogitsProcessor:
    mode_open = tokenizer.encode("<mode>", add_special_tokens=False)
    fast = tokenizer.encode("fast", add_special_tokens=False)
    slow = tokenizer.encode("slow", add_special_tokens=False)
    fast_layout = tokenizer.encode("<mode>fast", add_special_tokens=False)
    slow_layout = tokenizer.encode("<mode>slow", add_special_tokens=False)
    candidates_are_composable = (
        fast_layout == [*mode_open, *fast]
        and slow_layout == [*mode_open, *slow]
    )
    if not mode_open or len(fast) != 1 or len(slow) != 1 or not candidates_are_composable:
        raise ValueError(
            "adaptive mode routing requires tokenization(<mode> + candidate) to equal "
            "tokenization(<mode>) + one candidate token for both fast and slow"
        )
    return ModeThresholdLogitsProcessor(
        mode_open_token_ids=mode_open,
        fast_token_id=fast[0],
        slow_token_id=slow[0],
        **kwargs,
    )
