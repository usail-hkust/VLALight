#!/usr/bin/env python3
"""Validate the live Stage 2 online GRPO deployment before launching VERL."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for the online preflight") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _required_modules() -> tuple[str, ...]:
    return ("torch", "ray", "transformers", "requests", "yaml", "tensordict")


def _bootstrap_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"bootstrap row is not an object: {path}:{line_number}")
            rows.append(row)
    return rows


def _check_bootstrap(path: Path, split: str, expected: int, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing {split} bootstrap JSONL: {path}")
        return
    try:
        rows = _bootstrap_rows(path)
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
        return
    if len(rows) != expected:
        errors.append(f"{split} bootstrap has {len(rows)} rows; expected {expected}: {path}")
    required = {"city", "stream_id", "seed", "ordinal"}
    stream_ids: set[str] = set()
    for index, row in enumerate(rows):
        extra = row.get("extra_info")
        if not isinstance(extra, Mapping) or not required.issubset(extra):
            errors.append(f"{split} bootstrap row {index} is missing metadata {sorted(required)}")
            break
        stream_id = str(extra["stream_id"])
        if stream_id in stream_ids:
            errors.append(f"{split} bootstrap contains duplicate stream_id {stream_id!r}: {path}")
            break
        stream_ids.add(stream_id)


def _endpoint_models(endpoint: str, timeout: float) -> tuple[list[str], str | None]:
    try:
        import requests

        response = requests.get(endpoint.rstrip("/") + "/v1/models", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - expose one actionable preflight error
        return [], str(exc)
    values = payload.get("data", []) if isinstance(payload, Mapping) else []
    models = [str(item.get("id")) for item in values if isinstance(item, Mapping) and item.get("id")]
    return models, None


def _model_matches(configured: str, served: list[str]) -> bool:
    if configured in served:
        return True
    configured_path = Path(configured).as_posix().rstrip("/")
    for value in served:
        served_path = Path(value).as_posix().rstrip("/")
        if served_path == configured_path or Path(served_path).name == Path(configured_path).name:
            return True
    return False


def _check_contract(verl_root: Path, python_bin: str, errors: list[str]) -> None:
    script = verl_root / "v35_online_cooperative_grpo" / "verify_online_cooperative.py"
    if not script.is_file():
        errors.append(f"missing online contract test: {script}")
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(verl_root), str(verl_root.parent.parent.parent)) if value
    )
    try:
        result = subprocess.run(
            [python_bin, str(script)],
            cwd=verl_root,
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        errors.append(f"could not run online contract test: {exc}")
        return
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-20:])
        errors.append(f"online contract test failed (exit {result.returncode}):\n{tail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="online_sumo.yaml")
    parser.add_argument("--verl-config", type=Path, required=True, help="Hydra training config")
    parser.add_argument("--endpoint", default=None, help="OpenAI-compatible policy endpoint")
    parser.add_argument("--model", default=None, help="Expected served model ID/path")
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--skip-contract", action="store_true")
    parser.add_argument(
        "--check-endpoint",
        action="store_true",
        help=(
            "also require the optional OpenAI-compatible HTTP policy endpoint; "
            "the formal in-process VERL rollout does not need it"
        ),
    )
    parser.add_argument("--allow-model-mismatch", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    config_path = args.config.resolve()
    verl_config_path = args.verl_config.resolve()
    if not config_path.is_file():
        errors.append(f"missing online config: {config_path}")
        config: Mapping[str, Any] = {}
    else:
        try:
            config = _load_yaml(config_path)
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(str(exc))
            config = {}

    if not verl_config_path.is_file():
        errors.append(f"missing VERL config: {verl_config_path}")
    else:
        text = verl_config_path.read_text(encoding="utf-8")
        match = re.search(r"(?m)^\s*use_v1:\s*(true|false)\s*$", text, flags=re.I)
        if match is None:
            errors.append("VERL config must declare trainer.use_v1 (CLI may override it)")
        # Both trainer generations are supported.  The V1 online bridge is
        # selected by the launch override (trainer.use_v1=true) and no longer
        # conflicts with this preflight check.

    for module in _required_modules():
        if importlib.util.find_spec(module) is None:
            errors.append(f"missing Python dependency: {module}")

    model = args.model or str(((config.get("policy") or {}).get("model") or ""))
    if not model:
        errors.append("online policy.model is empty")
    elif not Path(model).is_dir():
        errors.append(f"policy model directory does not exist: {model}")

    cities = config.get("cities") if isinstance(config, Mapping) else {}
    if not isinstance(cities, Mapping) or not cities:
        errors.append("online config has no cities")
    else:
        for city, value in cities.items():
            if not isinstance(value, Mapping):
                errors.append(f"city config is not a mapping: {city}")
                continue
            data_root = value.get("data_root")
            if data_root and not Path(str(data_root)).is_dir():
                errors.append(f"{city} data_root does not exist: {data_root}")

    package_root = config_path.parent
    train_cfg = config.get("train") if isinstance(config, Mapping) else {}
    val_cfg = config.get("val") if isinstance(config, Mapping) else {}
    train_expected = int((train_cfg or {}).get("batch_size", 0))
    val_expected = int((val_cfg or {}).get("batch_size", 0))
    _check_bootstrap(package_root / "data" / "online_train_bootstrap.jsonl", "train", train_expected, errors)
    _check_bootstrap(package_root / "data" / "online_val_bootstrap.jsonl", "val", val_expected, errors)

    sumo = shutil.which("sumo")
    if sumo is None:
        sumo_home = os.environ.get("SUMO_HOME")
        candidate = Path(sumo_home) / "bin" / "sumo" if sumo_home else None
        if candidate is None or not candidate.is_file():
            errors.append("SUMO executable not found; set SUMO_HOME or add sumo to PATH")
        else:
            sumo = str(candidate)

    endpoint = args.endpoint or str(((config.get("policy") or {}).get("base_url") or "http://127.0.0.1:8088"))
    served: list[str] = []
    if args.check_endpoint:
        served, endpoint_error = _endpoint_models(endpoint, args.timeout)
        if endpoint_error:
            errors.append(f"policy endpoint check failed ({endpoint}): {endpoint_error}")
        elif not served:
            errors.append(f"policy endpoint returned no models: {endpoint}")
        elif model and not _model_matches(model, served):
            message = f"served model does not match configured model; configured={model!r}, served={served!r}"
            (warnings if args.allow_model_mismatch else errors).append(message)

    if not args.skip_contract:
        # The Hydra config lives at ``<verl_root>/verl/trainer/config``.
        # ``parents[2]`` is the inner Python package directory
        # (``<verl_root>/verl``), while the online package is a sibling of
        # that directory.  Resolve the outer root explicitly so preflight
        # checks the same package used by ``main_ppo_v0``.
        online_root = next(
            (parent for parent in verl_config_path.parents if (parent / "v35_online_cooperative_grpo").is_dir()),
            None,
        )
        if online_root is None:
            errors.append(
                "cannot locate v35_online_cooperative_grpo relative to VERL config: "
                f"{verl_config_path}"
            )
        else:
            _check_contract(online_root, args.python_bin, errors)

    print(f"preflight_config={config_path}")
    print(f"preflight_model={model}")
    if args.check_endpoint:
        print(f"preflight_endpoint={endpoint}")
    else:
        print("preflight_endpoint=skipped (formal path uses in-process VERL generation)")
    if sumo:
        print(f"preflight_sumo={sumo}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("PREFLIGHT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
