#!/usr/bin/env python3
"""Small SUMO/TraCI smoke test for the ada_v1 HPC environment.

It starts one city, advances SUMO for a few simulation steps, prints basic
vehicle/queue information, and optionally saves screenshots from sumo-gui.
The script intentionally does not depend on the training stack.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=("jinan", "hangzhou"), default="jinan")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="directory containing <city>.sumocfg (default: repository data/<City>/3_4 or 4_4)")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("runtime/sumo_smoke"))
    parser.add_argument("--screenshot-every", type=int, default=1)
    parser.add_argument("--gui", action="store_true", help="use sumo-gui and save GUI screenshots")
    parser.add_argument("--no-gui", dest="gui", action="store_false")
    parser.set_defaults(gui=False)
    parser.add_argument("--keep-running", action="store_true",
                        help="leave the GUI open after the requested steps")
    return parser.parse_args()


def default_data_dir(repo_root: Path, city: str) -> Path:
    return repo_root / "data" / ("Jinan" if city == "jinan" else "Hangzhou") / ("3_4" if city == "jinan" else "4_4")


def main() -> int:
    args = parse_args()
    if args.steps < 0 or args.screenshot_every <= 0:
        raise SystemExit("--steps must be non-negative and --screenshot-every must be positive")

    try:
        import traci
    except ImportError as exc:
        print("ERROR: traci is not importable in this Python environment.", file=sys.stderr)
        print("Install it or use SUMO_HOME/tools, for example:", file=sys.stderr)
        print("  export SUMO_HOME=/path/to/sumo", file=sys.stderr)
        print("  export PYTHONPATH=\"$SUMO_HOME/tools:$PYTHONPATH\"", file=sys.stderr)
        raise SystemExit(2) from exc

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = (args.data_dir or default_data_dir(repo_root, args.city)).resolve()
    sumocfg = data_dir / f"{args.city}.sumocfg"
    if not sumocfg.is_file():
        raise SystemExit(f"SUMO config not found: {sumocfg}")

    binary_name = "sumo-gui" if args.gui else "sumo"
    binary = shutil.which(binary_name)
    if binary is None:
        sumo_home = os.environ.get("SUMO_HOME")
        candidate = Path(sumo_home) / "bin" / binary_name if sumo_home else None
        if candidate is not None and candidate.is_file():
            binary = str(candidate)
    if binary is None:
        raise SystemExit(f"{binary_name} not found; set SUMO_HOME or add its bin directory to PATH")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"city={args.city} config={sumocfg}")
    print(f"binary={binary} gui={args.gui} steps={args.steps} output={output_dir}", flush=True)

    command = [
        binary, "-c", str(sumocfg), "--seed", str(args.seed),
        "--start", "--no-step-log", "--duration-log.disable",
    ]
    if not args.keep_running:
        command += ["--quit-on-end"]

    traci.start(command, label="ada_v1_sumo_smoke")
    try:
        if args.gui:
            # SUMO's default view is a 2D OpenGL view; screenshots verify the
            # GUI/OpenGL path and can be inspected as the requested 3D-style
            # visualization output.
            try:
                traci.gui.setZoom("View #0", 500.0)
            except Exception as exc:  # GUI view names vary by SUMO build.
                print(f"warning: could not configure GUI view: {exc}", file=sys.stderr)

        for step in range(args.steps):
            traci.simulationStep()
            sim_time = traci.simulation.getTime()
            vehicles = traci.vehicle.getIDCount()
            halted = sum(
                1 for vehicle_id in traci.vehicle.getIDList()
                if traci.vehicle.getSpeed(vehicle_id) < 0.1
            )
            print(f"step={step + 1} sim_time={sim_time:.1f} vehicles={vehicles} halted={halted}", flush=True)
            if args.gui and (step + 1) % args.screenshot_every == 0:
                target = output_dir / f"{args.city}_step_{step + 1:04d}.png"
                traci.gui.screenshot("View #0", filename=str(target), width=1280, height=720)
                print(f"screenshot={target}", flush=True)
    finally:
        traci.close()
    print("SUMO_SMOKE_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
