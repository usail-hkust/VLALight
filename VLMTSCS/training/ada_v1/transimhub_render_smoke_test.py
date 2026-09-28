"""Smoke-test TransSimHub offscreen rendering and image persistence.

This intentionally does not start sumo-gui.  It launches SUMO through the
project SUMOEnv, feeds TraCI observations to TSHubRenderer, and verifies that
ImageSaver writes non-empty image files to disk.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--city", default="jinan", choices=("jinan", "hangzhou"))
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--rendering-backend",
        default="p3headlessgl",
        choices=("p3headlessgl", "p3tinydisplay", "pandagl"),
        help="Panda3D backend; p3headlessgl is intended for headless HPC rendering",
    )
    p.add_argument("--output-dir", type=Path, default=ROOT / "training" / "ada_v1" / "runtime" / "transimhub_render_smoke")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")

    from utils.vlm_config import get_path_conf, get_traffic_env_conf
    from utils.sumo_env import SUMOEnv
    from utils.image_saver import ImageSaver

    try:
        renderer_mod = importlib.import_module(
            "TransSimHub.tshub.tshub_env3d.vis3d_renderer.tshub_render"
        )
        TSHubRenderer = renderer_mod.TSHubRenderer
    except Exception as exc:
        raise RuntimeError(
            "Cannot import TransSimHub TSHubRenderer; install/enable the "
            "TransSimHub and Panda3D dependencies in this environment: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    conf = get_traffic_env_conf(args.city, eightphase=False)
    conf["VLM_CONFIG"]["SHOW_3D_WINDOW"] = False
    conf["VLM_CONFIG"]["RENDERING_BACKEND"] = args.rendering_backend
    work_dir = args.output_dir / "sumo"
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = get_path_conf(args.city, str(work_dir))
    env = SUMOEnv(str(args.output_dir / "logs"), str(work_dir), conf, paths,
                  inter_phase_mapping=conf.get("INTER_PHASE_MAPPING"))
    renderer = None
    try:
        print(f"render_mode=offscreen rendering_backend={args.rendering_backend}")
        env.reset(use_gui=False, seed=args.seed, verbose=True)
        tls_ids = list(env.id_to_index.keys())
        if not tls_ids:
            raise RuntimeError("SUMO started but no traffic-light IDs were found")

        vlm = conf["VLM_CONFIG"]
        sensor_type = vlm.get("TLS_SENSOR_TYPE", "junction_front_all")
        sensor_config = {"tls": {
            tls_id: {"sensor_types": [sensor_type],
                     "tls_camera_height": vlm.get("TLS_CAMERA_HEIGHT", 15)}
            for tls_id in tls_ids
        }}
        renderer = TSHubRenderer(
            simid="sumo", sensor_config=sensor_config,
            preset=vlm.get("RENDER_PRESET", "480P"),
            resolution=vlm.get("RENDER_RESOLUTION", 1.0),
            scenario_glb_dir=vlm["SCENARIO_GLB_DIR"],
            vehicle_model=vlm.get("VEHICLE_MODEL", "low"),
            render_mode="offscreen", rendering_backend=vlm.get("RENDERING_BACKEND", "pandagl"),
            show_buildings=vlm.get("SHOW_BUILDINGS", False),
            tls_batch_size=None, keep_batch_sensors=True,
            reuse_batch_sensors=False, step_task_manager=True,
            netxml_path=str(Path(paths["PATH_TO_DATA"]) / conf["ROADNET_FILE"]),
            show_arrows=vlm.get("SHOW_ARROWS", True),
        )
        renderer.reset(env.get_tls_init_info(tls_ids))

        mapping_path = vlm.get("DIRECTION_MAPPING_PATH")
        saver = ImageSaver(args.city, direction_mapping_path=mapping_path,
                           base_dir=str(args.output_dir), session_id="", verbose=True,
                           enable_preprocess=False, strict=False)
        total = 0
        for step in range(1, args.steps + 1):
            if step > 1:
                env.traci_conn.simulationStep()
            obs = env.get_tshub_obs(tls_ids=tls_ids,
                                    radius=vlm.get("LOCAL_RENDER_RADIUS_M", 200.0))
            sensor_data = renderer.step(obs, should_count_vehicles=False)
            saved = saver.save_step_images(step, sensor_data, tls_ids=tls_ids)
            count = sum(len(v) for v in saved.values())
            total += count
            print(f"step={step} tls={len(tls_ids)} images={count}")

        files = [p for p in args.output_dir.rglob("*") if p.is_file()
                 and p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        bad = [p for p in files if p.stat().st_size == 0]
        if total == 0 or not files or bad:
            raise RuntimeError(f"render output invalid: total={total} files={len(files)} zero={len(bad)}")
        print(f"TRANSSIMHUB_RENDER_OK files={len(files)} bytes={sum(p.stat().st_size for p in files)}")
        print(f"output_dir={args.output_dir.resolve()}")
        return 0
    finally:
        if renderer is not None:
            for method in ("close", "destroy", "shutdown"):
                fn = getattr(renderer, method, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
                    break
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
