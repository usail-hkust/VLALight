"""Run the V35-compatible simulator with the dedicated four-video VideoAgent.

This is intentionally a separate entry point. It reuses only the simulator
orchestration and configuration validation from ``run_v35``; the collector's
agent class is replaced before startup.
"""

import run_v35 as _base
import time
from utils.vlm_oneline import VLMOneLine
from models.video_agent import VideoAgent


_original_build_collector = _base.build_collector


def _build_video_collector(args):
    if bool(getattr(args, "resume", False)) and args.work_dir is None:
        raise ValueError("--resume requires the original explicit --work-dir")
    if args.work_dir is None:
        timestamp = time.strftime("%m_%d_%H_%M_%S")
        args.work_dir = f"records/video_agent/{args.dataset}_seed{args.seed}_{timestamp}"
    collector, work_dir = _original_build_collector(args)
    if not bool(getattr(args, "stage1_sumo_only", 0)):
        # V35Collector disables visual inputs for its LLMLight teacher.
        # Re-enable them only when the deployment actually runs visual Stage 1.
        collector._build_agent_input = VLMOneLine._build_agent_input.__get__(
            collector, type(collector)
        )
        collector.dic_traffic_env_conf["SUMO_ONLY_AGENT_INPUT"] = False
        collector.dic_traffic_env_conf["SUMO_ONLY_MODE"] = False
    return collector, work_dir


def main() -> None:
    # run_v35.build_collector resolves the scenario, seed, API settings and
    # recorder configuration. Swap the teacher class at the module boundary so
    # no LLMLight decisions are created.
    _base.LLMLightAgent = VideoAgent
    _base.build_collector = _build_video_collector
    # V35Collector's override validates LLMLight-only fields such as
    # selected_action_idx. VideoAgent has its own compatible decision record,
    # so use the generic VLMOneLine action path for this entry point.
    _base.V35Collector._get_vlm_actions = VLMOneLine._get_vlm_actions
    _base.main()


if __name__ == "__main__":
    main()
