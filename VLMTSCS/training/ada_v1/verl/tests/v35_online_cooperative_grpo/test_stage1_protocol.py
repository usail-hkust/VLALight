from v35_online_cooperative_grpo.online_rollout import CitySnapshot, IntersectionObservation
from v35_online_cooperative_grpo.stage1_protocol import (
    apply_stage1_perceptions,
    build_stage1_messages,
    parse_perception,
)


def test_coordination_frame_reads_source_camera_but_labels_target_entry(
    tmp_path, monkeypatch
):
    import sys
    from types import SimpleNamespace
    import v35_online_cooperative_grpo.stage1_protocol as protocol

    reads = []

    class Capture:
        def __init__(self, path):
            reads.append(str(path))
            self.index = 0

        def read(self):
            self.index += 1
            return (True, f"frame-{self.index}") if self.index <= 6 else (False, None)

        def release(self):
            pass

    writes = []
    fake_cv2 = SimpleNamespace(
        VideoCapture=Capture,
        imwrite=lambda path, frame: writes.append((path, frame)) or True,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(
        protocol,
        "coordination_sources",
        lambda city, target_id: [("E", "source", "W", 600.0)],
    )

    result = protocol.extract_coordination_frames(
        city="jinan",
        target_id="target",
        video_paths={"source": {"W": "source-W.mp4"}},
        output_dir=tmp_path,
    )

    assert reads == ["source-W.mp4"]
    assert result == [("E", str(tmp_path / "coordination_frame_E.jpg"))]
    assert writes == [(str(tmp_path / "coordination_frame_E.jpg"), "frame-6")]


def _local(phase="ETWT", value=1):
    return {
        "current_phase": phase,
        "phases": {
            "ETWT": {"v": [value, 0], "q": [0, 0], "dv": 0, "dq": 0, "age": 0, "coord": {}},
            "NTST": {"v": [0, 0], "q": [0, 0], "dv": 0, "dq": 0, "age": 1, "coord": {}},
            "ELWL": {"v": [0, 0], "q": [0, 0], "dv": 0, "dq": 0, "age": 1, "coord": {}},
            "NLSL": {"v": [0, 0], "q": [0, 0], "dv": 0, "dq": 0, "age": 1, "coord": {}},
        },
    }


def test_stage1_messages_keep_media_order_and_parse_exact_schema():
    messages = build_stage1_messages(
        intersection_id="intersection_1_1", current_phase="ETWT",
        ages={"ETWT": 0, "NTST": 1, "ELWL": 2, "NLSL": 3},
        videos=["e.mp4", "w.mp4", "n.mp4", "s.mp4"],
        coordination_frames=[("N", "north.jpg"), ("W", "west.jpg")],
    )
    media = [part for part in messages[1]["content"] if part["type"] != "text"]
    assert [(part["type"], part.get(part["type"])) for part in media] == [
        ("video", "e.mp4"), ("video", "w.mp4"), ("video", "n.mp4"),
        ("video", "s.mp4"), ("image", "north.jpg"), ("image", "west.jpg"),
    ]
    import json
    assert parse_perception(f"<perception>{json.dumps(_local())}</perception>") == _local()


def test_stage1_parser_rejects_missing_and_duplicate_blocks():
    import json
    import pytest

    with pytest.raises(ValueError, match="exactly one perception block"):
        parse_perception(json.dumps(_local()))

    block = f"<perception>{json.dumps(_local())}</perception>"
    with pytest.raises(ValueError, match="exactly one perception block"):
        parse_perception(block + block)


def test_city_barrier_routes_only_completed_stage1_perceptions():
    rows = (
        IntersectionObservation("a", 1, _local(), {
            "neighbors": {"east": {"source_intersection": "b", "upstream_movements": {"ET": {}}}}
        }, "ETWT"),
        IntersectionObservation("b", 1, _local(), {"neighbors": {}}, "ETWT"),
    )
    snapshot = CitySnapshot("jinan", 1, rows)
    routed = apply_stage1_perceptions(snapshot, {"a": _local(value=2), "b": _local(value=7)})
    assert routed.observations[0].cooperative_perception["neighbors"]["east"]["upstream_movements"]["ET"]["v"] == 7
    assert routed.observations[0].local_perception == {
        "current_phase": "ETWT",
        "phases": {
            phase: {key: value for key, value in values.items() if key != "coord"}
            for phase, values in _local(value=2)["phases"].items()
        },
    }
    assert routed.observations[0].cooperative_perception["local_coordination"] == {
        phase: {} for phase in ("ETWT", "NTST", "ELWL", "NLSL")
    }

    import pytest
    with pytest.raises(ValueError, match="city barrier"):
        apply_stage1_perceptions(snapshot, {"a": _local()})


def test_online_sft_target_preserves_stage1_schema():
    from v35_offline_grpo.perception_sft import build_perception_target
    import json

    target = _local(value=4)
    text = build_perception_target(
        {"ground_truth": {"stage1_target": target, "perception_target": {}}}, []
    )
    payload = json.loads(text.split("<perception>\n", 1)[1].rsplit("\n</perception>", 1)[0])
    assert payload == target
    assert "candidate_phases" not in text


def test_forced_mode_never_changes_formal_stage2_prompt():
    from v35_online_cooperative_grpo.stage2_protocol import build_stage2_prompt

    local = _local()
    cooperative = {"local_coordination": {}, "neighbors": []}
    autonomous = build_stage2_prompt(local, cooperative)
    assert build_stage2_prompt(local, cooperative, forced_mode="fast") == autonomous
    assert build_stage2_prompt(local, cooperative, forced_mode="slow") == autonomous
    assert "forced" not in autonomous.lower()


def test_invalid_movement_signal_is_penalized_but_falls_back_for_sumo():
    from v35_online_cooperative_grpo.stage2_protocol import (
        executable_signal,
        parse_decision_response,
    )

    parsed = parse_decision_response("<mode>fast</mode><signal>WL</signal>")

    assert parsed.signal == "WL"
    assert parsed.signal_valid is False
    assert parsed.format_penalty == -1.0
    assert executable_signal(parsed, "NTST") == "NTST"
