import json
from pathlib import Path

from v35_online_cooperative_grpo.prompt_builder import (
    BASE_LAYOUT,
    SENDER_MESSAGE_RULES,
    SENDER_LAYOUT,
    build_receiver_prompt,
    build_sender_prompt,
)


def base_messages():
    # Resolve from the repository root so this test is independent of the
    # Windows working-directory encoding and the nested ``ada_v1/verl`` path.
    repo_root = Path(__file__).resolve().parents[5]
    dataset = repo_root / "grpo_v30_offline_local_video_dataset_reduced_pixels" / "train_2000.jsonl"
    row = json.loads(dataset.read_text(encoding="utf-8").splitlines()[0])
    return row["prompt"]


def user_content(messages):
    return next(message["content"] for message in messages if message["role"] == "user")


def test_sender_preserves_baseline_except_layout_and_adds_message_protocol():
    original = user_content(base_messages())
    result = user_content(build_sender_prompt(base_messages()))
    assert BASE_LAYOUT not in result
    assert SENDER_LAYOUT in result
    assert "Online cooperation sender requirement:" in result
    assert result.replace(SENDER_LAYOUT, BASE_LAYOUT).removesuffix(SENDER_MESSAGE_RULES) == original


def test_receiver_keeps_baseline_and_only_appends_current_cycle_context():
    original = base_messages()
    context = "Current-cycle routed coordination messages:\n- Source intersection: intersection_1_1"
    result = build_receiver_prompt(original, context)
    assert user_content(result).startswith(user_content(original))
    assert context in user_content(result)
    assert "Do not output a <message> tag." in user_content(result)
    assert original == base_messages()
