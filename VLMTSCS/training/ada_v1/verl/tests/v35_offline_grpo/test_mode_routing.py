import json

import pytest
import torch

from v35_offline_grpo.mode_routing import (
    AdaptiveModeThreshold,
    CalibrationObservation,
    ModeThresholdLogitsProcessor,
    binary_slow_probability,
    extract_prompt_sequence_logprob,
    extract_prompt_sequence_token_logprobs,
    mode_token_layout,
)


class _ComposableTokenizer:
    """Small tokenizer fixture with an explicit mode-token boundary."""

    _tokens = {
        "\n<mode>": [11, 12],
        "</mode>": [13],
        "fast": [14],
        "slow": [15],
        "\n<mode>fast</mode>": [11, 12, 14, 13],
        "\n<mode>slow</mode>": [11, 12, 15, 13],
    }

    def encode(self, text, add_special_tokens=False):
        return list(self._tokens.get(text, []))


def test_binary_probability_normalizes_only_fast_and_slow():
    assert binary_slow_probability(0.0, 0.0) == pytest.approx(0.5)
    assert binary_slow_probability(0.0, 1.0) == pytest.approx(0.7310585786)


def test_mode_layout_requires_composable_candidate_tokenization():
    layout = mode_token_layout(_ComposableTokenizer())

    assert layout["full"]["fast"] == [11, 12, 14, 13]
    assert layout["full"]["slow"] == [11, 12, 15, 13]


def test_extract_prompt_sequence_scores_the_actual_candidate_tokens():
    extra_fields = {
        "prompt_ids": [[101], [11], [12], [15], [13], [0]],
        "prompt_logprobs": [[-1.2], [-0.3], [-0.4], [-0.7], [-0.2], [0.0]],
    }

    values = extract_prompt_sequence_token_logprobs(extra_fields, [15])

    assert values == pytest.approx([-0.7])
    assert extract_prompt_sequence_logprob(extra_fields, [15]) == pytest.approx(-0.7)


def test_extract_prompt_sequence_returns_none_when_the_candidate_is_absent():
    extra_fields = {"prompt_ids": [[101], [11], [0]], "prompt_logprobs": [[-1.2], [-0.3], [0.0]]}

    assert extract_prompt_sequence_token_logprobs(extra_fields, [15]) is None


def test_calibrator_maximizes_utility_not_mode_ratio(tmp_path):
    path = tmp_path / "threshold.json"
    calibrator = AdaptiveModeThreshold(path=str(path), ema_decay=0.0, grid_step=0.1)
    rows = [
        CalibrationObservation(0.2, 0.9, 0.1),
        CalibrationObservation(0.4, 0.8, 0.2),
        CalibrationObservation(0.7, 0.1, 0.9),
        CalibrationObservation(0.8, 0.2, 1.0),
    ]

    metrics = calibrator.update(rows, global_step=10)

    assert 0.4 < calibrator.threshold <= 0.7
    assert metrics["best_window_utility"] == pytest.approx(0.9)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["global_step"] == 10
    assert payload["objective"] == "mean_paired_decision_utility"


def test_calibrator_restores_window_and_threshold(tmp_path):
    path = tmp_path / "threshold.json"
    first = AdaptiveModeThreshold(path=str(path), ema_decay=0.0)
    first.update([CalibrationObservation(0.8, 0.1, 0.9)], global_step=3)

    restored = AdaptiveModeThreshold(path=str(path))

    assert restored.threshold == pytest.approx(first.threshold)
    assert restored.last_step == 3
    assert len(restored.observations) == 1


def test_logits_processor_routes_at_mode_boundary(tmp_path):
    path = tmp_path / "threshold.json"
    path.write_text('{"threshold":0.6}\n', encoding="utf-8")
    processor = ModeThresholdLogitsProcessor(
        mode_open_token_ids=[1, 2],
        fast_token_id=3,
        slow_token_id=4,
        threshold_path=str(path),
    )
    logits = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])

    routed = processor([9, 1, 2], logits)

    assert torch.isneginf(routed[3])
    assert routed[4] == pytest.approx(1.0)


def test_logits_processor_accepts_three_argument_vllm_signature(tmp_path):
    processor = ModeThresholdLogitsProcessor(
        mode_open_token_ids=[1, 2],
        fast_token_id=3,
        slow_token_id=4,
        threshold_path=str(tmp_path / "missing.json"),
    )
    logits = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])

    routed = processor([101, 102], [9, 1, 2], logits)

    assert routed[3] == pytest.approx(1.0)
    assert torch.isneginf(routed[4])


def test_logits_processor_leaves_other_positions_unchanged(tmp_path):
    processor = ModeThresholdLogitsProcessor(
        mode_open_token_ids=[1, 2],
        fast_token_id=3,
        slow_token_id=4,
        threshold_path=str(tmp_path / "missing.json"),
    )
    logits = torch.arange(5, dtype=torch.float32)

    routed = processor([8, 9], logits.clone())

    assert torch.equal(routed, logits)
