import math

import pytest

from v35_offline_grpo.v35_offline_grpo_reward import PHASES, _traffic


def _ground_truth(scores=None, queues=None):
    return {
        "actions": {
            phase: {"remaining_queue_30s": queue}
            for phase, queue in zip(PHASES, queues or [1, 2, 3, 4], strict=True)
        },
        "phase_preference_teacher": {
            "scores": {
                phase: score
                for phase, score in zip(PHASES, scores or [1, 2, 3, 4], strict=True)
            }
        },
    }


def test_deepseek_score_is_selected_phase_score_divided_by_ten():
    scores = [_traffic(_ground_truth([3, 7, 9, 0]), phase) for phase in PHASES]
    assert scores == pytest.approx([0.3, 0.7, 0.9, 0.0])


def test_reward_ignores_sum_reward_actions():
    scores = [_traffic(_ground_truth([0, 0, 0, 10], queues=[100, 1, 1, 1]), phase) for phase in PHASES]
    assert scores == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_missing_deepseek_annotation_returns_zero():
    assert _traffic({"actions": {phase: {"remaining_queue_30s": 0} for phase in PHASES}}, "ETWT") == 0.0


def test_malformed_inputs_return_zero():
    assert _traffic({}, "ETWT") == 0.0
    assert _traffic([], "ETWT") == 0.0
    assert _traffic(_ground_truth([1, 2, 3, 4]), "BAD") == 0.0
    malformed = _ground_truth([1, 2, 3, math.nan])
    assert _traffic(malformed, "ETWT") == 0.0
