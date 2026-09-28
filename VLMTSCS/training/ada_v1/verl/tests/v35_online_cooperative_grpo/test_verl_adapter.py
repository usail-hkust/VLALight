from __future__ import annotations

from types import SimpleNamespace

import pytest

from v35_online_cooperative_grpo.online_config import CityConfig, OnlineConfig, SplitConfig
from v35_online_cooperative_grpo.online_rollout import (
    CitySnapshot,
    CityRolloutResult,
    IntersectionObservation,
    IntersectionRolloutResult,
)
from v35_online_cooperative_grpo.online_runtime import OnlineBatchItem, OnlineBatchResult
from v35_online_cooperative_grpo.online_samples import OnlineSampleSpec
from v35_online_cooperative_grpo.stage2_protocol import parse_decision_response
from v35_online_cooperative_grpo.verl_adapter import (
    OnlineVERLCollector,
    _disjoint_decision_masks,
    _online_mode_advantages,
    build_temporal_generation_jobs,
    records_for_verl,
    records_to_data_proto,
    specs_from_batch,
    stage1_snapshots_from_generation,
    validate_verl_records,
    _stage1_target,
)
from v35_online_cooperative_grpo.ray_rollout import RayCityRolloutCoordinator
from v35_online_cooperative_grpo.ray_rollout import _seed_temporal_age_tracker


def _config() -> OnlineConfig:
    return OnlineConfig(
        cities={"jinan": CityConfig("jinan")},
        train=SplitConfig(("jinan",), batch_size=2, seed_base=10),
        val=SplitConfig(("jinan",), batch_size=2, seed_base=20),
    )


def _candidate(rollout_id: int) -> CityRolloutResult:
    rows = []
    for intersection_id, signal in (("a", "ETWT"), ("b", "NTST")):
        response = f"<mode>fast</mode>\n<signal>{signal}</signal>"
        rows.append(
            IntersectionRolloutResult(
                intersection_id=intersection_id,
                forced_mode="fast",
                prompt=f"prompt-{intersection_id}",
                response=response,
                parsed=parse_decision_response(response, forced_mode="fast"),
                reward={
                    "score": float(rollout_id),
                    "global_queue_reward": float(rollout_id),
                    "local_queue_reward": 1.0,
                    "reasoning_cost_reward": 0.0,
                    "format_penalty": 0.0 if intersection_id == "a" else -0.5,
                    "network_reward": 0.4 + 0.01 * rollout_id,
                    "local_score": 0.45 + 0.02 * rollout_id,
                    "local_long_term_score": 0.45 + 0.02 * rollout_id,
                    "local_mean_score": 0.6,
                },
            )
        )
    return CityRolloutResult(
        city="jinan",
        step=6,
        rollout_id=rollout_id,
        results=rows,
        before_queues={"a": 5.0, "b": 4.0},
        after_queues={"a": 4.0, "b": 3.0},
    )


def test_specs_from_dataloader_columns_and_metadata():
    batch = {
        "extra_info": [
            {"city": "jinan", "stream_id": "train:slot:0000", "seed": 10, "ordinal": 0},
            {"city": "jinan", "stream_id": "train:slot:0001", "seed": 11, "ordinal": 1},
        ],
        "data_source": ["v35_online_sumo", "v35_online_sumo"],
    }
    specs = specs_from_batch(batch, config=_config(), split="train")
    assert [spec.sample_id for spec in specs] == ["train:slot:0000:00000000", "train:slot:0001:00000001"]
    assert [spec.seed for spec in specs] == [10, 11]


def test_records_have_six_candidate_group_and_selected_marker():
    spec = OnlineSampleSpec("v35_online_sumo", "jinan", "train:slot:0000", 0, 10, 0)
    item = OnlineBatchItem(spec, SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
    result = OnlineBatchResult(
        items=[item],
        rollouts=[[_candidate(index) for index in range(6)]],
        gdpo_records=[],
        selected_rollout_ids=[4],
    )
    records = records_for_verl(result)
    assert len(records) == 12
    grouped = {}
    for record in records:
        grouped.setdefault(record["uid"], []).append(record)
        assert record["uid"] == record["group_id"]
        assert record["extra_info"]["step"] == 6
    assert {
        key: len(value) for key, value in grouped.items()
    } == {
        "train:slot:0000:00000000:jinan:6:a": 6,
        "train:slot:0000:00000000:jinan:6:b": 6,
    }
    assert sum(record["extra_info"]["selected"] for record in records) == 2
    assert {record["network_group_id"] for record in records} == {
        "train:slot:0000:00000000:jinan:6"
    }
    assert {record["mode_group_id"] for record in records} == {
        "train:slot:0000:00000000:jinan:6:a",
        "train:slot:0000:00000000:jinan:6:b",
    }
    # The nested payload is what trainer_base consumes.  It must use the
    # enriched persistent group ids, not the candidate-local ids.
    assert {
        record["extra_info"]["decision_group_id"] for record in records
    } == {
        "train:slot:0000:00000000:jinan:6:a",
        "train:slot:0000:00000000:jinan:6:b",
    }
    assert {
        record["extra_info"]["mode_group_id"] for record in records
    } == {
        "train:slot:0000:00000000:jinan:6:a",
        "train:slot:0000:00000000:jinan:6:b",
    }
    assert all(record["extra_info"]["forced_mode"] == "fast" for record in records)
    assert all(
        record["extra_info"]["reward_extra_info"]["network_reward"]
        == record["network_reward"]
        for record in records
    )
    assert "format_mean_reward" not in records[0]
    assert "decision_reward" not in records[0]
    validate_verl_records(records, expected_rollouts=6)


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [len(text) % 7 + 1, 2]}


class _CharacterTokenizer:
    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(char) for char in text]}


def test_decision_masks_drop_malformed_spans_and_resolve_nested_overlap():
    tokenizer = _CharacterTokenizer()
    malformed = "<reasoning>queue <signal>ETWT</signal></reasoning>"
    ids = tokenizer(malformed)["input_ids"]

    reasoning, signal, overlap = _disjoint_decision_masks(
        ids, tokenizer, format_valid=False
    )
    assert sum(reasoning) == 0
    assert sum(signal) == 0
    assert overlap == 0

    reasoning, signal, overlap = _disjoint_decision_masks(
        ids, tokenizer, format_valid=True
    )
    assert overlap > 0
    assert sum(a and b for a, b in zip(reasoning, signal, strict=True)) == 0
    assert sum(signal) > 0


def test_records_to_data_proto_keeps_response_block_aligned():
    import pytest

    pytest.importorskip("tensordict")
    spec = OnlineSampleSpec("v35_online_sumo", "jinan", "train:slot:0000", 0, 10, 0)
    item = OnlineBatchItem(spec, SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
    result = OnlineBatchResult(
        items=[item],
        rollouts=[[_candidate(index) for index in range(6)]],
        gdpo_records=[],
        selected_rollout_ids=[0],
    )
    records = records_for_verl(result)
    stage1_mmi = {"video_grid_thw": "stage1-video"}
    for record in records:
        record["perception_sft_multi_modal_inputs"] = stage1_mmi
    data = records_to_data_proto(records, _Tokenizer())
    assert len(data) == len(records)
    assert data.batch["prompts"].shape[0] == len(records)
    assert data.batch["responses"].shape[0] == len(records)
    assert data.batch["response_mask"].shape == data.batch["responses"].shape
    assert len(data.non_tensor_batch["uid"]) == len(records)
    assert all(value == {} for value in data.non_tensor_batch["multi_modal_inputs"])
    assert all(
        value == stage1_mmi
        for value in data.non_tensor_batch["perception_sft_multi_modal_inputs"]
    )


def test_records_to_data_proto_trims_fixed_generation_padding():
    torch = pytest.importorskip("torch")
    TensorDict = pytest.importorskip("tensordict").TensorDict
    spec = OnlineSampleSpec("v35_online_sumo", "jinan", "train:slot:0000", 0, 10, 0)
    item = OnlineBatchItem(spec, SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
    result = OnlineBatchResult(
        items=[item], rollouts=[[_candidate(index) for index in range(6)]],
        gdpo_records=[], selected_rollout_ids=[0],
    )
    records = records_for_verl(result)
    rows, prompt_width, generated_width = len(records), 5, 16
    lengths = torch.tensor([3 + (index % 3) for index in range(rows)], dtype=torch.int64)
    response_mask = torch.arange(generated_width).unsqueeze(0) < lengths.unsqueeze(1)
    prompts = torch.ones((rows, prompt_width), dtype=torch.int64)
    responses = torch.arange(rows * generated_width, dtype=torch.int64).reshape(rows, generated_width)
    input_ids = torch.cat([prompts, responses], dim=1)

    class _Generated:
        def __init__(self):
            self.batch = TensorDict({
                "prompts": prompts,
                "responses": responses,
                "response_mask": response_mask.to(torch.int64),
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "position_ids": torch.arange(prompt_width + generated_width).repeat(rows, 1),
                "old_log_probs": torch.zeros((rows, generated_width)),
            }, batch_size=rows)
            self.meta_info = {}

        def __len__(self):
            return rows

    data = records_to_data_proto(records, _Tokenizer(), generated=_Generated(), jobs=records)
    assert data.batch["responses"].shape == (rows, 5)
    assert data.batch["response_mask"].shape == (rows, 5)
    assert data.batch["input_ids"].shape == (rows, prompt_width + 5)
    assert data.batch["old_log_probs"].shape == (rows, 5)
    assert data.non_tensor_batch["response_len"].tolist() == lengths.tolist()
    for index, length in enumerate(lengths.tolist()):
        assert data.batch["rm_scores"][index, length - 1].item() != 0.0
        assert data.batch["response_mask"][index, length:].sum().item() == 0


def test_collector_uses_explicit_stream_metadata():
    spec = OnlineSampleSpec("v35_online_sumo", "jinan", "train:slot:0000", 0, 10, 0)
    item = OnlineBatchItem(spec, SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
    result = OnlineBatchResult(
        items=[item],
        rollouts=[[_candidate(index) for index in range(6)]],
        gdpo_records=[],
        selected_rollout_ids=[0],
    )

    class _Runtime:
        config = _config()

        def rollout_and_commit_specs(self, specs):
            assert [value.stream_id for value in specs] == ["train:slot:0000"]
            return result

    collector = OnlineVERLCollector(_Runtime(), config=_config(), split="train")
    batch = {
        "extra_info": [{"city": "jinan", "stream_id": "train:slot:0000", "seed": 10, "ordinal": 0}]
    }
    output = collector.collect(batch)
    assert len(output.records) == 12
    assert {record["city"] for record in output.records} == {"jinan"}


def test_invalid_signal_can_fall_back_to_current_phase_for_commit():
    candidate = _candidate(0)
    candidate.results[0] = IntersectionRolloutResult(
        intersection_id="a",
        forced_mode="fast",
        prompt="prompt-a",
        response="<mode>fast</mode>\n<signal>bad</signal>",
        parsed=parse_decision_response("<mode>fast</mode>\n<signal>bad</signal>", forced_mode="fast"),
        reward={"score": -1.0},
    )
    signals = RayCityRolloutCoordinator.signals_from_result(
        candidate,
        fallback_signals={"a": "ETWT", "b": "NTST"},
    )
    assert signals == {"a": "ETWT", "b": "NTST"}


def test_invalid_stage1_generation_uses_sumo_gold_fallback_and_saves_diagnostic(tmp_path):
    class _Generated:
        batch = {"responses": [[1, 2, 3]]}

        def __len__(self):
            return 1

    class _Decoder:
        def decode(self, ids, skip_special_tokens=True):
            assert ids == [1, 2, 3]
            assert skip_special_tokens
            return "not a perception block"

    local = {
        "current_phase": "ETWT",
        "phases": {
            phase: {"v": [1, 2], "q": [0, 1], "dv": 1, "dq": 0, "age": 2}
            for phase in ("ETWT", "NTST", "ELWL", "NLSL")
        },
    }
    item = OnlineBatchItem(
        OnlineSampleSpec("v35_online_sumo", "jinan", "train:slot:0000", 0, 10, 0),
        CitySnapshot(
            city="jinan",
            step=1,
            observations=(
                IntersectionObservation("intersection_1_1", 1, local, {"neighbors": {}}),
            ),
        ),
        SimpleNamespace(),
        tmp_path,
    )
    job = {
        "uid": "sample:jinan:1:intersection_1_1:stage1",
        "sample_index": 0,
        "sample_id": "sample",
        "city": "jinan",
        "step": 1,
        "intersection_id": "intersection_1_1",
    }
    output = stage1_snapshots_from_generation([item], _Generated(), [job], _Decoder())
    assert output[0].snapshot.observations[0].local_perception["current_phase"] == "ETWT"

    diagnostic_dir = tmp_path / "stage1" / "intersection_1_1"
    assert (diagnostic_dir / "invalid_generation.txt").read_text() == "not a perception block"
    assert '"perception_open_tags": 0' in (
        diagnostic_dir / "invalid_generation.json"
    ).read_text()


def test_online_mode_gdpo_normalizes_each_local_reward_dimension(monkeypatch):
    monkeypatch.setenv("V35_MODE_GDPO_REWARD_KEYS", "local_queue_reward,format_penalty")
    records = [
        {"mode_group_id": "u", "local_queue_reward": value, "format_penalty": penalty,
         "signal_valid": True}
        for value, penalty in [(0.0, -1.0), (1.0, -1.0), (2.0, 0.0),
                               (3.0, 0.0), (4.0, 1.0), (5.0, 1.0)]
    ]
    advantages = _online_mode_advantages(records)
    assert sum(advantages) == pytest.approx(0.0, abs=1e-6)
    assert advantages == sorted(advantages)
    assert advantages[0] < 0 < advantages[-1]


def test_online_mode_gdpo_uses_long_term_network_reward(monkeypatch):
    monkeypatch.setenv("V35_MODE_GDPO_REWARD_KEYS", "network_reward")
    records = [
        {
            "mode_group_id": "snapshot:a",
            "network_reward": reward,
            # Opposite immediate trend must not affect the global dimension.
            "global_queue_reward": 100.0 - reward,
            "signal_valid": True,
        }
        for reward in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
    ]
    advantages = _online_mode_advantages(records)
    assert advantages == sorted(advantages)
    assert advantages[0] < 0 < advantages[-1]


def test_online_mode_gdpo_defaults_to_scale_free_local_score(monkeypatch):
    monkeypatch.delenv("V35_MODE_GDPO_REWARD_KEYS", raising=False)
    records = [
        {
            "mode_group_id": "snapshot:a",
            "network_reward": 0.5,
            "local_score": score,
            # The legacy raw queue delta deliberately has the opposite order.
            "local_queue_reward": 10.0 - score,
            "reasoning_cost_reward": 0.0,
            "format_penalty": 0.0,
            "signal_valid": True,
        }
        for score in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
    ]
    advantages = _online_mode_advantages(records)
    assert advantages == sorted(advantages)
    assert advantages[0] < 0 < advantages[-1]


def test_online_mode_gdpo_default_includes_format_penalty(monkeypatch):
    monkeypatch.delenv("V35_MODE_GDPO_REWARD_KEYS", raising=False)
    records = [
        {
            "mode_group_id": "snapshot:a",
            "network_reward": 0.5,
            "local_score": 0.5,
            "reasoning_cost_reward": 0.0,
            "format_penalty": penalty,
            "signal_valid": True,
        }
        for penalty in (-1.0, -0.5, 0.0, -1.0, -0.5, 0.0)
    ]
    advantages = _online_mode_advantages(records)
    assert advantages[0] < advantages[1] < advantages[2]
    assert advantages[3] < advantages[4] < advantages[5]


def test_hierarchical_decision_advantage_combines_network_local_and_raw_format():
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensordict")
    from tensordict import TensorDict

    from verl.protocol import DataProto
    from verl.trainer.ppo.core_algos import AdvantageEstimator
    from verl.trainer.ppo.ray_trainer import compute_advantage

    def make_data(format_penalty: float) -> DataProto:
        network = []
        local = []
        mode_groups = []
        formats = []
        for rollout_id in range(6):
            network.extend([0.40 + 0.05 * rollout_id] * 2)
            local.extend([0.40 + 0.08 * rollout_id, 0.80 - 0.08 * rollout_id])
            mode_groups.extend(["snapshot:a", "snapshot:b"])
            formats.extend([format_penalty if rollout_id == 0 else 0.0, 0.0])
        token_rewards = torch.zeros((12, 2), dtype=torch.float32)
        token_rewards[:, -1] = torch.tensor(network)
        mask = torch.ones_like(token_rewards)
        return DataProto(
            batch=TensorDict(
                {
                    "token_level_rewards": token_rewards,
                    "response_mask": mask,
                    "grpo_advantage_scale": mask.clone(),
                },
                batch_size=12,
            ),
            non_tensor_batch={
                "network_group_id": np.asarray(["snapshot"] * 12, dtype=object),
                "uid": np.asarray(mode_groups, dtype=object),
                "mode_group_id": np.asarray(mode_groups, dtype=object),
                "local_score": np.asarray(local, dtype=np.float32),
                "format_penalty": np.asarray(formats, dtype=np.float32),
            },
        )

    config = {
        "hierarchical_decision_advantage": True,
        "local_advantage_weight": 0.5,
        "format_penalty_weight": 1.0,
    }
    clean = compute_advantage(make_data(0.0), AdvantageEstimator.GRPO, config=config)
    penalized = compute_advantage(make_data(-1.0), AdvantageEstimator.GRPO, config=config)

    # Local credit is subtracted, so the higher-local-score intersection gets
    # lower decision advantage when network credit is shared.
    assert torch.all(clean.batch["advantages"][10] < clean.batch["advantages"][11])
    # Raw format punishment changes only rollout 0 / intersection a by -1.
    difference = penalized.batch["advantages"] - clean.batch["advantages"]
    assert torch.allclose(difference[0], torch.full((2,), -1.0))
    assert torch.allclose(difference[1:], torch.zeros((11, 2)))


def test_record_validation_rejects_inconsistent_network_reward():
    spec = OnlineSampleSpec("v35_online_sumo", "jinan", "train:slot:0000", 0, 10, 0)
    item = OnlineBatchItem(spec, SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
    result = OnlineBatchResult(
        items=[item],
        rollouts=[[_candidate(index) for index in range(6)]],
        gdpo_records=[],
        selected_rollout_ids=[0],
    )
    records = records_for_verl(result)
    records[0]["network_reward"] = 123.0
    with pytest.raises(ValueError, match="inconsistent rewards"):
        validate_verl_records(records, expected_rollouts=6)


def test_future_cycles_use_natural_mode_without_forcing():
    spec = OnlineSampleSpec("v35_online_sumo", "jinan", "train:slot:0000", 0, 10, 0)
    item = OnlineBatchItem(spec, SimpleNamespace(step=6), SimpleNamespace(), SimpleNamespace())
    snapshot = CitySnapshot(
        city="jinan",
        step=7,
        observations=(IntersectionObservation("a", 7, {"phases": {}}, {"neighbors": {}}),),
    )
    prepared = SimpleNamespace(assignments=({"a": "fast"},))
    jobs = build_temporal_generation_jobs([item], [prepared], [[snapshot]], cycle=1)
    assert len(jobs) == 1
    assert jobs[0]["forced_mode"] is None
    assert jobs[0]["deployment"] is True


def test_stage1_target_merges_same_snapshot_sumo_coordination():
    local = {
        "current_phase": "ETWT",
        "phases": {
            phase: {"v": [1, 2], "q": [0, 1], "dv": 1, "dq": 0, "age": 2}
            for phase in ("ETWT", "NTST", "ELWL", "NLSL")
        },
    }
    coordination = {
        phase: {
            movement: {"count": index + 1, "is_boundary": "no"}
            for index, movement in enumerate(movements)
        }
        for phase, movements in {
            "ETWT": ("ET", "WT"), "NTST": ("NT", "ST"),
            "ELWL": ("EL", "WL"), "NLSL": ("NL", "SL"),
        }.items()
    }
    observation = IntersectionObservation(
        "a", 7, local, {"local_coordination": coordination, "neighbors": {}}
    )

    target = _stage1_target(observation)

    assert target["phases"]["ETWT"]["coord"] == coordination["ETWT"]
    assert target["phases"]["NTST"]["v"] == [1, 2]
    assert "coord" not in local["phases"]["ETWT"]


def test_temporal_actor_restores_t0_sumo_age_history():
    simulator = SimpleNamespace()
    row = IntersectionObservation(
        "a",
        7,
        {"phases": {"ETWT": {"age": 99}}},
        {},
        audit_metadata={
            "sumo_age_by_phase": {
                "ETWT": 0, "NTST": 3, "ELWL": 4, "NLSL": 5,
            }
        },
    )
    snapshot = CitySnapshot("jinan", 7, (row,))

    _seed_temporal_age_tracker(simulator, snapshot)

    assert simulator._v35_age_tracker["a"] == {
        "ETWT": 0, "NTST": 3, "ELWL": 4, "NLSL": 5, "__step": 7,
    }
