from pathlib import Path

from v35_online_cooperative_grpo.ray_rollout import materialize_rollout_snapshots


class _Master:
    def save_snapshot(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("same t0 state", encoding="utf-8")
        return str(path)


def test_materialize_rollout_snapshots_creates_private_copies(tmp_path):
    paths = materialize_rollout_snapshots(_Master(), tmp_path, num_rollouts=6)
    assert len(paths) == 6
    assert all(Path(path).is_file() for path in paths)
    assert len({Path(path).resolve() for path in paths}) == 6
    assert all(Path(path).read_text(encoding="utf-8") == "same t0 state" for path in paths)
