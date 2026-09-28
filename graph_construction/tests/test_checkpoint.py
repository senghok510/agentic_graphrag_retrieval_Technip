from __future__ import annotations

from graph_construction.pipeline.checkpoint import init_empty_state, load_state, save_state


def test_checkpoint_round_trip_uses_target_directory(tmp_path):
    path = tmp_path / "nested" / "state.pkl"
    state = init_empty_state()
    state["completed_files"].add("document.pdf")

    save_state(state, path)

    assert path.exists()
    assert not path.with_suffix(".pkl.tmp").exists()
    assert load_state(path)["completed_files"] == {"document.pdf"}


def test_missing_checkpoint_returns_independent_empty_state(tmp_path):
    first = load_state(tmp_path / "missing-a.pkl")
    second = load_state(tmp_path / "missing-b.pkl")

    first["completed_files"].add("one.pdf")

    assert second["completed_files"] == set()
