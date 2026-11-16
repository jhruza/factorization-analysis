import copy
import json

import pytest
import torch

from factorization.checkpoints import latest_checkpoint, load_checkpoint
from factorization.train import WindowedPlateau, train


@pytest.fixture
def config():
    return dict(
        name="test",
        steps=4,
        batch_size=4,
        sequence_length=9,
        model_seed=7,
        data_seed=103,
        learning_rate=0.001,
        scheduler="windowed_plateau",
        checkpoint_every=1,
        log_every=1,
        model=dict(d_model=12, d_head=4, n_heads=3, n_layers=1, n_ctx=9, d_mlp=24, d_vocab=433),
    )


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


def test_resume_and_freeze_branch_preserve_training_state(tmp_path, config):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        direct, resumed = tmp_path / "direct", tmp_path / "resumed"
        train(config, direct, device="cpu")
        train(config, resumed, device="cpu", until=2)
        parent = latest_checkpoint(resumed)
        before = load_checkpoint(parent)
        train(config, resumed, device="cpu", resume=True)
        expected = load_checkpoint(latest_checkpoint(direct))
        actual = load_checkpoint(latest_checkpoint(resumed))
        assert actual["step"] == 4
        for key in ("model", "optimizer", "scheduler", "torch_rng"):
            assert_tree_equal(expected[key], actual[key])

        # A branch without a freeze must consume the same per-step data and
        # preserve the parent optimizer/scheduler, yielding the same trajectory.
        branch = tmp_path / "branch"
        train(config, branch, device="cpu", parent_checkpoint=parent)
        unfrozen = load_checkpoint(latest_checkpoint(branch))
        for key in ("model", "optimizer", "scheduler", "torch_rng"):
            assert_tree_equal(expected[key], unfrozen[key])
        branch_start = load_checkpoint(branch / "checkpoints" / "step_000002.pt")
        assert_tree_equal(before["optimizer"], branch_start["optimizer"])
        assert_tree_equal(before["scheduler"], branch_start["scheduler"])

        frozen_dir = tmp_path / "frozen"
        train(config, frozen_dir, device="cpu", parent_checkpoint=parent, freeze="attention")
        frozen = load_checkpoint(latest_checkpoint(frozen_dir))
        frozen_start = load_checkpoint(frozen_dir / "checkpoints" / "step_000002.pt")
        assert_tree_equal(before["optimizer"], frozen_start["optimizer"])
        assert_tree_equal(before["scheduler"], frozen_start["scheduler"])
        assert frozen["freeze"] == "attention"
        attention = [
            k for k in before["model"] if ".attn." in k and not k.endswith(("mask", "IGNORE"))
        ]
        assert attention
        assert all(torch.equal(before["model"][k], frozen["model"][k]) for k in attention)
        assert any(
            not torch.equal(before["model"][k], frozen["model"][k])
            for k in before["model"]
            if ".mlp." in k
        )
        metadata = json.loads((frozen_dir / "metadata.json").read_text())
        assert metadata["freeze"] == "attention"
        assert metadata["parent_checkpoint"] == str(parent)
    finally:
        torch.set_num_threads(old_threads)


def test_scheduler_resume_crosses_plateau_reduction():
    def make():
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.AdamW([parameter], lr=0.01)
        return optimizer, WindowedPlateau(optimizer)

    optimizer, scheduler = make()
    for _ in range(10350):
        scheduler.step(1.0)
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_scheduler = copy.deepcopy(scheduler.state_dict())
    for _ in range(150):
        scheduler.step(1.0)
    assert optimizer.param_groups[0]["lr"] < 0.01

    resumed_optimizer, resumed_scheduler = make()
    resumed_optimizer.load_state_dict(saved_optimizer)
    resumed_scheduler.load_state_dict(saved_scheduler)
    for _ in range(150):
        resumed_scheduler.step(1.0)
    assert_tree_equal(optimizer.state_dict(), resumed_optimizer.state_dict())
    assert_tree_equal(scheduler.state_dict(), resumed_scheduler.state_dict())


def test_output_and_config_guards(tmp_path, config):
    path = tmp_path / "run"
    train(config, path, device="cpu", until=0)
    with pytest.raises(FileExistsError):
        train(config, path, device="cpu")
    changed = dict(config, learning_rate=0.002)
    with pytest.raises(ValueError):
        train(changed, path, device="cpu", resume=True)
    with pytest.raises(ValueError):
        train(changed, tmp_path / "branch", device="cpu", parent_checkpoint=latest_checkpoint(path))
    with pytest.raises(ValueError):
        train(config, tmp_path / "negative", device="cpu", until=-1)
