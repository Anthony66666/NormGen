import contextlib
import io
import random
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from train_combined import (
    CombinedInteractionDataset,
    build_generation_timestep_mask,
    capture_rng_state,
    evaluating,
    gradients_are_finite,
    load_model_state_checked,
    load_training_checkpoint,
    save_training_checkpoint,
    trajectory_metric_sums,
    train,
    uses_label_condition,
)


def _write_explicit_prediction_npz(path):
    n, t, v = 1, 4, 2
    trajectories = np.zeros((n, 5, t, v), dtype=np.float32)
    trajectories[:, 0] = np.arange(t, dtype=np.float32)[None, :, None] + 1
    dimensions = np.zeros((n, 2, t, v), dtype=np.float32)
    dimensions[:, 0] = 2.0
    dimensions[:, 1] = 1.0
    timestep_mask = np.ones((n, t, v), dtype=bool)
    history_mask = np.array([[[True, False], [True, False]]])
    future_mask = np.ones((n, 2, v), dtype=bool)
    np.savez(
        path,
        trajectories=trajectories,
        dimensions=dimensions,
        labels=np.zeros((n, v), dtype=np.int64),
        agent_types=np.zeros((n, v), dtype=np.int64),
        map_data=np.zeros((n, 1, 2, 2), dtype=np.float32),
        map_mask=np.ones((n, 1, 2), dtype=bool),
        map_type=np.zeros((n, 1), dtype=np.int64),
        map_speed_limit=np.zeros((n, 1), dtype=np.float32),
        map_names=np.asarray(["toy"]),
        timestep_mask=timestep_mask,
        history_timestep_mask=history_mask,
        future_timestep_mask=future_mask,
        context_agent_mask=np.asarray([[True, False]]),
        future_vehicle_mask=np.asarray([[True, True]]),
        normalization_center=np.asarray([[10.0, 20.0]], dtype=np.float32),
        normalization_scale=np.asarray([50.0], dtype=np.float32),
        forecasting_safe=np.asarray(True),
    )


def _check_explicit_masks_separate_context_from_future_loss(tmp_path):
    path = tmp_path / "explicit.npz"
    _write_explicit_prediction_npz(path)
    dataset = CombinedInteractionDataset(
        path,
        in_channel=5,
        train_mode="prediction",
        history_steps=2,
        future_steps=2,
        prediction_target_steps=4,
        label_source="none",
    )
    assert dataset.target_data.shape == (1, 5, 4, 2)
    assert dataset.context_agent_mask.tolist() == [[True, False]]
    assert dataset.target_vehicle_mask.tolist() == [[True, False]]
    assert dataset.loss_timestep_mask[0, :2, 0].all()
    assert not dataset.loss_timestep_mask[0, :, 1].any()
    assert not dataset.loss_timestep_mask[0, 2:].any()
    assert dataset.dropped_future_only_agents == 1
    assert dataset.dropped_future_only_points == 2
    assert dataset.static_dimensions[0, :, 0].tolist() == [2.0, 1.0]
    assert dataset.static_dimensions[0, :, 1].tolist() == [0.0, 0.0]
    assert dataset.scene_stats.tolist() == [[10.0, 20.0, 50.0]]

    batch = {
        "target_data": dataset.target_data,
        "context_agent_mask": dataset.context_agent_mask,
        "loss_timestep_mask": dataset.loss_timestep_mask,
    }
    generation_mask = build_generation_timestep_mask(
        batch, train_mode="prediction", future_steps=2
    )
    assert generation_mask[0, :2, 0].all()
    assert not generation_mask[0, :, 1].any()
    assert not generation_mask[0, 2:].any()


def _check_legacy_npz_fallback_derives_prediction_context_from_history(tmp_path):
    path = tmp_path / "legacy.npz"
    trajectories = np.zeros((1, 5, 4, 2), dtype=np.float32)
    dimensions = np.zeros((1, 2, 4, 2), dtype=np.float32)
    trajectories[0, 0, 0:4, 0] = 1.0  # history-visible agent
    dimensions[0, :, 0:4, 0] = 1.0
    trajectories[0, 0, 2:4, 1] = 1.0  # future-only agent
    dimensions[0, :, 2:4, 1] = 1.0
    np.savez(
        path,
        trajectories=trajectories,
        dimensions=dimensions,
        labels=np.zeros((1, 2), dtype=np.int64),
        agent_types=np.zeros((1, 2), dtype=np.int64),
        map_data=np.zeros((1, 1, 2, 2), dtype=np.float32),
        map_mask=np.ones((1, 1, 2), dtype=bool),
        map_type=np.zeros((1, 1), dtype=np.int64),
        map_speed_limit=np.zeros((1, 1), dtype=np.float32),
        map_names=np.asarray(["legacy"]),
    )
    with unittest.TestCase().assertRaisesRegex(ValueError, "forecasting_safe=True"):
        CombinedInteractionDataset(
            path,
            in_channel=5,
            train_mode="prediction",
            history_steps=2,
            future_steps=2,
            prediction_target_steps=4,
            label_source="none",
        )
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        dataset = CombinedInteractionDataset(
            path,
            in_channel=5,
            train_mode="prediction",
            history_steps=2,
            future_steps=2,
            prediction_target_steps=4,
            label_source="none",
            allow_legacy_prediction_data=True,
        )
    assert dataset.context_agent_mask.tolist() == [[True, False]]
    assert dataset.target_vehicle_mask.tolist() == [[True, False]]
    assert dataset.dropped_future_only_agents == 1
    assert "HIGH-RISK LEGACY OVERRIDE" in captured.getvalue()


def _check_label_source_none_really_disables_conditioning():
    assert uses_label_condition("initialization", "dataset")
    assert not uses_label_condition("initialization", "none")
    assert not uses_label_condition("prediction", "dataset")


def _check_evaluating_restores_training_mode():
    module = torch.nn.Linear(2, 2)
    module.train()
    with evaluating(module):
        assert not module.training
    assert module.training


def _check_checkpoint_is_strict_and_restores_all_cpu_rng(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = SimpleNamespace(example=True)
    random.seed(7)
    np.random.seed(8)
    torch.manual_seed(9)
    path = tmp_path / "state.pt"
    save_training_checkpoint(path, model, optimizer, None, None, args, 0, 0, 0, 1)
    expected = (random.random(), float(np.random.rand()), torch.rand(3))

    random.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)
    load_training_checkpoint(path, model, optimizer=optimizer, restore_rng=True)
    actual = (random.random(), float(np.random.rand()), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])

    with unittest.TestCase().assertRaisesRegex(RuntimeError, "incompatible"):
        load_model_state_checked(torch.nn.Linear(3, 2), model.state_dict(), path)


def _check_gradient_finite_gate_and_trajectory_metrics():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([float("inf")])
    assert not gradients_are_finite([parameter])
    parameter.grad = torch.tensor([1.0])
    assert gradients_are_finite([parameter])

    target = torch.zeros(1, 5, 2, 1)
    samples = torch.zeros(2, 1, 5, 2, 1)
    samples[0, 0, 0, :, 0] = 1.0
    mask = torch.ones(1, 2, 1, dtype=torch.bool)
    metric = trajectory_metric_sums(
        samples, target, mask, torch.ones(1, 1, dtype=torch.bool), torch.tensor([2.0])
    )
    assert metric["agent_count"].item() == 1
    assert metric["ade_sum"].item() == 2.0
    assert metric["fde_sum"].item() == 2.0
    assert metric["minade_sum"].item() == 0.0
    assert metric["minfde_sum"].item() == 0.0


def _check_tiny_train_writes_final_best_and_machine_readable_metrics(tmp_path):
    import train_combined as training_module

    data_path = tmp_path / "safe.npz"
    _write_explicit_prediction_npz(data_path)

    class TinyGlow(torch.nn.Module):
        def __init__(self, in_channel, **kwargs):
            super().__init__()
            self.in_channel = in_channel
            self.weight = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, input, timestep_mask=None, **kwargs):
            mask = timestep_mask.unsqueeze(1).to(input) if timestep_mask is not None else 1.0
            log_p = -((input * mask * self.weight) ** 2).flatten(1).sum(1)
            return log_p, torch.zeros_like(log_p), [input]

        def reverse(self, z_list, timestep_mask=None, **kwargs):
            batch, _, latent_t, agents = z_list[0].shape
            steps = timestep_mask.shape[1] if timestep_mask is not None else latent_t * 2
            output = self.weight * torch.ones(
                batch, self.in_channel, steps, agents, device=self.weight.device
            )
            if timestep_mask is not None:
                output = output.masked_fill(~timestep_mask.unsqueeze(1), -1.0)
            return output

    args = SimpleNamespace(
        combined_path=str(data_path), val_combined_path=str(data_path),
        in_channel=5, train_mode="prediction", history_steps=2, future_steps=2,
        prediction_target_steps=4, label_source="none", turn_angle_threshold_deg=30.0,
        stationary_dist_threshold=0.0, allow_legacy_prediction_data=False,
        num_workers=0, worker_start_method="", persistent_workers=False, prefetch_factor=0,
        batch=1, val_batch=1, n_flow=1, n_block=1, affine=True, no_lu=False,
        img_size_h=-1, img_size_w=-1, lr=1e-3, amp=False, compile=False,
        resume_path="", resume_model_only=False, allow_partial_checkpoint=False,
        loadckpt=False, load_model_path="", load_optim_path="", no_resume_iter=False,
        start_iter=0, start_epoch=0, epochs=1, max_steps=1, seed=1,
        ddp_find_unused_parameters=False, loss_normalize="valid_dim", label_keep_prob=0.0,
        grad_clip_norm=5.0, temp=0.1, temp_block_decay=1.0, cfg_scale=1.0,
        n_modes=1, sample_interval=0, save_interval=0, save_epoch_interval=0,
        save_step_checkpoints=False, keep_legacy_checkpoints=False,
        val_interval=1, val_num_modes=1, val_max_batches=1,
        log_dir=str(tmp_path / "logs"), ckpt_dir=str(tmp_path / "ckpt"),
        sample_out_dir=str(tmp_path / "samples"), metrics_out_path="",
        vis_position_scale=1.0, save_sample_images=False,
        dist_backend="", deterministic=False,
    )
    with mock.patch.object(training_module, "Glow", TinyGlow), mock.patch.object(
        training_module, "SummaryWriter", None
    ):
        train(0, 1, args)

    checkpoint_dir = tmp_path / "ckpt"
    assert (checkpoint_dir / "last.pt").exists()
    assert (checkpoint_dir / "best.pt").exists()
    records = [
        json.loads(line)
        for line in (checkpoint_dir / "training_metrics.jsonl").read_text().splitlines()
    ]
    assert [record["type"] for record in records] == ["train_step", "validation_epoch"]
    summary = json.loads((checkpoint_dir / "training_summary.json").read_text())
    assert summary["global_step"] == 1
    assert summary["best_val_nll_per_valid_dim"] is not None


class TrainingReliabilityTest(unittest.TestCase):
    def test_explicit_masks_separate_context_from_future_loss(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _check_explicit_masks_separate_context_from_future_loss(Path(tmp_dir))

    def test_legacy_npz_fallback_derives_prediction_context_from_history(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _check_legacy_npz_fallback_derives_prediction_context_from_history(Path(tmp_dir))

    def test_label_source_none_really_disables_conditioning(self):
        _check_label_source_none_really_disables_conditioning()

    def test_evaluating_restores_training_mode(self):
        _check_evaluating_restores_training_mode()

    def test_checkpoint_is_strict_and_restores_all_cpu_rng(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _check_checkpoint_is_strict_and_restores_all_cpu_rng(Path(tmp_dir))

    def test_gradient_finite_gate_and_trajectory_metrics(self):
        _check_gradient_finite_gate_and_trajectory_metrics()

    def test_tiny_train_writes_final_best_and_machine_readable_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _check_tiny_train_writes_final_best_and_machine_readable_metrics(Path(tmp_dir))


if __name__ == "__main__":
    unittest.main()
