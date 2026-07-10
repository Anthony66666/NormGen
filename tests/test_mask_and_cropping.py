import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

if importlib.util.find_spec("h5py") is None:
    sys.modules["h5py"] = types.ModuleType("h5py")

from MapGlow11_27_original import Glow, ensure_channel_mask, squeeze_channel_mask
from train_combined import crop_time_steps, save_samples_npz
from convert_normgen_to_autobots import (
    crop_prediction_future as crop_autobots_future,
    load_sample_npz as load_autobots_samples,
)


class DummySampler:
    def __init__(self, channels, target_steps, agents):
        self.channels = channels
        self.target_steps = target_steps
        self.agents = agents

    def reverse(self, z_list, *args, **kwargs):
        batch = z_list[0].shape[0]
        timeline = torch.arange(
            self.target_steps,
            dtype=torch.float32,
            device=z_list[0].device,
        ).view(1, 1, self.target_steps, 1)
        return timeline.expand(batch, self.channels, self.target_steps, self.agents).clone()


class MaskPropagationTest(unittest.TestCase):
    @staticmethod
    def _masked_input():
        target = torch.randn(1, 7, 32, 1)
        target[:, :, 30:] = -1.0
        timestep_mask = torch.zeros(1, 32, 1, dtype=torch.bool)
        timestep_mask[:, :30] = True
        vehicle_mask = torch.ones(1, 1, dtype=torch.bool)
        agent_types = torch.ones(1, 1, dtype=torch.long)
        return target, timestep_mask, vehicle_mask, agent_types

    def test_three_blocks_preserve_exact_valid_dimension_count(self):
        timestep_mask = torch.zeros(1, 32, 1, dtype=torch.bool)
        timestep_mask[:, :30] = True
        current = ensure_channel_mask(timestep_mask, channels=7)

        latent_counts = []
        for split in (True, True, False):
            squeezed = squeeze_channel_mask(current)
            if split:
                current, latent_mask = squeezed.chunk(2, dim=1)
            else:
                latent_mask = squeezed
            latent_counts.append(int(latent_mask.sum()))

        self.assertEqual(latent_counts, [105, 52, 53])
        self.assertEqual(sum(latent_counts), 30 * 7)

    def test_masked_three_block_flow_reconstructs_valid_future(self):
        torch.manual_seed(0)
        model = Glow(7, condition_dim=32, n_flow=1, n_block=3, affine=True).eval()
        target, timestep_mask, vehicle_mask, agent_types = self._masked_input()

        with torch.no_grad():
            log_p, logdet, latents = model(
                target,
                agent_types=agent_types,
                target_vehicle_mask=vehicle_mask,
                timestep_mask=timestep_mask,
            )
            reconstructed = model.reverse(
                latents,
                agent_types=agent_types,
                target_vehicle_mask=vehicle_mask,
                timestep_mask=timestep_mask,
                reconstruct=True,
            )
            alternate_padding = target.clone()
            alternate_padding[:, :, 30:] = 123.0
            alternate_log_p, alternate_logdet, _ = model(
                alternate_padding,
                agent_types=agent_types,
                target_vehicle_mask=vehicle_mask,
                timestep_mask=timestep_mask,
            )

        self.assertTrue(torch.isfinite(log_p).all())
        self.assertTrue(torch.isfinite(logdet).all())
        torch.testing.assert_close(alternate_log_p, log_p, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(alternate_logdet, logdet, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            reconstructed[:, :, :30],
            target[:, :, :30],
            rtol=1e-5,
            atol=2e-6,
        )
        self.assertTrue((reconstructed[:, :, 30:] == -1.0).all())

    def test_masked_three_block_backward_is_finite(self):
        torch.manual_seed(1)
        model = Glow(7, condition_dim=32, n_flow=1, n_block=3, affine=True)
        target, timestep_mask, vehicle_mask, agent_types = self._masked_input()
        log_p, logdet, _ = model(
            target,
            agent_types=agent_types,
            target_vehicle_mask=vehicle_mask,
            timestep_mask=timestep_mask,
        )
        loss = -(log_p + logdet).mean()
        loss.backward()

        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_time_crop_requires_enough_steps(self):
        array = np.arange(32)[None, :]
        np.testing.assert_array_equal(crop_time_steps(array, 30, axis=1), array[:, :30])
        with self.assertRaises(ValueError):
            crop_time_steps(array, 33, axis=1)


class PredictionBoundaryTest(unittest.TestCase):
    def _batch(self, batch=2, channels=7, target_steps=32, agents=3):
        target = torch.zeros(batch, channels, target_steps, agents)
        target[:, :, 30:] = -1.0
        timestep_mask = torch.zeros(batch, target_steps, agents, dtype=torch.bool)
        timestep_mask[:, :30] = True
        return {
            "target_data": target,
            "labels": torch.zeros(batch, agents, dtype=torch.long),
            "raw_labels": torch.zeros(batch, agents, dtype=torch.long),
            "scene_stats": torch.ones(batch, 3),
            "map_name": np.array([f"scene_{idx}" for idx in range(batch)]),
            "map_type": torch.zeros(batch, 1, dtype=torch.long),
            "map_speed_limit": torch.zeros(batch, 1),
            "map_data": torch.zeros(batch, 1, 2, 6),
            "map_mask": torch.ones(batch, 1, 2, dtype=torch.bool),
            "agent_types": torch.ones(batch, agents, dtype=torch.long),
            "raw_agent_types": torch.zeros(batch, agents, dtype=torch.long),
            "target_vehicle_mask": torch.ones(batch, agents, dtype=torch.bool),
            "timestep_mask": timestep_mask,
            "history_data": torch.zeros(batch, channels, 10, agents),
            "history_vehicle_mask": torch.ones(batch, agents, dtype=torch.bool),
        }

    def test_prediction_npz_contains_only_real_future(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                batch=2,
                n_modes=2,
                temp=0.7,
                temp_block_decay=1.0,
                train_mode="prediction",
                cfg_scale=1.0,
                sample_out_dir=tmpdir,
                in_channel=7,
                history_steps=10,
                future_steps=30,
                prediction_target_steps=32,
                label_source="none",
                save_sample_images=False,
                sample_key="conditional_samples",
                mode_index=-1,
            )
            model = DummySampler(channels=7, target_steps=32, agents=3)
            save_samples_npz(
                model,
                self._batch(),
                args,
                z_shapes=[(1, 1, 1)],
                device=torch.device("cpu"),
                step=0,
            )

            saved = np.load(Path(tmpdir) / "000001_interaction_combined_samples.npz")
            self.assertEqual(saved["conditional_samples"].shape, (2, 2, 7, 30, 3))
            self.assertEqual(saved["unconditional_samples"].shape, (2, 2, 7, 30, 3))
            self.assertEqual(saved["gt"].shape, (2, 7, 30, 3))
            self.assertEqual(saved["timestep_mask"].shape, (2, 30, 3))
            self.assertEqual(int(saved["saved_target_steps"]), 30)
            self.assertEqual(int(saved["prediction_target_steps"]), 32)
            self.assertTrue(np.all(saved["conditional_samples"][:, :, :, -1] == 29.0))

            autobots_scenes = load_autobots_samples(saved, args)[0]
            self.assertEqual(autobots_scenes.shape, (4, 7, 40, 3))
            self.assertTrue(np.all(autobots_scenes[:, :, -1] == 29.0))

    def test_evaluation_helpers_remove_model_padding(self):
        samples = np.zeros((2, 3, 7, 32, 4), dtype=np.float32)
        self.assertEqual(crop_autobots_future(samples, 30).shape, (2, 3, 7, 30, 4))
        with self.assertRaises(ValueError):
            crop_autobots_future(samples, 33)


if __name__ == "__main__":
    unittest.main()
