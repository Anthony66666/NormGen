import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from train_combined import CombinedInteractionDataset, save_samples_npz
from trajectory_representation import (
    decode_state_deltas_torch,
    encode_state_deltas_np,
)


class IncrementalRepresentationTest(unittest.TestCase):
    def test_round_trip_uses_last_valid_history_state_and_wraps_yaw(self):
        history = np.zeros((1, 5, 3, 1), dtype=np.float32)
        history[0, :, 0, 0] = [0.0, 0.0, 1.0, 0.0, 0.7]
        history[0, :, 1, 0] = [1.0, 0.0, 1.0, 0.0, 0.9]
        history[0, :, 2, 0] = -1.0
        history_mask = np.array([[[True], [True], [False]]])

        future = np.zeros((1, 5, 3, 1), dtype=np.float32)
        future[0, :, 0, 0] = [1.2, 0.1, 1.1, 0.1, -0.95]
        future[0, :, 1, 0] = [1.5, 0.3, 1.2, 0.2, -0.8]
        future[0, :, 2, 0] = [1.9, 0.6, 1.3, 0.3, -0.6]
        future_mask = np.ones((1, 3, 1), dtype=bool)

        deltas = encode_state_deltas_np(
            future, history, future_mask, history_mask
        )
        self.assertAlmostEqual(float(deltas[0, 0, 0, 0]), 0.2, places=6)
        self.assertAlmostEqual(float(deltas[0, 4, 0, 0]), 0.15, places=6)

        decoded = decode_state_deltas_torch(
            torch.from_numpy(deltas),
            torch.from_numpy(history),
            torch.from_numpy(future_mask),
            torch.from_numpy(history_mask),
        )
        torch.testing.assert_close(decoded, torch.from_numpy(future), atol=1e-6, rtol=0)

    def test_missing_future_step_keeps_last_valid_anchor(self):
        history = np.zeros((1, 5, 1, 1), dtype=np.float32)
        history[0, :, 0, 0] = [2.0, 3.0, 0.0, 0.0, 0.0]
        history_mask = np.ones((1, 1, 1), dtype=bool)
        future = np.full((1, 5, 3, 1), -1.0, dtype=np.float32)
        future[0, :, 0, 0] = [2.5, 3.0, 0.1, 0.0, 0.1]
        future[0, :, 2, 0] = [3.5, 4.0, 0.3, 0.2, 0.2]
        future_mask = np.array([[[True], [False], [True]]])

        deltas = encode_state_deltas_np(
            future, history, future_mask, history_mask
        )
        np.testing.assert_allclose(deltas[0, :2, 2, 0], [1.0, 1.0])
        self.assertTrue(np.all(deltas[0, :, 1, 0] == -1.0))

        decoded = decode_state_deltas_torch(
            torch.from_numpy(deltas),
            torch.from_numpy(history),
            torch.from_numpy(future_mask),
            torch.from_numpy(history_mask),
        ).numpy()
        np.testing.assert_allclose(
            decoded[:, :, [0, 2]], future[:, :, [0, 2]], atol=1e-6
        )
        self.assertTrue(np.all(decoded[:, :, 1] == -1.0))

    def test_prediction_dataset_exposes_delta_target_without_padding(self):
        trajectories = np.zeros((1, 5, 40, 1), dtype=np.float32)
        trajectories[0, 0, :, 0] = np.arange(40, dtype=np.float32) / 10.0
        trajectories[0, 2, :, 0] = 0.1
        dimensions = np.ones((1, 2, 40, 1), dtype=np.float32)
        timestep_mask = np.ones((1, 40, 1), dtype=bool)
        scene_stats = np.asarray(
            [{"mean": np.zeros(2, dtype=np.float32), "scale": np.float32(1.0)}],
            dtype=object,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "combined.npz"
            np.savez(
                path,
                trajectories=trajectories,
                dimensions=dimensions,
                timestep_mask=timestep_mask,
                history_timestep_mask=timestep_mask[:, :10],
                future_timestep_mask=timestep_mask[:, 10:],
                context_agent_mask=np.ones((1, 1), dtype=bool),
                forecasting_safe=np.bool_(True),
                labels=np.zeros((1, 1), dtype=np.int64),
                agent_types=np.zeros((1, 1), dtype=np.int64),
                map_data=np.zeros((1, 1, 2, 2), dtype=np.float32),
                map_mask=np.ones((1, 1, 2), dtype=bool),
                map_type=np.zeros((1, 1), dtype=np.int64),
                map_speed_limit=np.zeros((1, 1), dtype=np.float32),
                map_names=np.asarray(["scene"]),
                scene_stats=scene_stats,
            )
            dataset = CombinedInteractionDataset(
                path,
                in_channel=5,
                train_mode="prediction",
                history_steps=10,
                future_steps=30,
                prediction_target_steps=30,
                prediction_representation="delta",
                label_source="none",
            )

        self.assertEqual(tuple(dataset.target_data.shape), (1, 5, 30, 1))
        self.assertTrue(dataset.loss_timestep_mask.all())
        np.testing.assert_allclose(
            dataset.target_data[0, 0, :, 0].numpy(),
            np.full(30, 0.1, dtype=np.float32),
            atol=1e-6,
        )

    def test_saved_prediction_samples_are_decoded_to_absolute_states(self):
        class DeltaSampler:
            def reverse(self, z_list, *args, **kwargs):
                output = torch.zeros(1, 5, 30, 1)
                output[:, 0] = 0.1
                return output

        history = torch.zeros(1, 5, 10, 1)
        history[:, 0, -1] = 1.0
        history_mask = torch.ones(1, 10, 1, dtype=torch.bool)
        future_mask = torch.ones(1, 30, 1, dtype=torch.bool)
        target_delta = torch.zeros(1, 5, 30, 1)
        target_delta[:, 0] = 0.1
        batch = {
            "target_data": target_delta,
            "labels": torch.zeros(1, 1, dtype=torch.long),
            "raw_labels": torch.zeros(1, 1, dtype=torch.long),
            "scene_stats": torch.ones(1, 3),
            "map_name": np.asarray(["scene"]),
            "map_type": torch.zeros(1, 1, dtype=torch.long),
            "map_speed_limit": torch.zeros(1, 1),
            "map_data": torch.zeros(1, 1, 2, 6),
            "map_mask": torch.ones(1, 1, 2, dtype=torch.bool),
            "agent_types": torch.ones(1, 1, dtype=torch.long),
            "raw_agent_types": torch.zeros(1, 1, dtype=torch.long),
            "target_vehicle_mask": torch.ones(1, 1, dtype=torch.bool),
            "context_agent_mask": torch.ones(1, 1, dtype=torch.bool),
            "timestep_mask": future_mask,
            "loss_timestep_mask": future_mask,
            "history_data": history,
            "history_vehicle_mask": torch.ones(1, 1, dtype=torch.bool),
            "history_timestep_mask": history_mask,
            "static_dimensions": None,
        }
        args = SimpleNamespace(
            batch=1,
            n_modes=1,
            temp=0.0,
            temp_block_decay=1.0,
            train_mode="prediction",
            prediction_representation="delta",
            cfg_scale=1.0,
            in_channel=5,
            history_steps=10,
            future_steps=30,
            prediction_target_steps=30,
            label_source="none",
            save_sample_images=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            args.sample_out_dir = tmpdir
            save_samples_npz(
                DeltaSampler(),
                batch,
                args,
                z_shapes=[(1, 1, 1)],
                device=torch.device("cpu"),
                step=0,
            )
            saved = np.load(Path(tmpdir) / "000001_interaction_combined_samples.npz")
            expected_x = 1.0 + 0.1 * np.arange(1, 31)
            np.testing.assert_allclose(
                saved["unconditional_samples"][0, 0, 0, :, 0],
                expected_x,
                atol=2e-6,
            )
            np.testing.assert_allclose(saved["gt"][0, 0, :, 0], expected_x, atol=2e-6)
            self.assertEqual(str(saved["trajectory_representation"]), "absolute")
            self.assertEqual(str(saved["model_output_representation"]), "delta")


if __name__ == "__main__":
    unittest.main()
