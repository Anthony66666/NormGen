import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

import data_preprocess
from data_preprocess import MAP_TYPE_MAPPING, build_scene_tensor


def make_args(**overrides):
    values = {
        "seq_len": 4,
        "history_steps": 2,
        "forecasting_safe": True,
        "max_agents": 3,
        "require_full_trajectory": True,
        "agent_sort": "distance",
        "adaptive_position_scale": True,
        "position_scale": 10.0,
        "scale_margin": 1.0,
        "min_position_scale": 0.5,
        "velocity_scale": 10.0,
        "dimension_scale": 10.0,
        "normalize_yaw": True,
        "compute_labels": True,
        "placeholder_label": 4,
        "stationary_dist_threshold": 0.0,
        "turn_angle_threshold_deg": 30.0,
        "max_lanes": 2,
        "num_points": 3,
        "split": "train",
        "limit_scenes": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def track_rows(track_id, frame_to_x, y=0.0, case_id=1):
    rows = []
    for frame_id, x in frame_to_x.items():
        rows.append(
            {
                "case_id": case_id,
                "track_id": track_id,
                "frame_id": frame_id,
                "timestamp_ms": frame_id * 100,
                "agent_type": "car",
                "x": float(x),
                "y": float(y),
                "vx": 1.0,
                "vy": 0.0,
                "psi_rad": 0.0,
                "length": 4.0,
                "width": 2.0,
            }
        )
    return rows


def raw_map():
    return [
        {
            "coords": np.asarray([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
            "type": MAP_TYPE_MAPPING["centerline"],
            "speed_limit_mps": 10.0,
        },
        {
            "coords": np.asarray([[98.0, 0.0], [100.0, 0.0], [102.0, 0.0]], dtype=np.float32),
            "type": MAP_TYPE_MAPPING["centerline"],
            "speed_limit_mps": 20.0,
        },
    ]


class ForecastingSafeSceneTest(unittest.TestCase):
    def test_future_changes_cannot_change_context_coordinate_system_or_map(self):
        history = track_rows(11, {0: -1.0, 1: 1.0}) + track_rows(
            22, {0: 9.0, 1: 11.0}
        )
        base_future = track_rows(11, {2: 2.0, 3: 3.0}) + track_rows(
            22, {2: 12.0, 3: 13.0}
        )
        changed_future = track_rows(11, {2: 1000.0, 3: 2000.0}) + track_rows(
            22, {2: -3000.0, 3: -4000.0}
        )
        # This agent exists only in the future and must never enter a context slot.
        changed_future += track_rows(99, {2: 5.0, 3: 5.0})

        args = make_args(max_agents=2)
        base, reason = build_scene_tensor(
            pd.DataFrame(history + base_future), "test_map", raw_map(), args
        )
        changed, changed_reason = build_scene_tensor(
            pd.DataFrame(history + changed_future), "test_map", raw_map(), args
        )

        self.assertIsNone(reason)
        self.assertIsNone(changed_reason)
        np.testing.assert_array_equal(base["track_ids"], np.asarray([11, 22]))
        np.testing.assert_array_equal(changed["track_ids"], base["track_ids"])
        np.testing.assert_allclose(
            changed["trajectories"][:, : args.history_steps],
            base["trajectories"][:, : args.history_steps],
        )
        np.testing.assert_allclose(changed["scene_stats"]["mean"], base["scene_stats"]["mean"])
        self.assertEqual(float(changed["scene_stats"]["scale"]), float(base["scene_stats"]["scale"]))
        np.testing.assert_allclose(changed["map_data"], base["map_data"])
        np.testing.assert_array_equal(changed["map_mask"], base["map_mask"])
        np.testing.assert_array_equal(changed["labels"], base["labels"])
        np.testing.assert_array_equal(changed["context_agent_mask"], base["context_agent_mask"])

        # Future values still use the history-derived center/scale instead of
        # silently creating their own future-dependent coordinate system.
        center_x = float(changed["scene_stats"]["mean"][0])
        scale = float(changed["scene_stats"]["scale"])
        np.testing.assert_allclose(
            changed["trajectories"][0, 2, 0],
            (1000.0 - center_x) / scale,
            rtol=1e-6,
            atol=1e-5,
        )
        self.assertFalse(np.allclose(changed["trajectories"][:, 2:], base["trajectories"][:, 2:]))

    def test_history_complete_agents_are_kept_and_missing_future_is_explicit(self):
        rows = []
        rows += track_rows(11, {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0})
        rows += track_rows(22, {0: 10.0, 1: 11.0, 2: 12.0})
        rows += track_rows(33, {0: 20.0, 1: 21.0})
        rows += track_rows(99, {2: 0.0, 3: 0.0})

        scene, reason = build_scene_tensor(
            pd.DataFrame(rows), "test_map", raw_map(), make_args(max_agents=4)
        )

        self.assertIsNone(reason)
        # Distance sorting uses the history-only center, hence track 22 is first.
        np.testing.assert_array_equal(scene["track_ids"], np.asarray([22, 11, 33, -1]))
        np.testing.assert_array_equal(scene["context_agent_mask"], [True, True, True, False])
        np.testing.assert_array_equal(scene["vehicle_mask"], scene["context_agent_mask"])
        np.testing.assert_array_equal(
            scene["history_timestep_mask"],
            np.asarray([[True, True, True, False], [True, True, True, False]]),
        )
        np.testing.assert_array_equal(
            scene["future_timestep_mask"],
            np.asarray([[True, True, False, False], [False, True, False, False]]),
        )
        np.testing.assert_array_equal(scene["future_vehicle_mask"], [True, True, False, False])

    def test_legacy_initialization_reference_remains_available(self):
        rows = track_rows(11, {0: -1.0, 1: 1.0, 2: 100.0, 3: 200.0})
        # This track has full history but not a full initialization trajectory.
        rows += track_rows(22, {0: 10.0, 1: 11.0, 2: 12.0})
        args = make_args(forecasting_safe=False, max_agents=2)

        scene, reason = build_scene_tensor(pd.DataFrame(rows), "test_map", raw_map(), args)

        self.assertIsNone(reason)
        np.testing.assert_array_equal(scene["track_ids"], np.asarray([11, -1]))
        self.assertEqual(scene["scene_stats"]["reference"], "full_sequence")
        self.assertAlmostEqual(float(scene["scene_stats"]["mean"][0]), 75.0)
        np.testing.assert_array_equal(scene["context_agent_mask"], [True, False])


class ForecastingSafeArchiveTest(unittest.TestCase):
    def test_main_saves_masks_and_numeric_normalization_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "raw"
            train_dir = root / "train"
            map_dir = root / "maps"
            output_dir = Path(tmp_dir) / "processed"
            train_dir.mkdir(parents=True)
            map_dir.mkdir(parents=True)

            rows = track_rows(11, {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0})
            pd.DataFrame(rows).to_csv(train_dir / "Tiny_train.csv", index=False)
            (map_dir / "Tiny.osm_xy").write_text("placeholder", encoding="utf-8")

            args = make_args(
                root=str(root),
                output_dir=str(output_dir),
                max_agents=2,
            )
            with mock.patch.object(data_preprocess, "parse_args", return_value=args), mock.patch.object(
                data_preprocess, "build_raw_map_polylines", return_value=raw_map()
            ):
                data_preprocess.main()

            archive_path = output_dir / "interaction_multi_train_combined.npz"
            with np.load(archive_path, allow_pickle=True) as archive:
                expected_keys = {
                    "timestep_mask",
                    "history_timestep_mask",
                    "future_timestep_mask",
                    "vehicle_mask",
                    "context_agent_mask",
                    "future_vehicle_mask",
                    "normalization_center",
                    "normalization_scale",
                    "normalization_velocity_scale",
                    "normalization_dimension_scale",
                    "normalization_yaw_scale",
                    "history_steps",
                    "forecasting_safe",
                }
                self.assertTrue(expected_keys.issubset(set(archive.files)))
                self.assertEqual(archive["timestep_mask"].shape, (1, 4, 2))
                self.assertEqual(archive["history_timestep_mask"].shape, (1, 2, 2))
                self.assertEqual(archive["future_timestep_mask"].shape, (1, 2, 2))
                self.assertEqual(archive["context_agent_mask"].shape, (1, 2))
                self.assertEqual(archive["normalization_center"].shape, (1, 2))
                self.assertEqual(archive["normalization_scale"].shape, (1,))
                self.assertTrue(bool(archive["forecasting_safe"]))
                self.assertEqual(int(archive["history_steps"]), 2)


if __name__ == "__main__":
    unittest.main()
