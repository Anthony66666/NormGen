#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert NormGen combined/generated NPZ files to AutoBots Interaction HDF5 files."
    )
    parser.add_argument("--input-npz", required=True, help="NormGen combined dataset or sample NPZ.")
    parser.add_argument("--output-dir", required=True, help="Output dataset root for AutoBots.")
    parser.add_argument(
        "--source",
        default="auto",
        choices=["auto", "combined", "samples"],
        help="Input NPZ format. auto detects combined preprocessing files vs generated sample files.",
    )
    parser.add_argument(
        "--sample-key",
        default="conditional_samples",
        choices=["conditional_samples", "unconditional_samples", "gt"],
        help="Which array to use from a sample NPZ.",
    )
    parser.add_argument(
        "--mode-index",
        type=int,
        default=-1,
        help="-1 expands all generated modes as separate scenes; otherwise selects one mode.",
    )
    parser.add_argument(
        "--split-name",
        default="train",
        choices=["train", "val"],
        help="Split name to write when --val-ratio is 0.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="If >0, split scenes into train_dataset.hdf5 and val_dataset.hdf5.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-scenes", type=int, default=-1)
    parser.add_argument("--max-agents", type=int, default=50, help="AutoBots HDF5 agent slots.")
    parser.add_argument("--min-agents", type=int, default=2, help="Drop scenes with fewer complete agents.")
    parser.add_argument("--history-steps", type=int, default=10)
    parser.add_argument("--future-steps", type=int, default=30)
    parser.add_argument("--velocity-scale", type=float, default=15.0)
    parser.add_argument("--dimension-scale", type=float, default=10.0)
    parser.add_argument(
        "--maps-root",
        default="",
        help="Directory containing INTERACTION .osm maps. If omitted, dummy maps are written.",
    )
    parser.add_argument("--map-copy-mode", default="symlink", choices=["symlink", "copy", "dummy"])
    parser.add_argument(
        "--allow-dummy-maps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create minimal maps when real .osm maps are unavailable.",
    )
    return parser.parse_args()


def get_scalar(data, key, default):
    if key not in data.files:
        return default
    value = data[key]
    if np.ndim(value) == 0:
        return value.item()
    return value


def scene_stats_to_array(raw):
    if isinstance(raw, dict) or hasattr(raw, "keys"):
        mean = np.asarray(raw["mean"], dtype=np.float32).reshape(-1)
        return np.array([mean[0], mean[1], float(raw["scale"])], dtype=np.float32)
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    out = np.zeros(3, dtype=np.float32)
    out[: min(3, arr.size)] = arr[:3]
    if abs(float(out[2])) < 1e-6:
        out[2] = 1.0
    return out


def normalize_map_name(name):
    name = str(name)
    name = Path(name).name
    if name.endswith(".osm"):
        return name[:-4]
    if name.endswith(".osm_xy"):
        return name[:-7]
    return name


def valid_mask_from_scene(scene):
    scene = np.asarray(scene)
    if scene.shape[0] >= 5:
        return ~np.isclose(scene[:5], -1.0, atol=0.05).all(axis=0)
    return ~np.isclose(scene[:2], -1.0, atol=0.05).all(axis=0)


def valid_mask_from_combined(traj, dims):
    return ~(
        np.isclose(traj, 0.0, atol=0.0).all(axis=0)
        & np.isclose(dims, 0.0, atol=0.0).all(axis=0)
    )


def denormalize_scene(scene, scene_stats, velocity_scale, dimension_scale):
    out = np.full((scene.shape[2], scene.shape[1], 7), -1.0, dtype=np.float32)
    valid = valid_mask_from_scene(scene)
    scale = float(scene_stats[2]) if abs(float(scene_stats[2])) > 1e-6 else 1.0
    mean = scene_stats[:2].astype(np.float32)

    xy = scene[:2].transpose(2, 1, 0).copy()
    xy = xy * scale + mean.reshape(1, 1, 2)
    vel = scene[2:4].transpose(2, 1, 0).copy() * float(velocity_scale)
    yaw = scene[4].transpose(1, 0).copy() * np.pi

    if scene.shape[0] >= 7:
        length = scene[5].transpose(1, 0).copy() * float(dimension_scale)
        width = scene[6].transpose(1, 0).copy() * float(dimension_scale)
    else:
        length = np.full_like(yaw, 4.5, dtype=np.float32)
        width = np.full_like(yaw, 1.8, dtype=np.float32)

    valid_av = valid.transpose(1, 0)
    out[..., 0:2] = xy
    out[..., 2:4] = vel
    out[..., 4] = yaw
    out[..., 5] = np.where(np.isfinite(length) & (length > 0), length, 4.5)
    out[..., 6] = np.where(np.isfinite(width) & (width > 0), width, 1.8)
    out[~valid_av] = -1.0
    return out


def raw_agent_types(agent_types, valid_agents):
    arr = np.asarray(agent_types).astype(np.int64).copy()
    valid_values = arr[valid_agents] if arr.shape[0] == valid_agents.shape[0] else arr
    valid_values = valid_values[valid_values >= 0]
    if valid_values.size and valid_values.min() >= 1 and valid_values.max() <= 2:
        arr = arr - 1
    return arr


def onehot_agent_types(agent_types, valid_agents, max_agents):
    out = np.full((max_agents, 2), -1.0, dtype=np.float32)
    raw = raw_agent_types(agent_types, valid_agents)
    for idx in np.flatnonzero(valid_agents)[:max_agents]:
        if int(raw[idx]) == 0:
            out[idx] = np.array([1.0, 0.0], dtype=np.float32)
        else:
            out[idx] = np.array([0.0, 1.0], dtype=np.float32)
    return out


def build_dummy_osm(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """<?xml version='1.0' encoding='UTF-8'?>
<osm version='0.6' generator='NormGen'>
  <node id='1' lat='0.0' lon='0.0' />
  <node id='2' lat='0.0' lon='0.001' />
  <way id='1'>
    <nd ref='1' />
    <nd ref='2' />
    <tag k='type' v='road_border' />
  </way>
</osm>
""",
        encoding="utf-8",
    )


def ensure_map(map_name, maps_root, output_maps_dir, copy_mode, allow_dummy):
    output_maps_dir.mkdir(parents=True, exist_ok=True)
    base = normalize_map_name(map_name)
    dst = output_maps_dir / f"{base}.osm"
    if dst.exists() or dst.is_symlink():
        return dst

    src = Path(maps_root) / f"{base}.osm" if maps_root else None
    if copy_mode != "dummy" and src is not None and src.exists():
        if copy_mode == "copy":
            shutil.copy2(src, dst)
        else:
            os.symlink(src.resolve(), dst)
        return dst

    if allow_dummy:
        build_dummy_osm(dst)
        return dst

    raise FileNotFoundError(f"Missing map {base}.osm under {maps_root}")


def load_combined_npz(data, args):
    trajectories = np.asarray(data["trajectories"], dtype=np.float32)
    dimensions = np.asarray(data["dimensions"], dtype=np.float32)
    if trajectories.shape[1] == 5:
        scenes = np.concatenate([trajectories, dimensions], axis=1)
    else:
        scenes = trajectories
    stats = np.stack([scene_stats_to_array(item) for item in data["scene_stats"]], axis=0)
    map_names = np.asarray(data["map_names"]).astype(str)
    agent_types = np.asarray(data["agent_types"], dtype=np.int64)
    valid = np.stack(
        [valid_mask_from_combined(trajectories[i], dimensions[i]).all(axis=0) for i in range(scenes.shape[0])],
        axis=0,
    )
    return scenes, stats, map_names, agent_types, valid


def select_sample_array(data, args):
    key = args.sample_key
    if key == "gt":
        arr = np.asarray(data["gt"], dtype=np.float32)[None]
    else:
        arr = np.asarray(data[key], dtype=np.float32)
    if args.mode_index >= 0:
        arr = arr[args.mode_index : args.mode_index + 1]
    return arr


def crop_prediction_future(samples, future_steps):
    """Strip model-only padding before generated trajectories reach evaluation."""
    future_steps = int(future_steps)
    if samples.ndim != 5:
        raise ValueError(f"Prediction samples must be [K,B,C,T,V], got {samples.shape}")
    if future_steps <= 0 or samples.shape[3] < future_steps:
        raise ValueError(
            f"Cannot select {future_steps} future steps from prediction shape {samples.shape}"
        )
    return samples[:, :, :, :future_steps, :]


def load_sample_npz(data, args):
    samples = select_sample_array(data, args)  # [K,B,C,T,V]
    k_modes, batch, channels, steps, agents = samples.shape
    train_mode = str(get_scalar(data, "train_mode", "initialization"))
    history_steps = int(get_scalar(data, "history_steps", args.history_steps))
    future_steps = int(get_scalar(data, "future_steps", args.future_steps))

    if train_mode == "prediction" or bool(get_scalar(data, "use_history", False)):
        if "history_data" not in data.files:
            raise ValueError("Prediction sample NPZ requires history_data.")
        history = np.asarray(data["history_data"], dtype=np.float32)
        if history.shape[2] < history_steps:
            raise ValueError(
                f"Cannot select {history_steps} history steps from shape {history.shape}"
            )
        history = history[:, :, :history_steps, :]
        future = crop_prediction_future(samples, future_steps)
        scenes = np.concatenate(
            [
                np.broadcast_to(history[None], (k_modes, batch, history.shape[1], history_steps, agents)),
                future,
            ],
            axis=3,
        )
    else:
        scenes = samples[:, :, :, :40, :]

    scenes = scenes.reshape(k_modes * batch, scenes.shape[2], scenes.shape[3], scenes.shape[4])
    stats = np.asarray(data["scene_stats"], dtype=np.float32)
    stats = np.repeat(stats[None], k_modes, axis=0).reshape(k_modes * batch, stats.shape[-1])
    map_names = np.asarray(data["map_name"]).astype(str)
    map_names = np.repeat(map_names[None], k_modes, axis=0).reshape(k_modes * batch)
    agent_types = np.asarray(data["agent_types"], dtype=np.int64)
    agent_types = np.repeat(agent_types[None], k_modes, axis=0).reshape(k_modes * batch, agent_types.shape[-1])
    valid = np.stack([valid_mask_from_scene(scene).all(axis=0) for scene in scenes], axis=0)
    return scenes, stats, map_names, agent_types, valid


def load_normgen_npz(path, args):
    data = np.load(path, allow_pickle=True)
    files = set(data.files)
    source = args.source
    if source == "auto":
        source = "combined" if "trajectories" in files else "samples"
    if source == "combined":
        return load_combined_npz(data, args)
    return load_sample_npz(data, args)


def split_indices(num_scenes, val_ratio, split_name, seed):
    indices = np.arange(num_scenes)
    if val_ratio <= 0.0:
        return {split_name: indices}
    if num_scenes == 1:
        return {"train": indices, "val": indices}
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    num_val = max(1, int(round(num_scenes * val_ratio)))
    num_val = min(num_val, num_scenes - 1)
    return {
        "train": np.sort(indices[num_val:]),
        "val": np.sort(indices[:num_val]),
    }


def write_h5(split_name, indices, scenes, stats, map_names, agent_types, valid_agents, args):
    output_dir = Path(args.output_dir)
    maps_dir = output_dir / "maps"
    out_path = output_dir / f"{split_name}_dataset.hdf5"
    output_dir.mkdir(parents=True, exist_ok=True)

    kept = []
    converted = []
    converted_types = []
    converted_meta = []
    converted_map_paths = []

    for idx in indices:
        valid = valid_agents[idx].copy()
        if int(valid.sum()) < args.min_agents:
            continue
        scene = scenes[idx]
        if scene.shape[1] != 40:
            raise ValueError(f"AutoBots requires 40 steps, got {scene.shape[1]} for scene {idx}")
        traj = denormalize_scene(scene, stats[idx], args.velocity_scale, args.dimension_scale)
        full_agent_valid = valid_mask_from_scene(scene).all(axis=0)
        valid = valid & full_agent_valid
        if int(valid.sum()) < args.min_agents:
            continue

        selected = np.flatnonzero(valid)[: args.max_agents]
        h5_traj = np.full((args.max_agents, 40, 7), -1.0, dtype=np.float32)
        h5_types = np.full((args.max_agents, 2), -1.0, dtype=np.float32)
        for out_slot, src_slot in enumerate(selected):
            h5_traj[out_slot] = traj[src_slot]
            raw_type = raw_agent_types(agent_types[idx], valid)[src_slot]
            h5_types[out_slot] = np.array([1.0, 0.0], dtype=np.float32) if int(raw_type) == 0 else np.array([0.0, 1.0], dtype=np.float32)

        map_path = ensure_map(map_names[idx], args.maps_root, maps_dir, args.map_copy_mode, args.allow_dummy_maps)
        converted.append(h5_traj)
        converted_types.append(h5_types)
        converted_meta.append(np.array([0.0, 0.0, 0.0, 0.0, float(idx)], dtype=np.float32))
        converted_map_paths.append(str(map_path.resolve()).encode("ascii", "ignore"))
        kept.append(int(idx))

    if not converted:
        raise RuntimeError(f"No scenes kept for split {split_name}.")

    with h5py.File(out_path, "w") as handle:
        n = len(converted)
        handle.create_dataset("agents_trajectories", data=np.stack(converted), chunks=(1, args.max_agents, 40, 7), dtype=np.float32)
        handle.create_dataset("agents_types", data=np.stack(converted_types), chunks=(1, args.max_agents, 2), dtype=np.float32)
        handle.create_dataset("metas", data=np.stack(converted_meta), chunks=(1, 5), dtype=np.float32)
        map_ds = handle.create_dataset("map_paths", shape=(n, 1), chunks=(1, 1), dtype="S200")
        map_ds[:, 0] = converted_map_paths
        handle.attrs["source_npz"] = str(Path(args.input_npz).resolve())
        handle.attrs["kept_indices"] = np.asarray(kept, dtype=np.int64)

    print(f"[done] {split_name}: {out_path} scenes={len(converted)}")


def main():
    args = parse_args()
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1).")

    scenes, stats, map_names, agent_types, valid_agents = load_normgen_npz(args.input_npz, args)
    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]
        stats = stats[: args.max_scenes]
        map_names = map_names[: args.max_scenes]
        agent_types = agent_types[: args.max_scenes]
        valid_agents = valid_agents[: args.max_scenes]

    splits = split_indices(len(scenes), args.val_ratio, args.split_name, args.seed)
    print(f"[load] scenes={len(scenes)} source={args.source} output={args.output_dir}")
    for split_name, indices in splits.items():
        if len(indices) == 0:
            continue
        write_h5(split_name, indices, scenes, stats, map_names, agent_types, valid_agents, args)


if __name__ == "__main__":
    main()
