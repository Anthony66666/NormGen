#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import h5py
import numpy as np


REQUIRED_KEYS = {
    "agents_trajectories": (50, 40, 7),
    "agents_types": (50, 2),
    "metas": (5,),
    "map_paths": (1,),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect a converted AutoBots Interaction HDF5 dataset.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing train_dataset.hdf5/val_dataset.hdf5.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    return parser.parse_args()


def check_shape(name, shape):
    expected_tail = REQUIRED_KEYS[name]
    if len(shape) != len(expected_tail) + 1:
        raise ValueError(f"{name} rank mismatch: expected N+{len(expected_tail)}, got {shape}")
    if tuple(shape[1:]) != expected_tail:
        raise ValueError(f"{name} trailing shape mismatch: expected (*, {expected_tail}), got {shape}")


def inspect_split(dataset_dir, split):
    path = Path(dataset_dir) / f"{split}_dataset.hdf5"
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as handle:
        for key in REQUIRED_KEYS:
            if key not in handle:
                raise KeyError(f"{path} missing dataset {key}")
            check_shape(key, handle[key].shape)

        n = handle["agents_trajectories"].shape[0]
        if n == 0:
            raise ValueError(f"{path} has zero scenes")

        traj = handle["agents_trajectories"][:]
        types = handle["agents_types"][:]
        valid_agents = (traj[:, :, :, 0] != -1).all(axis=2)
        if int(valid_agents.sum()) == 0:
            raise ValueError(f"{path} has no fully valid agents")

        map_paths = [handle["map_paths"][i, 0].decode("utf-8") for i in range(n)]
        missing_maps = [item for item in map_paths if not Path(item).exists()]
        if missing_maps:
            raise FileNotFoundError(f"{path} references missing maps: {missing_maps[:3]}")

        finite_traj = np.isfinite(traj[traj != -1]).all()
        finite_types = np.isfinite(types[types != -1]).all()
        if not bool(finite_traj and finite_types):
            raise ValueError(f"{path} contains non-finite trajectory/type values")

        print(
            f"[ok] {split}: scenes={n} "
            f"valid_agents_min={int(valid_agents.sum(axis=1).min())} "
            f"valid_agents_max={int(valid_agents.sum(axis=1).max())} "
            f"source={handle.attrs.get('source_npz', '')}"
        )


def main():
    args = parse_args()
    for split in args.splits:
        inspect_split(args.dataset_dir, split)


if __name__ == "__main__":
    main()
