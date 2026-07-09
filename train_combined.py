#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import io
import os
import pickle
import struct
import time
import zipfile
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR
from torch.amp import autocast, GradScaler

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

from MapGlow11_27_original import Glow

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


class _NumpyCoreCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_npz_object_array_compat(npz_path, member_name):
    member_file = f"{member_name}.npy"
    with zipfile.ZipFile(npz_path, "r") as archive:
        with archive.open(member_file, "r") as handle:
            if handle.read(6) != b"\x93NUMPY":
                raise ValueError(f"{member_file} is not a valid .npy payload")

            major, minor = handle.read(2)
            if major == 1:
                header_len = struct.unpack("<H", handle.read(2))[0]
            elif major in (2, 3):
                header_len = struct.unpack("<I", handle.read(4))[0]
            else:
                raise ValueError(f"unsupported npy version {major}.{minor} for {member_file}")

            handle.read(header_len)
            payload = io.BytesIO(handle.read())
            return _NumpyCoreCompatUnpickler(payload).load()


def load_scene_stats(scene_stats_raw, num_scenes):
    scene_stats = []
    for item in scene_stats_raw:
        center = np.asarray(item["mean"], dtype=np.float32)
        scale = np.float32(item["scale"])
        scene_stats.append(np.concatenate([center, np.array([scale], dtype=np.float32)]))
    return np.stack(scene_stats, axis=0).astype(np.float32)


def wrap_to_pi_np(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def infer_target_labels(target_data, target_vehicle_mask, turn_angle_threshold_deg=30.0, stationary_dist_threshold=0.0):
    labels = np.full(target_vehicle_mask.shape, 4, dtype=np.int64)
    turn_threshold = np.deg2rad(float(turn_angle_threshold_deg))

    # target_data uses -1.0 on every channel for padded timesteps.
    timestep_valid = ~np.isclose(target_data[:, :5], -1.0, atol=0.05).all(axis=1)
    num_scenes, _, _, num_agents = target_data.shape
    for scene_idx in range(num_scenes):
        for agent_idx in range(num_agents):
            if not target_vehicle_mask[scene_idx, agent_idx]:
                continue
            valid_t = timestep_valid[scene_idx, :, agent_idx]
            if valid_t.sum() < 2:
                continue

            xy = target_data[scene_idx, :2, valid_t, agent_idx].T
            yaw = target_data[scene_idx, 4, valid_t, agent_idx] * np.pi
            displacement = float(np.linalg.norm(xy[-1] - xy[0]))
            if stationary_dist_threshold > 0.0 and displacement < stationary_dist_threshold:
                labels[scene_idx, agent_idx] = 3
                continue

            yaw_unwrapped = np.unwrap(yaw.astype(np.float64))
            heading_delta = float(wrap_to_pi_np(yaw_unwrapped[-1] - yaw_unwrapped[0]))
            if heading_delta > turn_threshold:
                labels[scene_idx, agent_idx] = 1
            elif heading_delta < -turn_threshold:
                labels[scene_idx, agent_idx] = 2
            else:
                labels[scene_idx, agent_idx] = 0
    return labels


def uses_label_condition(train_mode):
    return train_mode == "initialization"


class CombinedInteractionDataset(Dataset):
    def __init__(
        self,
        combined_path,
        in_channel=7,
        train_mode="initialization",
        history_steps=10,
        future_steps=30,
        prediction_target_steps=32,
        label_source="auto",
        turn_angle_threshold_deg=30.0,
        stationary_dist_threshold=0.0,
    ):
        if in_channel not in (5, 7):
            raise ValueError(f"in_channel must be 5 or 7, got {in_channel}")
        if train_mode not in ("initialization", "prediction"):
            raise ValueError(f"train_mode must be initialization or prediction, got {train_mode}")
        if label_source not in ("auto", "none", "dataset", "target"):
            raise ValueError(f"label_source must be auto, none, dataset, or target, got {label_source}")

        data = np.load(combined_path, allow_pickle=True)

        trajectories = data["trajectories"].astype(np.float32)  # [N, 5, T, V]
        dimensions = data["dimensions"].astype(np.float32)      # [N, 2, T, V]
        labels = data["labels"].astype(np.int64)
        agent_types = data["agent_types"].astype(np.int64)
        map_data = data["map_data"].astype(np.float32)          # [N, L, P, 2]
        map_mask = data["map_mask"].astype(bool)
        map_type = data["map_type"].astype(np.int64)
        map_speed_limit = data["map_speed_limit"].astype(np.float32)
        map_names = data["map_names"]
        scene_stats = None
        try:
            scene_stats_raw = data["scene_stats"]
        except ModuleNotFoundError as exc:
            print(f"[dataset] scene_stats pickle compatibility fallback: {exc}")
            scene_stats_raw = load_npz_object_array_compat(combined_path, "scene_stats")
        try:
            scene_stats = load_scene_stats(scene_stats_raw, trajectories.shape[0])
        except Exception as exc:
            print(f"[dataset] warning: failed to load scene_stats, fallback to zeros. reason: {exc}")
            scene_stats = np.zeros((trajectories.shape[0], 3), dtype=np.float32)

        if in_channel == 7:
            full_data = np.concatenate([trajectories, dimensions], axis=1)
        else:
            full_data = trajectories

        pad_pos = (
            np.isclose(trajectories, 0.0, atol=0.0).all(axis=1)
            & np.isclose(dimensions, 0.0, atol=0.0).all(axis=1)
        )  # [N, T, V]
        full_data = np.where(pad_pos[:, None, :, :], -1.0, full_data)

        target_vehicle_mask = ~pad_pos.all(axis=1)              # [N, V]
        history_vehicle_mask = np.zeros_like(target_vehicle_mask)

        if train_mode == "prediction":
            if history_steps <= 0:
                raise ValueError("prediction mode requires history_steps > 0")
            if future_steps <= 0:
                raise ValueError("prediction mode requires future_steps > 0")
            if prediction_target_steps < future_steps:
                raise ValueError(
                    "prediction_target_steps must be >= future_steps "
                    f"({prediction_target_steps} < {future_steps})"
                )
            if history_steps + future_steps > full_data.shape[2]:
                raise ValueError(
                    "history_steps + future_steps exceeds dataset sequence length: "
                    f"{history_steps} + {future_steps} > {full_data.shape[2]}"
                )

            history_data = full_data[:, :, :history_steps, :]
            future_data = full_data[:, :, history_steps:history_steps + future_steps, :]
            if prediction_target_steps > future_steps:
                pad_shape = (
                    future_data.shape[0],
                    future_data.shape[1],
                    prediction_target_steps - future_steps,
                    future_data.shape[3],
                )
                pad_data = np.full(pad_shape, -1.0, dtype=future_data.dtype)
                target_data = np.concatenate([future_data, pad_data], axis=2)
            else:
                target_data = future_data
            history_vehicle_mask = ~pad_pos[:, :history_steps, :].all(axis=1)
            target_pad_pos = np.isclose(target_data[:, :5], -1.0, atol=0.05).all(axis=1)
            target_vehicle_mask = ~target_pad_pos.all(axis=1)
        else:
            history_data = full_data[:, :, :0, :]
            target_data = full_data

        label_source_used = label_source
        if train_mode == "prediction":
            label_source_used = "none"
        elif label_source_used == "auto":
            label_source_used = "dataset"
        if label_source_used == "target":
            labels = infer_target_labels(
                target_data,
                target_vehicle_mask,
                turn_angle_threshold_deg=turn_angle_threshold_deg,
                stationary_dist_threshold=stationary_dist_threshold,
            )

        self.history_data = torch.from_numpy(history_data)
        self.target_data = torch.from_numpy(target_data)
        self.labels = torch.from_numpy(labels)
        self.agent_types = torch.from_numpy(agent_types)
        self.map_data = torch.from_numpy(map_data)
        self.map_mask = torch.from_numpy(map_mask)
        self.map_type = torch.from_numpy(map_type)
        self.map_speed_limit = torch.from_numpy(map_speed_limit)
        self.scene_stats = torch.from_numpy(scene_stats)
        self.target_vehicle_mask = torch.from_numpy(target_vehicle_mask)
        self.history_vehicle_mask = torch.from_numpy(history_vehicle_mask)
        self.map_names = map_names

        self.input_channels = int(target_data.shape[1])
        self.time_steps = int(target_data.shape[2])
        self.max_agents = int(target_data.shape[3])
        self.max_lanes = int(map_data.shape[1])
        self.max_points = int(map_data.shape[2])
        self.train_mode = train_mode
        self.history_steps = int(history_steps if train_mode == "prediction" else 0)
        self.future_steps = int(future_steps if train_mode == "prediction" else 0)
        self.prediction_target_steps = int(prediction_target_steps if train_mode == "prediction" else 0)
        self.label_source = label_source_used

        valid_agents_per_scene = self.target_vehicle_mask.sum(dim=1).float()
        print("Combined dataset loaded:")
        print(f"  path: {combined_path}")
        print(f"  mode: {self.train_mode}")
        print(f"  label source: {self.label_source}")
        print(f"  samples: {len(self.labels)}")
        print(f"  history shape: {tuple(self.history_data.shape)}")
        print(f"  target shape: {tuple(self.target_data.shape)}")
        print(f"  map shape: {tuple(self.map_data.shape)}")
        print(f"  avg valid agents: {valid_agents_per_scene.mean().item():.2f}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.history_data[idx],
            self.target_data[idx],
            self.labels[idx],
            self.map_data[idx],
            self.map_mask[idx],
            self.agent_types[idx],
            self.scene_stats[idx],
            self.map_names[idx],
            self.target_vehicle_mask[idx],
            self.history_vehicle_mask[idx],
            self.map_type[idx],
            self.map_speed_limit[idx],
        )


def calc_z_shapes(n_channel, input_size_h, input_size_w, n_block):
    divisor = 2 ** int(n_block)
    if input_size_h % divisor != 0:
        raise ValueError(
            f"input_size_h={input_size_h} must be divisible by 2**n_block={divisor}. "
            "For prediction mode use future_steps=30 with prediction_target_steps=32."
        )
    z_shapes = []
    for _ in range(n_block - 1):
        input_size_h //= 2
        z_shapes.append((n_channel, input_size_h, input_size_w))

    input_size_h //= 2
    z_shapes.append((n_channel * 2, input_size_h, input_size_w))
    return z_shapes


def sample_latents(batch_size, z_shapes, device, base_temp, block_decay=1.0):
    temps = [base_temp * (block_decay ** idx) for idx in range(len(z_shapes))]
    return [torch.randn(batch_size, *z, device=device) * temps[idx] for idx, z in enumerate(z_shapes)]


def build_map_features(map_xy, map_mask, map_type):
    """
    map_xy:   [B, L, P, 2]
    map_mask: [B, L, P]
    map_type: [B, L]
    return:   [B, L, P, 6] = [x, y, yaw/pi, type0, type1, type2]
    """
    B, L, P, _ = map_xy.shape
    yaw = torch.zeros(B, L, P, 1, device=map_xy.device, dtype=map_xy.dtype)
    if P > 1:
        diff = map_xy[:, :, 1:, :] - map_xy[:, :, :-1, :]
        yaw_vals = torch.atan2(diff[..., 1], diff[..., 0])
        yaw[:, :, :-1, 0] = yaw_vals
        yaw[:, :, -1, 0] = yaw[:, :, -2, 0]
    yaw = yaw / torch.pi

    type_ids = map_type.long().clamp(min=0, max=2)
    type_onehot = torch.nn.functional.one_hot(type_ids, num_classes=3).to(dtype=map_xy.dtype)
    type_onehot = type_onehot.unsqueeze(2).expand(-1, -1, P, -1)
    type_onehot = type_onehot * map_mask.unsqueeze(-1).float()
    return torch.cat([map_xy, yaw, type_onehot], dim=-1)


def load_yaml_config(config_path, parser):
    if not config_path:
        return {}

    with open(config_path, "r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle) or {}

    if not isinstance(config_data, dict):
        raise ValueError(f"config file must contain a dict: {config_path}")

    valid_keys = {
        action.dest for action in parser._actions
        if action.dest not in (argparse.SUPPRESS, "help")
    }
    unknown_keys = sorted(set(config_data.keys()) - valid_keys)
    if unknown_keys:
        raise ValueError(
            f"config file contains unknown args: {unknown_keys}. "
            "Please keep YAML keys aligned with argparse names."
        )
    return config_data


def safe_load_state(path, map_location="cpu"):
    if path is None:
        return None
    if not os.path.exists(path):
        print(f"[safe_load_state] missing file: {path}")
        return None
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        print(f"[safe_load_state] failed to load {path}: {exc}")
        return None


def build_dataloader(args, is_distributed, local_rank, ngpus_per_node):
    dataset = CombinedInteractionDataset(
        combined_path=args.combined_path,
        in_channel=args.in_channel,
        train_mode=args.train_mode,
        history_steps=args.history_steps,
        future_steps=args.future_steps,
        prediction_target_steps=args.prediction_target_steps,
        label_source=args.label_source,
        turn_angle_threshold_deg=args.turn_angle_threshold_deg,
        stationary_dist_threshold=args.stationary_dist_threshold,
    )

    dataloader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        dataloader_kwargs["multiprocessing_context"] = "spawn"
        dataloader_kwargs["persistent_workers"] = True

    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=ngpus_per_node,
            rank=local_rank,
            shuffle=True,
            drop_last=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch,
            sampler=sampler,
            **dataloader_kwargs,
        )
    else:
        sampler = None
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch,
            shuffle=True,
            **dataloader_kwargs,
        )

    return dataloader, sampler, dataset


def process_batch(batch_raw, device, train_mode="initialization"):
    use_history = train_mode == "prediction"
    history_data = batch_raw[0].to(device, non_blocking=True) if use_history else None
    target_data = batch_raw[1].to(device, non_blocking=True)
    labels = batch_raw[2].to(device, non_blocking=True)
    map_xy = batch_raw[3].to(device, non_blocking=True)
    map_mask = batch_raw[4].to(device, non_blocking=True)
    agent_types = batch_raw[5].to(device, non_blocking=True)
    scene_stats = batch_raw[6].to(device, non_blocking=True)
    map_name = batch_raw[7]
    target_vehicle_mask = batch_raw[8].to(device, non_blocking=True)
    history_vehicle_mask = batch_raw[9].to(device, non_blocking=True) if use_history else None
    map_type = batch_raw[10].to(device, non_blocking=True)
    map_speed_limit = batch_raw[11].to(device, non_blocking=True)
    raw_labels = labels.clone()
    raw_agent_types = agent_types.clone()

    map_data = build_map_features(map_xy, map_mask, map_type)
    timestep_mask = ~torch.isclose(target_data, torch.tensor(-1.0, device=device), atol=0.05).all(dim=1)

    valid_mask = target_vehicle_mask.bool()
    labels = torch.where(valid_mask, labels + 1, torch.zeros_like(labels))
    agent_types = torch.where(valid_mask, agent_types + 1, torch.zeros_like(agent_types))

    return {
        "history_data": history_data,
        "target_data": target_data,
        "labels": labels,
        "raw_labels": raw_labels,
        "map_data": map_data,
        "map_mask": map_mask,
        "agent_types": agent_types,
        "raw_agent_types": raw_agent_types,
        "scene_stats": scene_stats,
        "map_name": map_name,
        "target_vehicle_mask": target_vehicle_mask,
        "history_vehicle_mask": history_vehicle_mask,
        "timestep_mask": timestep_mask,
        "map_type": map_type,
        "map_speed_limit": map_speed_limit,
    }


def build_model_context_kwargs(batch, train_mode="initialization"):
    use_history = train_mode == "prediction"
    model_kwargs = {
        "map_data": batch["map_data"],
        "map_mask": batch["map_mask"],
        "agent_types": batch["agent_types"],
        "target_vehicle_mask": batch["target_vehicle_mask"],
        "timestep_mask": batch["timestep_mask"],
    }
    if use_history:
        model_kwargs["history_data"] = batch["history_data"]
        model_kwargs["history_vehicle_mask"] = batch["history_vehicle_mask"]
    return model_kwargs


def initialize_model_once(model, dataloader, device, is_distributed, local_rank, sampler=None, train_mode="initialization"):
    if sampler is not None:
        sampler.set_epoch(0)

    if (not is_distributed) or local_rank == 0:
        warmup_iter = iter(dataloader)
        batch = process_batch(next(warmup_iter), device, train_mode=train_mode)

        with torch.no_grad():
            condition = batch["labels"] if uses_label_condition(train_mode) else None
            _ = model(
                batch["target_data"],
                condition=condition,
                **build_model_context_kwargs(batch, train_mode=train_mode),
            )

    if is_distributed:
        dist.barrier()
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in model.buffers():
            dist.broadcast(buffer.data, src=0)
        dist.barrier()


def save_samples_npz(model_single, batch, args, z_shapes, device, step):
    batch_size = min(args.batch, batch["target_data"].shape[0])
    sample_labels = batch["labels"][:batch_size]
    sample_raw_labels = batch["raw_labels"][:batch_size]
    sample_scene_stats = batch["scene_stats"][:batch_size]
    sample_map_name = batch["map_name"][:batch_size] if hasattr(batch["map_name"], "__len__") else batch["map_name"]
    sample_gt = batch["target_data"][:batch_size]
    sample_map_type = batch["map_type"][:batch_size]
    sample_map_speed_limit = batch["map_speed_limit"][:batch_size]
    sample_batch = {
        "map_data": batch["map_data"][:batch_size],
        "map_mask": batch["map_mask"][:batch_size],
        "agent_types": batch["agent_types"][:batch_size],
        "raw_agent_types": batch["raw_agent_types"][:batch_size],
        "target_vehicle_mask": batch["target_vehicle_mask"][:batch_size],
        "timestep_mask": batch["timestep_mask"][:batch_size],
        "history_data": (
            batch["history_data"][:batch_size]
            if args.train_mode == "prediction" and batch["history_data"] is not None
            else None
        ),
        "history_vehicle_mask": (
            batch["history_vehicle_mask"][:batch_size]
            if args.train_mode == "prediction" and batch["history_vehicle_mask"] is not None
            else None
        ),
    }
    sample_model_kwargs = build_model_context_kwargs(sample_batch, train_mode=args.train_mode)
    label_condition_used = uses_label_condition(args.train_mode)

    conditional_samples = []
    unconditional_samples = []

    for _ in range(args.n_modes):
        z_cond = sample_latents(batch_size, z_shapes, device, args.temp, args.temp_block_decay)
        if label_condition_used:
            cond_sample = model_single.reverse(
                z_cond,
                sample_labels,
                guidance_scale=args.cfg_scale,
                **sample_model_kwargs,
            ).cpu().data
        else:
            cond_sample = model_single.reverse(
                z_cond,
                **sample_model_kwargs,
            ).cpu().data
        conditional_samples.append(np.array(cond_sample))

        z_uncond = sample_latents(batch_size, z_shapes, device, args.temp, args.temp_block_decay)
        uncond_sample = model_single.reverse(
            z_uncond,
            **sample_model_kwargs,
        ).cpu().data
        unconditional_samples.append(np.array(uncond_sample))

    conditional_samples = np.stack(conditional_samples, axis=0)
    unconditional_samples = np.stack(unconditional_samples, axis=0)

    out_dir = Path(args.sample_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{step + 1:06d}_interaction_combined_samples.npz"
    save_dict = dict(
        conditional_samples=conditional_samples,
        unconditional_samples=unconditional_samples,
        gt=np.array(sample_gt.cpu().data),
        labels=np.array(sample_raw_labels.cpu().data),
        model_labels=np.array(sample_labels.cpu().data),
        maps=np.array(sample_batch["map_data"].cpu().data),
        map_mask=np.array(sample_batch["map_mask"].cpu().data),
        map_type=np.array(sample_map_type.cpu().data),
        map_speed_limit=np.array(sample_map_speed_limit.cpu().data),
        agent_types=np.array(sample_batch["raw_agent_types"].cpu().data),
        model_agent_types=np.array(sample_batch["agent_types"].cpu().data),
        scene_stats=np.array(sample_scene_stats.cpu().data),
        map_name=np.array(sample_map_name),
        target_vehicle_mask=np.array(sample_batch["target_vehicle_mask"].cpu().data),
        timestep_mask=np.array(sample_batch["timestep_mask"].cpu().data),
        n_modes=args.n_modes,
        in_channel=args.in_channel,
        train_mode=np.asarray(args.train_mode),
        use_history=np.bool_(args.train_mode == "prediction"),
        label_condition_used=np.bool_(label_condition_used),
        history_steps=np.int64(args.history_steps),
        future_steps=np.int64(args.future_steps),
        prediction_target_steps=np.int64(args.prediction_target_steps),
        label_source=np.asarray(args.label_source),
    )
    if args.train_mode == "prediction":
        save_dict["history_data"] = np.array(sample_batch["history_data"].cpu().data)
        save_dict["history_vehicle_mask"] = np.array(sample_batch["history_vehicle_mask"].cpu().data)
    np.savez(out_path, **save_dict)
    print(f"[samples] saved to {out_path}")

    if args.save_sample_images:
        save_sample_visualizations(
            out_dir=out_dir,
            step=step,
            conditional_samples=conditional_samples,
            unconditional_samples=unconditional_samples,
            gt=np.array(sample_gt.cpu().data),
            map_data=np.array(sample_batch["map_data"].cpu().data),
            map_mask=np.array(sample_batch["map_mask"].cpu().data),
            map_name=np.array(sample_map_name),
            agent_types=np.array(sample_batch["raw_agent_types"].cpu().data),
            target_vehicle_mask=np.array(sample_batch["target_vehicle_mask"].cpu().data),
            timestep_mask=np.array(sample_batch["timestep_mask"].cpu().data),
            scene_stats=np.array(sample_scene_stats.cpu().data),
            args=args,
        )


MAP_STYLE = {
    "centerline": {"color": "#6f7d8c", "lw": 1.4, "alpha": 0.75},
    "boundary": {"color": "#b9a27a", "lw": 1.0, "alpha": 0.55},
    "crosswalk": {"color": "#d8c58a", "lw": 2.2, "alpha": 0.35},
    "unknown": {"color": "#a0a0a0", "lw": 1.0, "alpha": 0.35},
}

AGENT_TYPE_COLORS = {
    0: "#4c78a8",
    1: "#1f77b4",
    2: "#f58518",
    3: "#54a24b",
    4: "#e45756",
}


def vis_denormalize_map_xy(map_data, position_scale):
    xy = map_data[..., :2].copy()
    return xy * float(position_scale)


def vis_denormalize_traj_xy(traj, position_scale):
    xy = traj[:2].copy()
    valid = ~np.isclose(xy, -1.0, atol=1e-4)
    xy[valid] *= float(position_scale)
    return xy


def vis_infer_map_type(map_data, lane_idx, valid_len):
    if valid_len <= 0 or map_data.shape[-1] < 6:
        return "unknown"
    type_scores = np.mean(map_data[lane_idx, :valid_len, 3:6], axis=0)
    type_idx = int(np.argmax(type_scores))
    if type_idx == 0:
        return "centerline"
    if type_idx == 1:
        return "boundary"
    if type_idx == 2:
        return "crosswalk"
    return "unknown"


def vis_valid_path(traj_xy, agent_idx, timestep_mask):
    path = traj_xy[:, :, agent_idx].T
    if timestep_mask is not None:
        valid_t = timestep_mask[:, agent_idx]
        if not valid_t.any():
            return None
        path = path[valid_t]
    else:
        valid_t = ~(np.isclose(path[:, 0], -1.0, atol=1e-4) & np.isclose(path[:, 1], -1.0, atol=1e-4))
        path = path[valid_t]
        if len(path) == 0:
            return None
    return path


def vis_compute_limits(map_xy, map_mask, traj_list, agent_indices, timestep_mask, pad):
    points = []

    for lane_idx in range(map_xy.shape[0]):
        valid_len = int(map_mask[lane_idx].sum())
        if valid_len > 1:
            points.append(map_xy[lane_idx, :valid_len, :2])

    for traj_xy in traj_list:
        for agent_idx in agent_indices:
            path = vis_valid_path(traj_xy, agent_idx, timestep_mask)
            if path is not None and len(path) > 0:
                points.append(path[:, :2])

    if not points:
        return (-50.0, 50.0), (-50.0, 50.0)

    pts = np.concatenate(points, axis=0)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    radius = max(x_max - x_min, y_max - y_min) * 0.5 + pad
    radius = max(radius, 20.0)
    return (center_x - radius, center_x + radius), (center_y - radius, center_y + radius)


def vis_draw_map(ax, map_data, map_xy, map_mask):
    for lane_idx in range(map_xy.shape[0]):
        valid_len = int(map_mask[lane_idx].sum())
        if valid_len <= 1:
            continue
        lane = map_xy[lane_idx, :valid_len, :2]
        lane_type = vis_infer_map_type(map_data, lane_idx, valid_len)
        style = MAP_STYLE.get(lane_type, MAP_STYLE["unknown"])
        ax.plot(
            lane[:, 0],
            lane[:, 1],
            color=style["color"],
            alpha=style["alpha"],
            linewidth=style["lw"],
            zorder=1,
            solid_capstyle="round",
        )


def vis_draw_traj(ax, traj_xy, agent_indices, timestep_mask, agent_types, color_alpha=0.95):
    for agent_idx in agent_indices:
        path = vis_valid_path(traj_xy, agent_idx, timestep_mask)
        if path is None or path.shape[0] == 0:
            continue

        agent_type = int(agent_types[agent_idx]) if agent_types is not None and agent_idx < len(agent_types) else 0
        color = AGENT_TYPE_COLORS.get(agent_type, "#2060ff")
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.1, alpha=color_alpha, zorder=5)
        ax.scatter(path[0, 0], path[0, 1], color=color, s=18, marker="o", zorder=6, edgecolors="white", linewidths=0.4)
        ax.scatter(path[-1, 0], path[-1, 1], color=color, s=34, marker="*", zorder=7, edgecolors="white", linewidths=0.4)
        if path.shape[0] >= 2:
            ax.annotate(
                "",
                xy=path[-1, :2],
                xytext=path[-2, :2],
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=color_alpha),
                zorder=8,
            )


def scene_position_scale(scene_stats, args):
    if scene_stats is not None and len(scene_stats) >= 3 and np.isfinite(scene_stats[2]) and scene_stats[2] > 0:
        return float(scene_stats[2])
    return float(args.vis_position_scale)


def save_single_visualization(image_path, title, pred_xy, gt_xy, map_data, map_mask, map_name, agent_types, target_vehicle_mask, timestep_mask, scene_stats, args):
    import matplotlib.pyplot as plt

    position_scale = scene_position_scale(scene_stats, args)
    map_xy = vis_denormalize_map_xy(map_data, position_scale)
    pred_xy = vis_denormalize_traj_xy(pred_xy, position_scale)
    gt_xy = vis_denormalize_traj_xy(gt_xy, position_scale) if gt_xy is not None else None

    agent_indices = np.where(target_vehicle_mask.astype(bool))[0]
    traj_list = [pred_xy]
    if gt_xy is not None:
        traj_list.append(gt_xy)
    xlim, ylim = vis_compute_limits(map_xy, map_mask, traj_list, agent_indices, timestep_mask, args.vis_pad)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor("#f5f1e7")
    panels = [
        (axes[0], "Ground Truth", gt_xy),
        (axes[1], title, pred_xy),
    ]

    for ax, panel_title, traj_xy in panels:
        ax.set_facecolor("#fbfaf6")
        vis_draw_map(ax, map_data, map_xy, map_mask)
        if traj_xy is not None:
            vis_draw_traj(ax, traj_xy, agent_indices, timestep_mask, agent_types)
        ax.set_title(f"{panel_title} | {map_name}")
        ax.set_aspect("equal")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.12, linestyle="--", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, dpi=args.vis_dpi, bbox_inches="tight")
    plt.close(fig)


def save_sample_visualizations(
    out_dir,
    step,
    conditional_samples,
    unconditional_samples,
    gt,
    map_data,
    map_mask,
    map_name,
    agent_types,
    target_vehicle_mask,
    timestep_mask,
    scene_stats,
    args,
):
    image_dir = out_dir / f"{step + 1:06d}_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    num_scenes = min(args.vis_num_scenes, gt.shape[0])
    num_modes = min(args.vis_num_modes, conditional_samples.shape[0], unconditional_samples.shape[0])

    for scene_idx in range(num_scenes):
        scene_name = str(map_name[scene_idx]) if np.ndim(map_name) > 0 else str(map_name)
        for mode_idx in range(num_modes):
            cond_path = image_dir / f"scene{scene_idx:03d}_mode{mode_idx:02d}_conditional.png"
            uncond_path = image_dir / f"scene{scene_idx:03d}_mode{mode_idx:02d}_unconditional.png"

            save_single_visualization(
                image_path=cond_path,
                title=f"Conditional | mode {mode_idx}",
                pred_xy=conditional_samples[mode_idx, scene_idx],
                gt_xy=gt[scene_idx],
                map_data=map_data[scene_idx],
                map_mask=map_mask[scene_idx],
                map_name=scene_name,
                agent_types=agent_types[scene_idx],
                target_vehicle_mask=target_vehicle_mask[scene_idx],
                timestep_mask=timestep_mask[scene_idx],
                scene_stats=scene_stats[scene_idx] if scene_stats is not None else None,
                args=args,
            )
            save_single_visualization(
                image_path=uncond_path,
                title=f"Unconditional | mode {mode_idx}",
                pred_xy=unconditional_samples[mode_idx, scene_idx],
                gt_xy=gt[scene_idx],
                map_data=map_data[scene_idx],
                map_mask=map_mask[scene_idx],
                map_name=scene_name,
                agent_types=agent_types[scene_idx],
                target_vehicle_mask=target_vehicle_mask[scene_idx],
                timestep_mask=timestep_mask[scene_idx],
                scene_stats=scene_stats[scene_idx] if scene_stats is not None else None,
                args=args,
            )

    print(f"[samples] saved images to {image_dir}")


def train(gpu, ngpus_per_node, args):
    is_distributed = ngpus_per_node > 1
    local_rank = gpu
    if torch.cuda.is_available():
        if is_distributed:
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if is_distributed:
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
            world_size=ngpus_per_node,
            rank=local_rank,
        )

    is_main = (not is_distributed) or local_rank == 0

    dataloader, sampler, dataset = build_dataloader(args, is_distributed, local_rank, ngpus_per_node)
    args.label_source = dataset.label_source

    if args.img_size_h <= 0:
        args.img_size_h = dataset.time_steps
    elif args.img_size_h != dataset.time_steps:
        raise ValueError(
            f"img_size_h={args.img_size_h} does not match dataset target time steps={dataset.time_steps}. "
            "Use -1 to infer it from the selected train_mode."
        )
    if args.img_size_w <= 0:
        args.img_size_w = dataset.max_agents
    elif args.img_size_w != dataset.max_agents:
        raise ValueError(
            f"img_size_w={args.img_size_w} does not match dataset max agents={dataset.max_agents}. "
            "Use -1 to infer it from the dataset."
        )

    model_single = Glow(
        in_channel=args.in_channel,
        condition_dim=32,
        n_flow=args.n_flow,
        n_block=args.n_block,
        affine=args.affine,
        conv_lu=not args.no_lu,
    )
    model_single.to(device)

    optimizer = torch.optim.AdamW(model_single.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = StepLR(optimizer, step_size=10000, gamma=0.96)

    use_amp = args.amp and torch.cuda.is_available()
    scaler = GradScaler("cuda", enabled=use_amp)

    if args.compile and hasattr(torch, "compile"):
        if is_main:
            print(f"[rank {local_rank}] compiling model with torch.compile")
        model_single = torch.compile(model_single, mode="reduce-overhead")

    if args.loadckpt:
        if args.load_model_path:
            ckpt = safe_load_state(args.load_model_path, map_location="cpu")
            if ckpt is not None:
                model_single.load_state_dict(ckpt, strict=False)
        if args.load_optim_path:
            opt_state = safe_load_state(args.load_optim_path, map_location="cpu")
            if opt_state is not None:
                optimizer.load_state_dict(opt_state)

    initialize_model_once(
        model_single,
        dataloader,
        device,
        is_distributed=is_distributed,
        local_rank=local_rank,
        sampler=sampler,
        train_mode=args.train_mode,
    )

    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            model_single, device_ids=[local_rank], find_unused_parameters=True
        )
    else:
        model = model_single

    z_shapes = calc_z_shapes(args.in_channel, args.img_size_h, args.img_size_w, args.n_block)

    writer = None
    if is_main:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        logdir = Path(args.log_dir) / f"run_interaction_combined_{run_id}"
        logdir.mkdir(parents=True, exist_ok=True)
        if SummaryWriter is not None:
            writer = SummaryWriter(log_dir=str(logdir))
            print(f"[rank {local_rank}] TensorBoard logdir: {logdir}")
        else:
            print("[tensorboard] tensorboard is not installed; scalar logging disabled")

    total_steps = args.iter
    start_iter = args.start_iter
    epoch = 0
    dataloader_iter = iter(dataloader)
    progress = tqdm(range(start_iter, total_steps), ncols=120) if is_main else range(start_iter, total_steps)

    for i in progress:
        if sampler is not None and (i == start_iter or (i - start_iter) % len(dataloader) == 0):
            sampler.set_epoch(epoch)
            epoch += 1
            dataloader_iter = iter(dataloader)

        try:
            batch_raw = next(dataloader_iter)
        except StopIteration:
            if sampler is not None:
                sampler.set_epoch(epoch)
                epoch += 1
            dataloader_iter = iter(dataloader)
            batch_raw = next(dataloader_iter)

        batch = process_batch(batch_raw, device, train_mode=args.train_mode)
        target_data = batch["target_data"]
        labels = batch["labels"]
        map_data = batch["map_data"]
        map_mask = batch["map_mask"]
        agent_types = batch["agent_types"]

        if torch.isnan(map_data).any() or torch.isnan(target_data).any():
            if is_main:
                print(f"[Iter {i}] NaN detected in input tensors, skip batch")
            continue

        use_conditional = uses_label_condition(args.train_mode) and torch.rand(1).item() < args.label_keep_prob
        condition_input = labels if use_conditional else None

        with autocast("cuda", enabled=use_amp):
            log_p, logdet, _ = model(
                target_data,
                condition=condition_input,
                **build_model_context_kwargs(batch, train_mode=args.train_mode),
            )
            loss_value = -(logdet + log_p).mean()

        if (
            torch.isnan(loss_value)
            or torch.isinf(loss_value)
            or torch.isnan(log_p).any()
            or torch.isnan(logdet).any()
        ):
            if is_main:
                print(f"[Iter {i}] NaN/Inf detected, skip batch")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss_value).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale()
        if old_scale <= new_scale:
            scheduler.step()

        if is_main:
            log_p_mean = log_p.mean()
            logdet_mean = logdet.mean()
            progress.set_description(
                f"Loss: {loss_value.item():.5f}; logP: {log_p_mean.item():.5f}; "
                f"logdet: {logdet_mean.item():.5f}; lr: {optimizer.param_groups[0]['lr']:.7f}"
            )
            if writer is not None:
                writer.add_scalar("train/loss", loss_value.item(), i)
                writer.add_scalar("train/log_p", log_p_mean.item(), i)
                writer.add_scalar("train/logdet", logdet_mean.item(), i)

        if is_main and (i % args.sample_interval == 0 or i == start_iter + 20):
            with torch.no_grad():
                save_samples_npz(model_single, batch, args, z_shapes, device, i)

        if is_main and (i % args.save_interval == 0):
            ckpt_dir = Path(args.ckpt_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model_path = ckpt_dir / "model_interaction_combined.pt"
            optim_path = ckpt_dir / "optim_interaction_combined.pt"
            torch.save(model_single.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optim_path)
            print(f"[rank {local_rank}] saved checkpoint: {model_path}")

    if writer is not None:
        writer.close()
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train MapGlow on combined INTERACTION NPZ")
    parser.add_argument("--config", default="", type=str, help="optional YAML config path")
    parser.add_argument(
        "--combined_path",
        default="",
        type=str,
        help="path to combined interaction npz",
    )
    parser.add_argument("--batch", default=8, type=int, help="batch size")
    parser.add_argument("--iter", default=600000, type=int, help="maximum iterations")
    parser.add_argument("--start_iter", default=0, type=int, help="start iteration")
    parser.add_argument("--n_flow", default=16, type=int, help="number of flows per block")
    parser.add_argument("--n_block", default=3, type=int, help="number of blocks")
    parser.add_argument("--no_lu", action="store_true", help="disable LU conv")
    parser.add_argument("--affine", action="store_true", help="use affine coupling")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--img_size_h", default=-1, type=int, help="time dimension, -1 means infer from dataset")
    parser.add_argument("--img_size_w", default=-1, type=int, help="agent dimension, -1 means infer from dataset")
    parser.add_argument("--in_channel", default=7, type=int, choices=[5, 7], help="5=traj only, 7=traj+dimensions")
    parser.add_argument("--train_mode", default="initialization", choices=["initialization", "prediction"],
                        help="initialization trains on full 40-frame target; prediction trains on future target conditioned on history")
    parser.add_argument("--use_history", action=argparse.BooleanOptionalAction, default=False,
                        help="deprecated compatibility flag; --use_history maps to --train_mode prediction")
    parser.add_argument("--history_steps", default=10, type=int, help="history steps used in prediction mode")
    parser.add_argument("--future_steps", default=30, type=int, help="real future steps used in prediction mode")
    parser.add_argument("--prediction_target_steps", default=32, type=int,
                        help="padded target length for prediction mode; must be divisible by 2**n_block")
    parser.add_argument("--label_source", default="auto", choices=["auto", "none", "dataset", "target"],
                        help="label semantics: prediction always uses none; auto=dataset for initialization")
    parser.add_argument("--turn_angle_threshold_deg", default=30.0, type=float,
                        help="heading-change threshold used when inferring target labels")
    parser.add_argument("--stationary_dist_threshold", default=0.0, type=float,
                        help="normalized displacement threshold for stationary target labels; 0 disables stationary override")
    parser.add_argument("--label_keep_prob", default=0.8, type=float, help="probability of keeping label condition")
    parser.add_argument("--temp", default=0.7, type=float, help="sampling temperature")
    parser.add_argument("--temp_block_decay", default=1.0, type=float, help="per-block latent temp decay")
    parser.add_argument("--cfg_scale", default=1.0, type=float, help="CFG scale for label-conditioned sampling")
    parser.add_argument("--n_modes", default=6, type=int, help="number of samples to save per batch")
    parser.add_argument("--num_workers", default=4, type=int, help="dataloader workers")
    parser.add_argument("--sample_interval", default=2000, type=int, help="sample save interval")
    parser.add_argument("--save_interval", default=2000, type=int, help="checkpoint save interval")
    parser.add_argument("--log_dir", default="./runs", type=str, help="tensorboard log dir")
    parser.add_argument("--ckpt_dir", default="./results", type=str, help="checkpoint dir")
    parser.add_argument("--sample_out_dir", default="./results", type=str, help="sample output dir")
    parser.add_argument("--save_sample_images", action=argparse.BooleanOptionalAction, default=True,
                        help="save visualization pngs together with sample npz")
    parser.add_argument("--vis_num_scenes", default=4, type=int, help="number of sampled scenes to visualize")
    parser.add_argument("--vis_num_modes", default=2, type=int, help="number of modes per scene to visualize")
    parser.add_argument("--vis_position_scale", default=50.0, type=float,
                        help="position scale used to denormalize x/y for visualization")
    parser.add_argument("--vis_pad", default=10.0, type=float, help="extra padding in visualization limits")
    parser.add_argument("--vis_dpi", default=180, type=int, help="dpi for saved visualization images")
    parser.add_argument("--load_model_path", default="", type=str, help="optional model checkpoint path")
    parser.add_argument("--load_optim_path", default="", type=str, help="optional optimizer checkpoint path")
    parser.add_argument("--loadckpt", action="store_true", help="load ckpt before training")
    parser.add_argument("--amp", action="store_true", help="enable automatic mixed precision")
    parser.add_argument("--compile", action="store_true", help="enable torch.compile when available")

    config_probe_args, _ = parser.parse_known_args()
    if config_probe_args.config:
        if os.path.exists(config_probe_args.config):
            config_data = load_yaml_config(config_probe_args.config, parser)
            parser.set_defaults(**config_data)
            print(f"[config] loaded: {config_probe_args.config}")
        else:
            print(f"[config] missing config: {config_probe_args.config}, use CLI/defaults")

    args = parser.parse_args()
    if not os.path.exists(args.combined_path):
        raise FileNotFoundError(f"combined dataset not found: {args.combined_path}")

    if args.use_history and args.train_mode == "initialization":
        args.train_mode = "prediction"
        print("[config] --use_history is deprecated; switching train_mode to prediction")
    args.use_history = args.train_mode == "prediction"
    if not 0.0 <= args.label_keep_prob <= 1.0:
        raise ValueError(f"label_keep_prob must be in [0, 1], got {args.label_keep_prob}")

    target_steps_for_mode = (
        args.prediction_target_steps
        if args.train_mode == "prediction"
        else args.img_size_h
    )
    if args.train_mode == "prediction" and target_steps_for_mode % (2 ** args.n_block) != 0:
        raise ValueError(
            "prediction_target_steps must be divisible by 2**n_block. "
            f"Got prediction_target_steps={args.prediction_target_steps}, n_block={args.n_block}."
        )

    print("Training args:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "12349")

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node == 0:
        print("No GPU detected, run in single-process CPU mode.")
        train(0, 1, args)
    elif ngpus_per_node == 1:
        print("Detected 1 GPU, run in single-process CUDA mode.")
        train(0, 1, args)
    else:
        print(f"Detected {ngpus_per_node} GPUs, launching DDP with mp.spawn")
        mp.spawn(train, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


if __name__ == "__main__":
    main()
