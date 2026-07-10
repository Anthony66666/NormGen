#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import io
import os
import pickle
import random
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

try:
    from torch.amp import autocast as _autocast
    from torch.amp import GradScaler as _GradScaler

    _AMP_REQUIRES_DEVICE_TYPE = True
except ImportError:
    from torch.cuda.amp import autocast as _autocast
    from torch.cuda.amp import GradScaler as _GradScaler

    _AMP_REQUIRES_DEVICE_TYPE = False

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

from MapGlow11_27_original import Glow
from trajectory_representation import (
    decode_state_deltas_torch,
    encode_state_deltas_np,
)

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


def make_grad_scaler(enabled):
    if _AMP_REQUIRES_DEVICE_TYPE:
        return _GradScaler("cuda", enabled=enabled)
    return _GradScaler(enabled=enabled)


def cuda_autocast(enabled):
    if _AMP_REQUIRES_DEVICE_TYPE:
        return _autocast("cuda", enabled=enabled)
    return _autocast(enabled=enabled)


def get_env_int(name, default=None):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def set_random_seed(seed, rank=0):
    if seed is None or seed < 0:
        return
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_model(model):
    model = getattr(model, "module", model)
    return getattr(model, "_orig_mod", model)


def normalize_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    prefixes = ("module.", "_orig_mod.")
    normalized = state_dict
    for prefix in prefixes:
        if normalized and all(isinstance(k, str) and k.startswith(prefix) for k in normalized.keys()):
            normalized = {k[len(prefix):]: v for k, v in normalized.items()}
    return normalized


def torch_load_checkpoint(path, map_location="cpu", weights_only=False):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_training_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    args,
    step,
    epoch,
    epoch_step,
    steps_per_epoch,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    next_epoch_step = int(epoch_step) + 1
    next_epoch = int(epoch)
    if next_epoch_step >= int(steps_per_epoch):
        next_epoch += 1
        next_epoch_step = 0
    payload = {
        "format_version": 3,
        "step": int(step),
        "global_step": int(step),
        "next_iter": int(step) + 1,
        "epoch": int(epoch),
        "epoch_step": int(epoch_step),
        "next_epoch": int(next_epoch),
        "next_epoch_step": int(next_epoch_step),
        "steps_per_epoch": int(steps_per_epoch),
        "model_state": unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "args": vars(args).copy(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(payload, path)


def load_training_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, load_optimizer=True):
    if not path:
        return None
    if not os.path.exists(path):
        print(f"[checkpoint] missing file: {path}")
        return None

    ckpt = torch_load_checkpoint(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        missing, unexpected = model.load_state_dict(
            normalize_state_dict_keys(ckpt["model_state"]),
            strict=False,
        )
        print(f"[checkpoint] loaded model: {path}; missing={len(missing)} unexpected={len(unexpected)}")
        if load_optimizer and optimizer is not None and ckpt.get("optimizer_state") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if load_optimizer and scheduler is not None and ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if load_optimizer and scaler is not None and ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        return ckpt

    missing, unexpected = model.load_state_dict(normalize_state_dict_keys(ckpt), strict=False)
    print(f"[checkpoint] loaded legacy state dict: {path}; missing={len(missing)} unexpected={len(unexpected)}")
    return None


def distributed_bad_flag(local_bad, is_distributed, device):
    flag = torch.tensor(int(bool(local_bad)), device=device, dtype=torch.int32)
    if is_distributed:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def broadcast_tensor_from_rank0_(tensor):
    if tensor.is_contiguous():
        dist.broadcast(tensor, src=0)
        return

    synced = tensor.contiguous()
    dist.broadcast(synced, src=0)
    tensor.copy_(synced)


def distributed_label_condition(train_mode, keep_prob, is_distributed, device):
    if not uses_label_condition(train_mode):
        return False
    keep = torch.rand(1, device=device) < float(keep_prob)
    if is_distributed:
        dist.broadcast(keep, src=0)
    return bool(keep.item())


def valid_dimension_count(batch, channels):
    valid_tv = batch["timestep_mask"].float().sum(dim=(1, 2))
    return (valid_tv * int(channels)).clamp_min(1.0)


class CombinedInteractionDataset(Dataset):
    def __init__(
        self,
        combined_path,
        in_channel=7,
        train_mode="initialization",
        history_steps=10,
        future_steps=30,
        prediction_target_steps=32,
        prediction_representation="absolute",
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
        if prediction_representation not in ("absolute", "delta"):
            raise ValueError(
                "prediction_representation must be absolute or delta, got "
                f"{prediction_representation}"
            )
        if train_mode == "prediction" and prediction_representation == "delta" and in_channel != 5:
            raise ValueError(
                "delta prediction represents the five dynamic trajectory channels; "
                "use in_channel=5"
            )

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
        history_timestep_mask = np.zeros(
            (full_data.shape[0], 0, full_data.shape[3]), dtype=bool
        )
        target_timestep_mask = ~pad_pos

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
            history_timestep_mask = ~pad_pos[:, :history_steps, :]
            future_timestep_mask = ~pad_pos[
                :, history_steps:history_steps + future_steps, :
            ]
            if prediction_target_steps > future_steps:
                pad_shape = (
                    future_data.shape[0],
                    future_data.shape[1],
                    prediction_target_steps - future_steps,
                    future_data.shape[3],
                )
                pad_data = np.full(pad_shape, -1.0, dtype=future_data.dtype)
                target_data = np.concatenate([future_data, pad_data], axis=2)
                target_timestep_mask = np.concatenate(
                    [
                        future_timestep_mask,
                        np.zeros(
                            (
                                future_data.shape[0],
                                prediction_target_steps - future_steps,
                                future_data.shape[3],
                            ),
                            dtype=bool,
                        ),
                    ],
                    axis=1,
                )
            else:
                target_data = future_data
                target_timestep_mask = future_timestep_mask
            history_vehicle_mask = history_timestep_mask.any(axis=1)
            target_vehicle_mask = target_timestep_mask.any(axis=1)
            if prediction_representation == "delta":
                target_data = encode_state_deltas_np(
                    target_data,
                    history_data,
                    target_timestep_mask,
                    history_timestep_mask,
                )
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
        self.history_timestep_mask = torch.from_numpy(history_timestep_mask)
        self.target_timestep_mask = torch.from_numpy(target_timestep_mask)
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
        self.prediction_representation = (
            prediction_representation if train_mode == "prediction" else "absolute"
        )
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
            self.target_timestep_mask[idx],
            self.history_timestep_mask[idx],
        )


def calc_z_shapes(n_channel, input_size_h, input_size_w, n_block):
    divisor = 2 ** int(n_block)
    if input_size_h % divisor != 0:
        raise ValueError(
            f"input_size_h={input_size_h} must be divisible by 2**n_block={divisor}. "
            "Choose prediction_target_steps and n_block so no model padding is required."
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


def crop_time_steps(array, steps, axis):
    """Return exactly the requested leading timesteps along ``axis``."""
    steps = int(steps)
    axis = int(axis) % array.ndim
    available = int(array.shape[axis])
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if available < steps:
        raise ValueError(f"cannot crop {steps} timesteps from axis length {available}")
    slices = [slice(None)] * array.ndim
    slices[axis] = slice(0, steps)
    return array[tuple(slices)]


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


def build_dataloader(args, is_distributed, rank, world_size):
    dataset = CombinedInteractionDataset(
        combined_path=args.combined_path,
        in_channel=args.in_channel,
        train_mode=args.train_mode,
        history_steps=args.history_steps,
        future_steps=args.future_steps,
        prediction_target_steps=args.prediction_target_steps,
        prediction_representation=getattr(args, "prediction_representation", "absolute"),
        label_source=args.label_source,
        turn_angle_threshold_deg=args.turn_angle_threshold_deg,
        stationary_dist_threshold=args.stationary_dist_threshold,
    )

    dataloader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": is_distributed,
    }
    if args.num_workers > 0:
        if args.worker_start_method:
            dataloader_kwargs["multiprocessing_context"] = args.worker_start_method
        dataloader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        if args.prefetch_factor > 0:
            dataloader_kwargs["prefetch_factor"] = args.prefetch_factor

    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
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
    timestep_mask = batch_raw[12].to(device, non_blocking=True)
    history_timestep_mask = (
        batch_raw[13].to(device, non_blocking=True) if use_history else None
    )
    raw_labels = labels.clone()
    raw_agent_types = agent_types.clone()

    map_data = build_map_features(map_xy, map_mask, map_type)
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
        "history_timestep_mask": history_timestep_mask,
        "timestep_mask": timestep_mask,
        "map_type": map_type,
        "map_speed_limit": map_speed_limit,
    }


def decode_prediction_states(states, batch, args, timestep_mask=None):
    representation = getattr(args, "prediction_representation", "absolute")
    if args.train_mode != "prediction" or representation == "absolute":
        return states
    if representation != "delta":
        raise ValueError(f"unsupported prediction representation: {representation}")
    return decode_state_deltas_torch(
        states,
        batch["history_data"],
        batch["timestep_mask"] if timestep_mask is None else timestep_mask,
        batch["history_timestep_mask"],
    )


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
            broadcast_tensor_from_rank0_(param.data)
        for buffer in model.buffers():
            broadcast_tensor_from_rank0_(buffer.data)
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
    sample_history_timestep_mask = batch.get("history_timestep_mask")
    if sample_history_timestep_mask is None and batch.get("history_data") is not None:
        sample_history_timestep_mask = ~torch.isclose(
            batch["history_data"][:, :5],
            torch.tensor(-1.0, device=batch["history_data"].device),
            atol=0.05,
        ).all(dim=1)
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
        "history_timestep_mask": (
            sample_history_timestep_mask[:batch_size]
            if args.train_mode == "prediction" and sample_history_timestep_mask is not None
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
            )
        cond_sample = decode_prediction_states(cond_sample, sample_batch, args).cpu().data
        conditional_samples.append(np.array(cond_sample))

        z_uncond = sample_latents(batch_size, z_shapes, device, args.temp, args.temp_block_decay)
        uncond_sample = model_single.reverse(
            z_uncond,
            **sample_model_kwargs,
        )
        uncond_sample = decode_prediction_states(uncond_sample, sample_batch, args).cpu().data
        unconditional_samples.append(np.array(uncond_sample))

    conditional_samples = np.stack(conditional_samples, axis=0)
    unconditional_samples = np.stack(unconditional_samples, axis=0)
    saved_gt_tensor = decode_prediction_states(sample_gt, sample_batch, args)
    saved_gt = np.array(saved_gt_tensor.cpu().data)
    saved_timestep_mask = np.array(sample_batch["timestep_mask"].cpu().data)
    if args.train_mode == "prediction":
        conditional_samples = crop_time_steps(conditional_samples, args.future_steps, axis=3)
        unconditional_samples = crop_time_steps(unconditional_samples, args.future_steps, axis=3)
        saved_gt = crop_time_steps(saved_gt, args.future_steps, axis=2)
        saved_timestep_mask = crop_time_steps(saved_timestep_mask, args.future_steps, axis=1)

    out_dir = Path(args.sample_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{step + 1:06d}_interaction_combined_samples.npz"
    save_dict = dict(
        conditional_samples=conditional_samples,
        unconditional_samples=unconditional_samples,
        gt=saved_gt,
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
        timestep_mask=saved_timestep_mask,
        n_modes=args.n_modes,
        in_channel=args.in_channel,
        train_mode=np.asarray(args.train_mode),
        use_history=np.bool_(args.train_mode == "prediction"),
        label_condition_used=np.bool_(label_condition_used),
        history_steps=np.int64(args.history_steps),
        future_steps=np.int64(args.future_steps),
        prediction_target_steps=np.int64(args.prediction_target_steps),
        saved_target_steps=np.int64(conditional_samples.shape[3]),
        label_source=np.asarray(args.label_source),
        trajectory_representation=np.asarray("absolute"),
        model_output_representation=np.asarray(
            getattr(args, "prediction_representation", "absolute")
        ),
    )
    if args.train_mode == "prediction":
        save_dict["history_data"] = np.array(sample_batch["history_data"].cpu().data)
        save_dict["history_vehicle_mask"] = np.array(sample_batch["history_vehicle_mask"].cpu().data)
        if sample_batch["history_timestep_mask"] is not None:
            save_dict["history_timestep_mask"] = np.array(
                sample_batch["history_timestep_mask"].cpu().data
            )
    np.savez(out_path, **save_dict)
    print(f"[samples] saved to {out_path}")

    if args.save_sample_images and args.train_mode == "prediction":
        save_sample_visualizations(
            out_dir=out_dir,
            step=step,
            prediction_samples=unconditional_samples,
            gt=saved_gt,
            history_data=(
                np.array(sample_batch["history_data"].cpu().data)
                if args.train_mode == "prediction" and sample_batch["history_data"] is not None
                else None
            ),
            map_data=np.array(sample_batch["map_data"].cpu().data),
            map_mask=np.array(sample_batch["map_mask"].cpu().data),
            map_type=np.array(sample_map_type.cpu().data),
            map_name=np.array(sample_map_name),
            target_vehicle_mask=np.array(sample_batch["target_vehicle_mask"].cpu().data),
            timestep_mask=saved_timestep_mask,
            args=args,
        )


MAP_STYLE = {
    0: {"color": "#111827", "lw": 1.25, "alpha": 0.95, "label": "map centerline"},
    1: {"color": "#9ca3af", "lw": 0.9, "alpha": 0.75, "label": "map boundary"},
    2: {"color": "#f59e0b", "lw": 1.0, "alpha": 0.75, "label": "crosswalk"},
}


def vis_traj_xy(traj):
    return traj[:2].copy()


def vis_infer_map_type(map_data, map_type, lane_idx, valid_len):
    if map_type is not None and lane_idx < len(map_type):
        return int(np.clip(map_type[lane_idx], 0, 2))
    if valid_len <= 0 or map_data.shape[-1] < 6:
        return -1
    type_scores = np.mean(map_data[lane_idx, :valid_len, 3:6], axis=0)
    return int(np.argmax(type_scores))


def vis_valid_path(traj_xy, agent_idx, timestep_mask=None):
    path = traj_xy[:, :, agent_idx].T
    if timestep_mask is not None:
        valid_t = timestep_mask[:, agent_idx]
        if not valid_t.any():
            return None
        path = path[valid_t]
    else:
        valid_t = ~(
            np.isclose(path[:, 0], -1.0, atol=0.05)
            & np.isclose(path[:, 1], -1.0, atol=0.05)
        )
        path = path[valid_t]
        if len(path) == 0:
            return None
    path = path[np.isfinite(path).all(axis=1)]
    return path if len(path) > 0 else None


def vis_compute_limits(map_xy, map_mask, limit_paths, pad):
    points = []

    for lane_idx in range(map_xy.shape[0]):
        valid_len = int(map_mask[lane_idx].sum())
        if valid_len > 1:
            points.append(map_xy[lane_idx, :valid_len, :2])

    points.extend(path[:, :2] for path in limit_paths if path is not None and len(path) > 0)
    points.append(np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32))

    if not points:
        return (-1.1, 1.1), (-1.1, 1.1)

    pts = np.concatenate(points, axis=0)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return (-1.1, 1.1), (-1.1, 1.1)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    span = np.maximum(np.array([x_max - x_min, y_max - y_min]), 0.25)
    pad_ratio = float(pad)
    if pad_ratio < 0.0 or pad_ratio >= 1.0:
        pad_ratio = 0.08
    pad_xy = span * pad_ratio
    return (float(x_min - pad_xy[0]), float(x_max + pad_xy[0])), (
        float(y_min - pad_xy[1]),
        float(y_max + pad_xy[1]),
    )


def vis_draw_map(ax, map_data, map_xy, map_mask, map_type):
    for lane_idx in range(map_xy.shape[0]):
        valid_len = int(map_mask[lane_idx].sum())
        if valid_len <= 1:
            continue
        lane = map_xy[lane_idx, :valid_len, :2]
        lane_type = vis_infer_map_type(map_data, map_type, lane_idx, valid_len)
        style = MAP_STYLE.get(lane_type, {"color": "#9ca3af", "lw": 0.8, "alpha": 0.6, "label": "map"})
        ax.plot(
            lane[:, 0],
            lane[:, 1],
            color=style["color"],
            alpha=style["alpha"],
            linewidth=style["lw"],
            zorder=1,
            solid_capstyle="round",
        )


def vis_draw_paths(ax, traj_xy, agent_indices, timestep_mask, color, label, linestyle="-", linewidth=2.0, alpha=0.95):
    first_line = None
    for agent_idx in agent_indices:
        path = vis_valid_path(traj_xy, agent_idx, timestep_mask)
        if path is None or path.shape[0] < 2:
            continue
        line = ax.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            zorder=4,
            solid_capstyle="round",
        )[0]
        first_line = first_line or line
        ax.scatter(
            path[0, 0],
            path[0, 1],
            s=16,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
        )
        ax.scatter(
            path[-1, 0],
            path[-1, 1],
            s=26,
            marker="x",
            color=color,
            linewidth=1.2,
            zorder=5,
        )
    if first_line is not None:
        first_line.set_label(label)
    return first_line


def vis_history_mask(history_data):
    if history_data is None:
        return None
    return ~np.isclose(history_data[:, :5], -1.0, atol=0.05).all(axis=1)


def vis_setup_axis(ax, map_data, map_xy, map_mask, map_type, xlim, ylim):
    from matplotlib.patches import Rectangle

    ax.set_facecolor("#ffffff")
    vis_draw_map(ax, map_data, map_xy, map_mask, map_type)
    ax.add_patch(
        Rectangle(
            (-1.0, -1.0),
            2.0,
            2.0,
            fill=False,
            linestyle="--",
            linewidth=1.0,
            edgecolor="#ef4444",
            alpha=0.75,
            zorder=2,
        )
    )
    ax.axhline(0.0, color="#d1d5db", linewidth=0.7, zorder=0)
    ax.axvline(0.0, color="#d1d5db", linewidth=0.7, zorder=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("normalized x")
    ax.set_ylabel("normalized y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e5e7eb", linewidth=0.6)


def save_single_visualization(image_path, title, pred_xy, gt_xy, history_xy, map_data, map_mask, map_type, map_name, target_vehicle_mask, timestep_mask, history_timestep_mask, args):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    map_xy = map_data[..., :2].copy()
    pred_xy = vis_traj_xy(pred_xy)
    gt_xy = vis_traj_xy(gt_xy) if gt_xy is not None else None
    history_xy = vis_traj_xy(history_xy) if history_xy is not None else None

    agent_indices = np.where(target_vehicle_mask.astype(bool))[0]
    limit_paths = []
    for agent_idx in agent_indices:
        if history_xy is not None:
            limit_paths.append(vis_valid_path(history_xy, agent_idx, history_timestep_mask))
        if gt_xy is not None:
            limit_paths.append(vis_valid_path(gt_xy, agent_idx, timestep_mask))
    xlim, ylim = vis_compute_limits(map_xy, map_mask, limit_paths, args.vis_pad)

    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    fig.patch.set_facecolor("#ffffff")

    vis_setup_axis(ax, map_data, map_xy, map_mask, map_type, xlim, ylim)
    handles = [
        Line2D([0], [0], color=style["color"], linewidth=style["lw"], alpha=style["alpha"], label=style["label"])
        for style in MAP_STYLE.values()
    ] + [
        Line2D([0], [0], color="#ef4444", linestyle="--", linewidth=1.0, label="[-1, 1] box"),
    ]

    if history_xy is not None:
        handle = vis_draw_paths(
            ax,
            history_xy,
            agent_indices,
            history_timestep_mask,
            color="#64748b",
            label="history",
            linestyle="-",
            linewidth=1.8,
            alpha=0.9,
        )
        if handle is not None:
            handles.append(Line2D([0], [0], color="#64748b", linewidth=1.8, label="history"))
    if gt_xy is not None:
        handle = vis_draw_paths(
            ax,
            gt_xy,
            agent_indices,
            timestep_mask,
            color="#2563eb",
            label="gt future",
            linestyle="-",
            linewidth=2.2,
            alpha=0.95,
        )
        if handle is not None:
            handles.append(Line2D([0], [0], color="#2563eb", linewidth=2.2, label="gt future"))
    if pred_xy is not None:
        handle = vis_draw_paths(
            ax,
            pred_xy,
            agent_indices,
            timestep_mask,
            color="#ea580c",
            label="prediction",
            linestyle="-",
            linewidth=2.2,
            alpha=0.95,
        )
        if handle is not None:
            handles.append(Line2D([0], [0], color="#ea580c", linewidth=2.2, label="prediction"))
    ax.set_title(f"{title} | {map_name}")
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)

    fig.tight_layout()
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, dpi=args.vis_dpi)
    plt.close(fig)


def save_sample_visualizations(
    out_dir,
    step,
    prediction_samples,
    gt,
    history_data,
    map_data,
    map_mask,
    map_type,
    map_name,
    target_vehicle_mask,
    timestep_mask,
    args,
):
    image_dir = out_dir / f"{step + 1:06d}_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    num_scenes = min(args.vis_num_scenes, gt.shape[0])
    num_modes = min(args.vis_num_modes, prediction_samples.shape[0])

    for scene_idx in range(num_scenes):
        scene_name = str(map_name[scene_idx]) if np.ndim(map_name) > 0 else str(map_name)
        for mode_idx in range(num_modes):
            image_path = image_dir / f"scene{scene_idx:03d}_mode{mode_idx:02d}_prediction.png"

            save_single_visualization(
                image_path=image_path,
                title=f"Prediction | mode {mode_idx}",
                pred_xy=prediction_samples[mode_idx, scene_idx],
                gt_xy=gt[scene_idx],
                history_xy=history_data[scene_idx] if history_data is not None else None,
                map_data=map_data[scene_idx],
                map_mask=map_mask[scene_idx],
                map_type=map_type[scene_idx] if map_type is not None else None,
                map_name=scene_name,
                target_vehicle_mask=target_vehicle_mask[scene_idx],
                timestep_mask=timestep_mask[scene_idx],
                history_timestep_mask=vis_history_mask(history_data[scene_idx:scene_idx + 1])[0] if history_data is not None else None,
                args=args,
            )

    print(f"[samples] saved images to {image_dir}")


def train(local_rank, world_size, args, rank=None):
    rank = local_rank if rank is None else rank
    is_distributed = world_size > 1
    set_random_seed(args.seed, rank=rank)

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
            backend=args.dist_backend or ("nccl" if torch.cuda.is_available() else "gloo"),
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )

    is_main = (not is_distributed) or rank == 0

    dataloader, sampler, dataset = build_dataloader(args, is_distributed, rank, world_size)
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
    scaler = make_grad_scaler(enabled=use_amp)

    resume_checkpoint = None
    if args.resume_path:
        resume_checkpoint = load_training_checkpoint(
            args.resume_path,
            model_single,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            load_optimizer=not args.resume_model_only,
        )
    if args.loadckpt:
        if args.load_model_path:
            ckpt = safe_load_state(args.load_model_path, map_location="cpu")
            if ckpt is not None:
                model_single.load_state_dict(normalize_state_dict_keys(ckpt), strict=False)
        if args.load_optim_path:
            opt_state = safe_load_state(args.load_optim_path, map_location="cpu")
            if opt_state is not None:
                optimizer.load_state_dict(opt_state)

    resume_epoch_step = 0
    if args.resume_path and resume_checkpoint is not None and not args.no_resume_iter:
        args.start_iter = int(
            resume_checkpoint.get(
                "next_iter",
                resume_checkpoint.get("global_step", resume_checkpoint.get("step", -1) + 1),
            )
        )
        args.start_epoch = int(resume_checkpoint.get("next_epoch", resume_checkpoint.get("epoch", args.start_epoch)))
        resume_epoch_step = int(resume_checkpoint.get("next_epoch_step", 0))
        if is_main:
            print(
                "[checkpoint] resume position set to "
                f"epoch={args.start_epoch}, epoch_step={resume_epoch_step}, global_step={args.start_iter}"
            )

    if args.compile and hasattr(torch, "compile"):
        if is_main:
            print(f"[rank {rank}] compiling model with torch.compile")
        model_single = torch.compile(model_single, mode="reduce-overhead")

    initialize_model_once(
        model_single,
        dataloader,
        device,
        is_distributed=is_distributed,
        local_rank=rank,
        sampler=sampler,
        train_mode=args.train_mode,
    )

    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            model_single,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=args.ddp_find_unused_parameters,
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
            print(f"[rank {rank}] TensorBoard logdir: {logdir}")
        else:
            print("[tensorboard] tensorboard is not installed; scalar logging disabled")

    steps_per_epoch = len(dataloader)
    global_step = int(args.start_iter)
    max_steps = int(args.max_steps) if args.max_steps is not None and args.max_steps > 0 else None
    if is_main:
        limit_msg = f", max_steps={max_steps}" if max_steps is not None else ""
        print(
            f"[train] epoch-based loop: epochs={args.epochs}, "
            f"start_epoch={args.start_epoch}, steps_per_epoch={steps_per_epoch}, "
            f"start_global_step={global_step}{limit_msg}"
        )

    stop_training = False
    for epoch in range(int(args.start_epoch), int(args.epochs)):
        if sampler is not None:
            sampler.set_epoch(epoch)

        iterable = enumerate(dataloader)
        if is_main:
            progress = tqdm(
                iterable,
                total=steps_per_epoch,
                ncols=120,
                desc=f"Epoch {epoch + 1}/{args.epochs}",
            )
        else:
            progress = iterable

        last_epoch_step = None
        for epoch_step, batch_raw in progress:
            if epoch == int(args.start_epoch) and epoch_step < resume_epoch_step:
                continue

            i = global_step

            batch = process_batch(batch_raw, device, train_mode=args.train_mode)
            target_data = batch["target_data"]
            labels = batch["labels"]
            map_data = batch["map_data"]

            bad_input = torch.isnan(map_data).any() or torch.isnan(target_data).any()
            if distributed_bad_flag(bad_input, is_distributed, device):
                if is_main:
                    print(f"[Step {i}] NaN detected in input tensors, skip batch on all ranks")
                continue

            use_conditional = distributed_label_condition(
                args.train_mode,
                args.label_keep_prob,
                is_distributed,
                device,
            )
            condition_input = labels if use_conditional else None

            with cuda_autocast(enabled=use_amp):
                log_p, logdet, _ = model(
                    target_data,
                    condition=condition_input,
                    **build_model_context_kwargs(batch, train_mode=args.train_mode),
                )
                nll_per_scene = -(logdet + log_p)
                valid_dims = valid_dimension_count(batch, args.in_channel)
                nll_per_valid_dim = nll_per_scene / valid_dims
                if args.loss_normalize == "valid_dim":
                    loss_value = nll_per_valid_dim.mean()
                else:
                    loss_value = nll_per_scene.mean()

            bad_loss = (
                torch.isnan(loss_value)
                or torch.isinf(loss_value)
                or torch.isnan(log_p).any()
                or torch.isnan(logdet).any()
            )
            if distributed_bad_flag(bad_loss, is_distributed, device):
                if is_main:
                    print(f"[Step {i}] NaN/Inf detected, skip batch on all ranks")
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

            completed_step = global_step + 1
            if is_main:
                log_p_mean = log_p.mean()
                logdet_mean = logdet.mean()
                nll_dim_mean = nll_per_valid_dim.mean()
                progress.set_description(
                    f"Epoch {epoch + 1}/{args.epochs}; step: {completed_step}; "
                    f"Loss: {loss_value.item():.5f}; logP: {log_p_mean.item():.5f}; "
                    f"logdet: {logdet_mean.item():.5f}; "
                    f"nll/dim: {nll_dim_mean.item():.5f}; lr: {optimizer.param_groups[0]['lr']:.7f}"
                )
                if writer is not None:
                    writer.add_scalar("train/loss", loss_value.item(), global_step)
                    writer.add_scalar("train/log_p", log_p_mean.item(), global_step)
                    writer.add_scalar("train/logdet", logdet_mean.item(), global_step)
                    writer.add_scalar("train/nll_per_valid_dim", nll_dim_mean.item(), global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                    writer.add_scalar("train/epoch", epoch, global_step)

            if is_main and args.sample_interval > 0 and completed_step % args.sample_interval == 0:
                with torch.no_grad():
                    save_samples_npz(model_single, batch, args, z_shapes, device, completed_step - 1)

            if is_main and args.save_interval > 0 and completed_step % args.save_interval == 0:
                ckpt_dir = Path(args.ckpt_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                last_path = ckpt_dir / "last.pt"
                save_training_checkpoint(
                    last_path,
                    model_single,
                    optimizer,
                    scheduler,
                    scaler,
                    args,
                    completed_step - 1,
                    epoch,
                    epoch_step,
                    steps_per_epoch,
                )
                if args.save_step_checkpoints:
                    step_path = ckpt_dir / f"step_{completed_step:06d}.pt"
                    save_training_checkpoint(
                        step_path,
                        model_single,
                        optimizer,
                        scheduler,
                        scaler,
                        args,
                        completed_step - 1,
                        epoch,
                        epoch_step,
                        steps_per_epoch,
                    )
                print(f"[rank {rank}] saved checkpoint: {last_path}")

            if is_main and args.keep_legacy_checkpoints and args.save_interval > 0 and completed_step % args.save_interval == 0:
                model_path = ckpt_dir / "model_interaction_combined.pt"
                optim_path = ckpt_dir / "optim_interaction_combined.pt"
                torch.save(unwrap_model(model_single).state_dict(), model_path)
                torch.save(optimizer.state_dict(), optim_path)
                print(f"[rank {rank}] saved legacy checkpoints: {model_path}, {optim_path}")

            global_step = completed_step
            last_epoch_step = epoch_step
            if max_steps is not None and global_step >= max_steps:
                stop_training = True
                break

        resume_epoch_step = 0
        if (
            is_main
            and not stop_training
            and last_epoch_step is not None
            and args.save_epoch_interval > 0
            and (epoch + 1) % args.save_epoch_interval == 0
        ):
            ckpt_dir = Path(args.ckpt_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            last_path = ckpt_dir / "last.pt"
            save_training_checkpoint(
                last_path,
                model_single,
                optimizer,
                scheduler,
                scaler,
                args,
                global_step - 1,
                epoch,
                last_epoch_step,
                steps_per_epoch,
            )
            print(f"[rank {rank}] saved epoch checkpoint: {last_path}")
        if stop_training:
            break

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
    parser.add_argument("--batch", default=8, type=int, help="per-GPU batch size")
    parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
    parser.add_argument("--start_epoch", default=0, type=int, help="epoch index to start from")
    parser.add_argument("--max_steps", default=-1, type=int, help="optional global-step cap; <=0 trains full epochs")
    parser.add_argument("--iter", default=-1, type=int, help="deprecated alias for --max_steps")
    parser.add_argument("--start_iter", default=0, type=int, help="global step to start logging/checkpoint numbering")
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
    parser.add_argument(
        "--prediction_representation",
        default="absolute",
        choices=["absolute", "delta"],
        help="prediction target representation; delta accumulates from the last valid history state",
    )
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
    parser.add_argument("--worker_start_method", default="", choices=["", "fork", "spawn", "forkserver"],
                        help="dataloader multiprocessing context; empty uses PyTorch default")
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True,
                        help="keep dataloader workers alive when num_workers > 0")
    parser.add_argument("--prefetch_factor", default=2, type=int,
                        help="dataloader prefetch factor when num_workers > 0; <=0 disables explicit setting")
    parser.add_argument("--sample_interval", default=2000, type=int, help="sample save interval")
    parser.add_argument("--save_interval", default=2000, type=int, help="checkpoint save interval")
    parser.add_argument("--save_epoch_interval", default=1, type=int, help="save last.pt every N epochs; <=0 disables")
    parser.add_argument("--log_dir", default="./runs", type=str, help="tensorboard log dir")
    parser.add_argument("--ckpt_dir", default="./results", type=str, help="checkpoint dir")
    parser.add_argument("--sample_out_dir", default="./results", type=str, help="sample output dir")
    parser.add_argument("--save_sample_images", action=argparse.BooleanOptionalAction, default=True,
                        help="save visualization pngs together with sample npz")
    parser.add_argument("--vis_num_scenes", default=4, type=int, help="number of sampled scenes to visualize")
    parser.add_argument("--vis_num_modes", default=2, type=int, help="number of modes per scene to visualize")
    parser.add_argument("--vis_position_scale", default=50.0, type=float,
                        help="position scale used to denormalize x/y for visualization")
    parser.add_argument("--vis_pad", default=0.08, type=float, help="fractional padding in normalized visualization limits")
    parser.add_argument("--vis_dpi", default=180, type=int, help="dpi for saved visualization images")
    parser.add_argument("--load_model_path", default="", type=str, help="optional model checkpoint path")
    parser.add_argument("--load_optim_path", default="", type=str, help="optional optimizer checkpoint path")
    parser.add_argument("--loadckpt", action="store_true", help="load ckpt before training")
    parser.add_argument("--resume_path", default="", type=str,
                        help="full training checkpoint path, e.g. results/last.pt")
    parser.add_argument("--resume_model_only", action="store_true",
                        help="with --resume_path, load model weights but not optimizer/scheduler/scaler")
    parser.add_argument("--no_resume_iter", action="store_true",
                        help="with --resume_path, keep CLI/config start_iter instead of checkpoint next_iter")
    parser.add_argument("--save_step_checkpoints", action=argparse.BooleanOptionalAction, default=False,
                        help="also save numbered step_*.pt checkpoints")
    parser.add_argument("--keep_legacy_checkpoints", action=argparse.BooleanOptionalAction, default=True,
                        help="also save model_interaction_combined.pt and optim_interaction_combined.pt")
    parser.add_argument("--loss_normalize", default="scene", choices=["scene", "valid_dim"],
                        help="scene keeps original summed scene NLL; valid_dim averages by valid target dimensions")
    parser.add_argument("--amp", action="store_true", help="enable automatic mixed precision")
    parser.add_argument("--compile", action="store_true", help="enable torch.compile when available")
    parser.add_argument("--seed", default=42, type=int, help="base random seed; negative disables seeding")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False,
                        help="enable deterministic algorithms where PyTorch supports them")
    parser.add_argument("--devices", default=-1, type=int,
                        help="number of local CUDA devices for spawn launcher; -1 uses all visible GPUs")
    parser.add_argument("--launcher", default="auto", choices=["auto", "spawn", "torchrun", "none"],
                        help="distributed launcher mode")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str, help="MASTER_ADDR for spawn launcher")
    parser.add_argument("--master_port", default="12349", type=str, help="MASTER_PORT for spawn launcher")
    parser.add_argument("--dist_backend", default="", type=str, help="override distributed backend")
    parser.add_argument("--ddp_find_unused_parameters", action=argparse.BooleanOptionalAction, default=True,
                        help="DDP find_unused_parameters; keep true because label conditioning can disable branches")

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

    if args.iter is not None and args.iter > 0 and args.max_steps <= 0:
        args.max_steps = args.iter
        print(f"[config] --iter is deprecated; using max_steps={args.max_steps}")
    if args.epochs <= 0:
        raise ValueError(f"epochs must be > 0, got {args.epochs}")
    if args.start_epoch < 0:
        raise ValueError(f"start_epoch must be >= 0, got {args.start_epoch}")
    if args.use_history and args.train_mode == "initialization":
        args.train_mode = "prediction"
        print("[config] --use_history is deprecated; switching train_mode to prediction")
    args.use_history = args.train_mode == "prediction"
    if not 0.0 <= args.label_keep_prob <= 1.0:
        raise ValueError(f"label_keep_prob must be in [0, 1], got {args.label_keep_prob}")
    if args.num_workers == 0 and not args.persistent_workers:
        args.persistent_workers = False
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

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
    if (
        args.train_mode == "prediction"
        and args.prediction_representation == "delta"
        and args.in_channel != 5
    ):
        raise ValueError("delta prediction requires in_channel=5")

    print("Training args:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    torchrun_world_size = get_env_int("WORLD_SIZE", 1)
    torchrun_rank = get_env_int("RANK", 0)
    torchrun_local_rank = get_env_int("LOCAL_RANK", 0)
    use_torchrun = args.launcher == "torchrun" or (
        args.launcher == "auto" and torchrun_world_size is not None and torchrun_world_size > 1
    )

    visible_gpus = torch.cuda.device_count()
    if use_torchrun:
        print(
            f"Detected torchrun environment: rank={torchrun_rank}, "
            f"local_rank={torchrun_local_rank}, world_size={torchrun_world_size}"
        )
        train(torchrun_local_rank, torchrun_world_size, args, rank=torchrun_rank)
        return

    if args.launcher == "none":
        print("Launcher disabled, running single process.")
        train(0, 1, args, rank=0)
        return

    os.environ.setdefault("MASTER_ADDR", args.master_addr)
    os.environ.setdefault("MASTER_PORT", args.master_port)

    requested_gpus = visible_gpus if args.devices is None or args.devices < 0 else args.devices
    if visible_gpus == 0 or requested_gpus == 0:
        print("No GPU detected, run in single-process CPU mode.")
        train(0, 1, args, rank=0)
    elif requested_gpus == 1:
        print("Detected 1 GPU, run in single-process CUDA mode.")
        train(0, 1, args, rank=0)
    else:
        if requested_gpus > visible_gpus:
            raise ValueError(f"requested devices={requested_gpus}, but only {visible_gpus} CUDA devices are visible")
        print(f"Detected {visible_gpus} GPUs, launching DDP on {requested_gpus} GPUs with mp.spawn")
        mp.spawn(train, nprocs=requested_gpus, args=(requested_gpus, args))


if __name__ == "__main__":
    main()
