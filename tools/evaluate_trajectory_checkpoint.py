#!/usr/bin/env python3
"""Evaluate only trajectory displacement metrics for a prediction checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from MapGlow11_27_original import Glow as MapGlow
from RouteMixtureMapGlow import Glow as RouteMixtureGlow
from train_combined import (
    CombinedInteractionDataset,
    build_generation_timestep_mask,
    build_model_context_kwargs,
    calc_z_shapes,
    decode_prediction_states,
    load_training_checkpoint,
    process_batch,
    sample_latents,
    trajectory_metric_sums,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-npz", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-modes", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    cli = parse_args()
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)

    checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    saved = dict(checkpoint.get("args") or {})
    args = SimpleNamespace(**saved)
    args.model_arch = saved.get("model_arch", "mapglow")
    args.train_mode = saved.get("train_mode", "prediction")
    args.prediction_representation = saved.get("prediction_representation", "absolute")
    args.future_steps = int(saved.get("future_steps", 30))
    args.turn_angle_threshold_deg = float(saved.get("turn_angle_threshold_deg", 30.0))
    args.vis_position_scale = float(saved.get("vis_position_scale", 50.0))
    args.temp = float(saved.get("temp", 0.7))
    args.temp_block_decay = float(saved.get("temp_block_decay", 1.0))

    val_path = cli.val_npz or saved.get("val_combined_path", "")
    if not val_path:
        raise ValueError("validation path is missing from both CLI and checkpoint")
    batch_size = cli.batch_size or int(saved.get("val_batch", 0)) or int(saved.get("batch", 8))
    device = torch.device(cli.device if cli.device != "cuda" or torch.cuda.is_available() else "cpu")

    dataset = CombinedInteractionDataset(
        combined_path=val_path,
        in_channel=int(saved.get("in_channel", 5)),
        train_mode=args.train_mode,
        history_steps=int(saved.get("history_steps", 10)),
        future_steps=args.future_steps,
        prediction_target_steps=int(saved.get("prediction_target_steps", 30)),
        prediction_representation=args.prediction_representation,
        label_source=saved.get("label_source", "none"),
        turn_angle_threshold_deg=args.turn_angle_threshold_deg,
        stationary_dist_threshold=float(saved.get("stationary_dist_threshold", 0.0)),
        allow_legacy_prediction_data=bool(saved.get("allow_legacy_prediction_data", False)),
        require_map_topology=args.model_arch == "route_mixture_glow",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model_class = RouteMixtureGlow if args.model_arch == "route_mixture_glow" else MapGlow
    model_kwargs = dict(
        in_channel=int(saved.get("in_channel", 5)),
        condition_dim=32,
        n_flow=int(saved.get("n_flow", 16)),
        n_block=int(saved.get("n_block", 1)),
        affine=bool(saved.get("affine", True)),
        conv_lu=not bool(saved.get("no_lu", False)),
        history_input_dim=int(saved.get("in_channel", 5)),
    )
    if args.model_arch == "route_mixture_glow":
        model_kwargs.update(
            route_modes=int(saved.get("route_modes", 6)),
            future_steps=int(saved.get("prediction_target_steps", 30)),
            hidden_dim=int(saved.get("route_hidden_dim", 256)),
            mixture_chunk_size=int(saved.get("mixture_chunk_size", 2)),
        )
    model = model_class(**model_kwargs).to(device)
    load_training_checkpoint(cli.checkpoint, model, load_optimizer=False)
    model.eval()

    z_shapes = calc_z_shapes(
        int(saved.get("in_channel", 5)), dataset.time_steps, dataset.max_agents,
        int(saved.get("n_block", 1)),
    )
    totals = torch.zeros(8, dtype=torch.float64, device=device)

    with torch.inference_mode():
        for batch_index, batch_raw in enumerate(loader):
            batch = process_batch(batch_raw, device, train_mode=args.train_mode)
            reverse_kwargs = build_model_context_kwargs(
                batch, train_mode=args.train_mode, model=model,
                method_name="reverse", for_sampling=True,
                future_steps=args.future_steps,
            )
            generation_mask = build_generation_timestep_mask(
                batch, args.train_mode, args.future_steps
            )
            samples = []
            for mode_index in range(cli.num_modes):
                z = sample_latents(
                    batch["target_data"].shape[0], z_shapes, device,
                    args.temp, args.temp_block_decay,
                )
                route_kwargs = {}
                if args.model_arch == "route_mixture_glow":
                    route_kwargs["mode_index"] = mode_index % int(saved.get("route_modes", 6))
                sample = model.reverse(z, **route_kwargs, **reverse_kwargs)
                samples.append(decode_prediction_states(
                    sample, batch, args, timestep_mask=generation_mask
                ))
            samples = torch.stack(samples)
            target = decode_prediction_states(batch["target_data"], batch, args)
            position_scale = batch["scene_stats"][:, 2].abs()
            position_scale = torch.where(
                position_scale > 0, position_scale,
                torch.full_like(position_scale, args.vis_position_scale),
            )
            metric = trajectory_metric_sums(
                samples, target, batch["loss_timestep_mask"],
                batch["context_agent_mask"], position_scale,
                history=batch.get("history_data"),
                history_timestep_mask=batch.get("history_timestep_mask"),
                velocity_scale=batch["state_scales"][:, 1],
                yaw_scale=batch["state_scales"][:, 2],
                turn_angle_threshold_deg=args.turn_angle_threshold_deg,
            )
            totals[0] += metric["ade_sum"].double()
            totals[1] += metric["fde_sum"].double()
            totals[2] += metric["minade_sum"].double()
            totals[3] += metric["minfde_sum"].double()
            totals[4] += metric["agent_count"].double()
            totals[5] += metric.get("turn_minade_sum", 0.0)
            totals[6] += metric.get("turn_minfde_sum", 0.0)
            totals[7] += metric.get("turn_agent_count", 0.0)
            if (batch_index + 1) % 100 == 0 or batch_index + 1 == len(loader):
                print(f"[{batch_index + 1}/{len(loader)}]", flush=True)

    agent_count = totals[4]
    if agent_count <= 0:
        raise RuntimeError("validation set contains no valid target agents")
    turn_count = totals[7]

    result = {
        "checkpoint": str(Path(cli.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint.get("global_step", checkpoint.get("step", -1))) + 1,
        "validation_npz": str(Path(val_path).resolve()),
        "validation_scenes": len(dataset),
        "evaluated_agents": int(agent_count.item()),
        "num_samples": cli.num_modes,
        "temperature": args.temp,
        "seed": cli.seed,
        "ade_m": float((totals[0] / agent_count).item()),
        "fde_m": float((totals[1] / agent_count).item()),
        f"minade{cli.num_modes}_m": float((totals[2] / agent_count).item()),
        f"minfde{cli.num_modes}_m": float((totals[3] / agent_count).item()),
    }
    if turn_count > 0:
        result.update(
            turning_agents=int(turn_count.item()),
            **{
                f"turn_minade{cli.num_modes}_m": float((totals[5] / turn_count).item()),
                f"turn_minfde{cli.num_modes}_m": float((totals[6] / turn_count).item()),
            },
        )
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
