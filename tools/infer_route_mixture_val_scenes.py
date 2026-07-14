#!/usr/bin/env python3
"""Run a RouteMixtureGlow checkpoint on diverse validation scenes.

The selector first covers every available map name, then round-robins over maps
until ``--num-scenes`` distinct (map_name, case_id) pairs have been selected.
Each saved panel contains all explicit route components, not an implicit route
sample from the mixture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data._utils.collate import default_collate

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
    save_single_visualization,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-scenes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def select_scene_indices(path: str, count: int, seed: int):
    with np.load(path, allow_pickle=False) as data:
        names = np.asarray(data["map_names"]).astype(str)
        case_ids = np.asarray(data["case_ids"]).astype(np.int64)

    rng = np.random.default_rng(seed)
    groups = {}
    for name in sorted(np.unique(names)):
        candidates = np.flatnonzero(names == name)
        rng.shuffle(candidates)
        # A case can appear in several temporal windows. Keep one window per case.
        deduplicated = []
        seen_cases = set()
        for idx in candidates.tolist():
            case_id = int(case_ids[idx])
            if case_id not in seen_cases:
                seen_cases.add(case_id)
                deduplicated.append(idx)
        groups[name] = deduplicated

    selected = []
    depth = 0
    map_names = sorted(groups)
    while len(selected) < count:
        added = False
        for name in map_names:
            candidates = groups[name]
            if depth < len(candidates):
                idx = candidates[depth]
                selected.append(idx)
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        depth += 1
    if len(selected) < count:
        raise ValueError(f"only {len(selected)} distinct map/case scenes available")
    return selected, names, case_ids


def make_panel(image_paths, output_path, title):
    images = [Image.open(path).convert("RGB") for path in image_paths]
    thumb_w = 560
    thumbs = []
    for image in images:
        height = round(image.height * thumb_w / image.width)
        thumbs.append(image.resize((thumb_w, height), Image.Resampling.LANCZOS))
    cell_h = max(image.height for image in thumbs)
    header_h = 46
    canvas = Image.new("RGB", (thumb_w * 3, header_h + cell_h * 2), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 14), title, fill="black")
    for index, image in enumerate(thumbs):
        x = (index % 3) * thumb_w
        y = header_h + (index // 3) * cell_h
        canvas.paste(image, (x, y))
    canvas.save(output_path, quality=92)
    for image in images:
        image.close()


def make_overview(panel_paths, output_path):
    panels = [Image.open(path).convert("RGB") for path in panel_paths]
    cell_w = 520
    thumbs = []
    for panel in panels:
        height = round(panel.height * cell_w / panel.width)
        thumbs.append(panel.resize((cell_w, height), Image.Resampling.LANCZOS))
    cell_h = max(image.height for image in thumbs)
    canvas = Image.new("RGB", (cell_w * 4, cell_h * 5), "white")
    for index, image in enumerate(thumbs):
        canvas.paste(image, ((index % 4) * cell_w, (index // 4) * cell_h))
    canvas.save(output_path, quality=90)
    for panel in panels:
        panel.close()


def main():
    cli = parse_args()
    device = torch.device(cli.device if cli.device != "cuda" or torch.cuda.is_available() else "cpu")
    output_dir = Path(cli.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    indices, all_names, all_case_ids = select_scene_indices(cli.val_npz, cli.num_scenes, cli.seed)
    checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint.get("args", {})
    args = SimpleNamespace(**saved)
    args.train_mode = "prediction"
    args.prediction_representation = saved.get("prediction_representation", "cv_residual")
    args.future_steps = int(saved.get("future_steps", 30))
    args.prediction_target_steps = int(saved.get("prediction_target_steps", 30))
    args.vis_pad = float(saved.get("vis_pad", 0.08))
    args.vis_dpi = cli.dpi

    dataset = CombinedInteractionDataset(
        combined_path=cli.val_npz,
        in_channel=int(saved.get("in_channel", 5)),
        train_mode="prediction",
        history_steps=int(saved.get("history_steps", 10)),
        future_steps=args.future_steps,
        prediction_target_steps=args.prediction_target_steps,
        prediction_representation=args.prediction_representation,
        label_source="none",
        turn_angle_threshold_deg=float(saved.get("turn_angle_threshold_deg", 30.0)),
        stationary_dist_threshold=float(saved.get("stationary_dist_threshold", 0.0)),
        allow_legacy_prediction_data=False,
        require_map_topology=True,
    )

    model = RouteMixtureGlow(
        in_channel=int(saved.get("in_channel", 5)),
        condition_dim=32,
        n_flow=int(saved.get("n_flow", 12)),
        n_block=int(saved.get("n_block", 1)),
        affine=bool(saved.get("affine", True)),
        conv_lu=not bool(saved.get("no_lu", False)),
        history_input_dim=int(saved.get("in_channel", 5)),
        route_modes=int(saved.get("route_modes", 6)),
        future_steps=args.prediction_target_steps,
        hidden_dim=int(saved.get("route_hidden_dim", 256)),
        mixture_chunk_size=int(saved.get("mixture_chunk_size", 2)),
    ).to(device)
    load_training_checkpoint(cli.checkpoint, model, load_optimizer=False)
    model.eval()

    z_shapes = calc_z_shapes(
        int(saved.get("in_channel", 5)),
        dataset.time_steps,
        dataset.max_agents,
        int(saved.get("n_block", 1)),
    )
    route_modes = int(saved.get("route_modes", 6))
    temperature = float(saved.get("temp", 0.7))
    decay = float(saved.get("temp_block_decay", 1.0))
    records = []
    panel_paths = []

    with torch.inference_mode():
        for output_index, dataset_index in enumerate(indices):
            batch = process_batch(default_collate([dataset[dataset_index]]), device, "prediction")
            kwargs = build_model_context_kwargs(
                batch, train_mode="prediction", model=model, method_name="reverse",
                for_sampling=True, future_steps=args.future_steps,
            )
            generation_mask = build_generation_timestep_mask(batch, "prediction", args.future_steps)
            gt = decode_prediction_states(batch["target_data"], batch, args).cpu().numpy()[0]
            history = batch["history_data"].cpu().numpy()[0]
            map_data = batch["map_data"].cpu().numpy()[0]
            map_mask = batch["map_mask"].cpu().numpy()[0]
            map_type = batch["map_type"].cpu().numpy()[0]
            agent_mask = batch["target_vehicle_mask"].cpu().numpy()[0]
            timestep_mask = batch["timestep_mask"].cpu().numpy()[0]
            history_mask = batch["history_timestep_mask"].cpu().numpy()[0]

            mode_predictions = []
            mode_paths = []
            for mode_index in range(route_modes):
                torch.manual_seed(cli.seed + output_index * 100 + mode_index)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(cli.seed + output_index * 100 + mode_index)
                z = sample_latents(1, z_shapes, device, temperature, decay)
                prediction = model.reverse(z, mode_index=mode_index, **kwargs)
                prediction = decode_prediction_states(
                    prediction, batch, args, timestep_mask=generation_mask
                ).cpu().numpy()[0]
                mode_predictions.append(prediction)
                image_path = image_dir / f"scene{output_index:02d}_mode{mode_index}.png"
                save_single_visualization(
                    image_path=image_path,
                    title=f"Validation scene {output_index + 1}/20 | explicit route mode {mode_index}",
                    pred_xy=prediction,
                    gt_xy=gt,
                    history_xy=history,
                    map_data=map_data,
                    map_mask=map_mask,
                    map_type=map_type,
                    map_name=f"{all_names[dataset_index]} | case {int(all_case_ids[dataset_index])}",
                    target_vehicle_mask=agent_mask,
                    timestep_mask=timestep_mask,
                    history_timestep_mask=history_mask,
                    args=args,
                )
                mode_paths.append(image_path)

            panel_path = output_dir / f"scene{output_index:02d}_all_modes.jpg"
            make_panel(
                mode_paths, panel_path,
                f"Scene {output_index + 1}: {all_names[dataset_index]} | case {int(all_case_ids[dataset_index])}",
            )
            panel_paths.append(panel_path)
            records.append({
                "output_scene": output_index,
                "dataset_index": int(dataset_index),
                "map_name": str(all_names[dataset_index]),
                "case_id": int(all_case_ids[dataset_index]),
                "panel": panel_path.name,
            })
            np.savez_compressed(
                output_dir / f"scene{output_index:02d}_predictions.npz",
                predictions=np.stack(mode_predictions), gt=gt, history=history,
                map_data=map_data, map_mask=map_mask, map_type=map_type,
                target_vehicle_mask=agent_mask, timestep_mask=timestep_mask,
                history_timestep_mask=history_mask,
            )
            print(f"[{output_index + 1:02d}/{cli.num_scenes}] {panel_path}", flush=True)

    overview_path = output_dir / "overview_20_scenes.jpg"
    make_overview(panel_paths, overview_path)
    metadata = {
        "checkpoint": str(Path(cli.checkpoint).resolve()),
        "validation_npz": str(Path(cli.val_npz).resolve()),
        "checkpoint_next_iter": int(checkpoint.get("next_iter", -1)),
        "selection_seed": cli.seed,
        "available_unique_map_names": int(len(np.unique(all_names))),
        "selection_definition": "distinct (map_name, case_id), round-robin over map names",
        "scenes": records,
        "overview": overview_path.name,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved overview: {overview_path}")


if __name__ == "__main__":
    main()
