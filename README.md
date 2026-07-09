# NormGen

NormGen contains the MapGlow model, INTERACTION preprocessing script, and combined training script.
The dataset is intentionally not included in this repository.

## Files

- `MapGlow11_27_original.py`: model code.
- `train_combined.py`: training entrypoint for initialization and prediction modes.
- `data_preprocess.py`: INTERACTION multi-scenario preprocessing.
- `configs/prediction.yaml`: default future-prediction training config.
- `configs/initialization.yaml`: default full-trajectory initialization training config.
- `scripts/train.sh`: one-command training wrapper.
- `scripts/preprocess.sh`: preprocessing wrapper.
- `scripts/prepare_autobots_dataset.sh`: convert NormGen NPZ output to AutoBots HDF5.
- `scripts/prepare_autobots_experiment.sh`: build an AutoBots train/val root from generated train data and optional real validation data.
- `scripts/smoke_autobots_pipeline.sh`: quick server check for the NormGen-to-AutoBots pipeline.
- `scripts/train_autobots.sh`: train AutoBots on the converted HDF5 dataset.
- `scripts/eval_autobots.sh`: evaluate an AutoBots checkpoint.

## Environment

Recommended conda setup:

```bash
conda env create -f environment.yml
conda activate normgen
```

If your server uses a different CUDA version, install PyTorch for that CUDA version first, then install the remaining packages:

```bash
pip install -r requirements.txt
```

## Dataset

Training expects a processed combined NPZ such as:

```text
data/interaction_multi_train_combined.npz
```

You can either copy your already processed file to that path, or point the script to any location:

```bash
bash scripts/train.sh /absolute/path/to/interaction_multi_train_combined.npz
```

## Preprocess Raw INTERACTION Data

Raw dataset layout should look like:

```text
INTERACTION-Dataset-DR-multi-v1_2/
  train/
  val/
  maps/
```

Run preprocessing:

```bash
bash scripts/preprocess.sh /absolute/path/to/INTERACTION-Dataset-DR-multi-v1_2
```

By default this writes:

```text
data/processed_train/interaction_multi_train_combined.npz
```

Then train from that output:

```bash
bash scripts/train.sh data/processed_train/interaction_multi_train_combined.npz
```

## One-Command Training

Prediction mode is the default:

```bash
bash scripts/train.sh /absolute/path/to/interaction_multi_train_combined.npz
```

Prediction mode uses:

- history: 10 frames
- target: 30 future frames padded to 32 for squeeze
- label condition: disabled

Initialization mode:

```bash
MODE=initialization bash scripts/train.sh /absolute/path/to/interaction_multi_train_combined.npz
```

You can also create a local `.env`:

```bash
cp .env.example .env
```

Then edit `COMBINED_PATH` and run:

```bash
bash scripts/train.sh
```

## Useful Overrides

Any extra arguments after the dataset path are passed to `train_combined.py`:

```bash
bash scripts/train.sh /data/interaction_multi_train_combined.npz --batch 4 --iter 100000
```

`--batch` is the per-GPU batch size. With 4 GPUs and `--batch 8`, the global
batch size is 32.

Single-node multi-GPU training:

```bash
NUM_GPUS=4 bash scripts/train.sh /data/interaction_multi_train_combined.npz
```

Equivalent explicit `torchrun` command:

```bash
torchrun --standalone --nproc_per_node 4 train_combined.py \
  --launcher torchrun \
  --config configs/prediction.yaml \
  --combined_path /data/interaction_multi_train_combined.npz
```

For low-memory servers, start conservatively because every DDP rank loads the
processed NPZ:

```bash
NUM_GPUS=4 bash scripts/train.sh /data/interaction_multi_train_combined.npz \
  --batch 4 \
  --num_workers 0
```

Resume from the full training checkpoint:

```bash
bash scripts/train.sh /data/interaction_multi_train_combined.npz \
  --resume_path results/last.pt
```

To save visualization PNGs during sampling:

```bash
bash scripts/train.sh /data/interaction_multi_train_combined.npz --save_sample_images
```

Outputs:

- full checkpoint: `results/last.pt`
- legacy checkpoints: `results/model_interaction_combined.pt`, `results/optim_interaction_combined.pt`
- samples: `results/*_interaction_combined_samples.npz`
- TensorBoard logs: `runs/`

## Server Quick Start

After pulling this repository on the server:

```bash
conda env create -f environment.yml
conda activate normgen
cp .env.example .env
```

Edit `.env` so `COMBINED_PATH`, `INTERACTION_ROOT`, `AUTOBOTS_ROOT`, `INTERACTION_MAPS_ROOT`, and `NORMGEN_NPZ` point to your server paths. The repo does not include datasets.
If the command `python` does not point to the intended conda environment, set `PYTHON_BIN=/path/to/env/bin/python` in `.env`.

Run NormGen training:

```bash
bash scripts/train.sh
```

For the AutoBots pipeline, install the extra dependencies in the environment used to run AutoBots:

```bash
pip install -r requirements-autobots.txt
```

Then run the smoke check before starting a long AutoBots job:

```bash
bash scripts/smoke_autobots_pipeline.sh
```

The smoke script writes temporary output under `server_workspace/` by default, not `/tmp`.

For paper-style AutoBots experiments, prepare generated training data with a real validation split:

```bash
AUTOBOTS_DATASET_DIR=autobots_data/prediction_generated_train_real_val \
NORMGEN_TRAIN_NPZ=/path/to/prediction_samples.npz \
REAL_VAL_NPZ=/path/to/real_val_interaction_multi_combined.npz \
bash scripts/prepare_autobots_experiment.sh
```

## AutoBots Pipeline

AutoBots does not train directly from NormGen NPZ files. Its Interaction-Dataset loader expects:

```text
DATASET_ROOT/
  train_dataset.hdf5
  val_dataset.hdf5
  maps/*.osm
```

This repository provides a converter from NormGen output to that AutoBots HDF5 format.

### 1. Clone AutoBots

Put AutoBots next to this repository, or set `AUTOBOTS_ROOT` in `.env`:

```bash
git clone https://github.com/roggirg/AutoBots.git ../AutoBots
```

Install AutoBots dependencies in the environment you use to run AutoBots:

```bash
pip install -r requirements-autobots.txt
```

If your server already has a PyTorch/CUDA build installed, install the non-PyTorch packages first or adjust the requirements file to avoid replacing your CUDA-compatible torch build.

### 2. Convert NormGen Output to AutoBots HDF5

For a NormGen sample file from prediction mode:

```bash
INTERACTION_MAPS_ROOT=/path/to/INTERACTION-Dataset-DR-multi-v1_2/maps \
bash scripts/prepare_autobots_dataset.sh results/000001_interaction_combined_samples.npz
```

For initialization mode samples, use the same command. The converter detects the format:

- prediction samples: `history_data` + generated future `30` frames are combined into a `40` frame AutoBots trajectory.
- initialization samples: generated `40` frame trajectories are used directly.
- combined preprocessed NPZ files: original `trajectories + dimensions` are converted directly.

By default, generated modes are expanded as separate AutoBots scenes. To use only one mode:

```bash
MODE_INDEX=0 bash scripts/prepare_autobots_dataset.sh /path/to/sample.npz
```

To use unconditional samples:

```bash
SAMPLE_KEY=unconditional_samples bash scripts/prepare_autobots_dataset.sh /path/to/sample.npz
```

To convert a preprocessed combined NPZ:

```bash
python tools/convert_normgen_to_autobots.py \
  --input-npz /path/to/interaction_multi_train_combined.npz \
  --output-dir autobots_data/real_train \
  --source combined \
  --val-ratio 0.1 \
  --maps-root /path/to/INTERACTION-Dataset-DR-multi-v1_2/maps
```

The default output is:

```text
autobots_data/normgen_generated/
  train_dataset.hdf5
  val_dataset.hdf5
  maps/*.osm
```

If `INTERACTION_MAPS_ROOT` is not set, the converter writes minimal dummy maps. That is enough for `USE_MAP_LANES=0`; for map-lane experiments, use the real INTERACTION maps.

Before a long run, check the converted format and AutoBots loader:

```bash
bash scripts/smoke_autobots_pipeline.sh /path/to/000001_interaction_combined_samples.npz
```

If the smoke check reports missing modules such as `pyproj` or `cv2`, install `requirements-autobots.txt` in the AutoBots environment.
To inspect the HDF5 files without importing AutoBots:

```bash
python tools/inspect_autobots_dataset.py --dataset-dir autobots_data/prediction_generated_train_real_val
```

### 3. Train AutoBots on NormGen Data

For strict evaluation, prefer this command because it uses generated data for train and real data for validation:

```bash
AUTOBOTS_DATASET_DIR=autobots_data/prediction_generated_train_real_val \
NORMGEN_TRAIN_NPZ=/path/to/prediction_samples.npz \
REAL_VAL_NPZ=/path/to/real_val_interaction_multi_combined.npz \
bash scripts/prepare_autobots_experiment.sh
```

For initialization-mode generated samples, only change `NORMGEN_TRAIN_NPZ`:

```bash
AUTOBOTS_DATASET_DIR=autobots_data/init_generated_train_real_val \
NORMGEN_TRAIN_NPZ=/path/to/initialization_samples.npz \
REAL_VAL_NPZ=/path/to/real_val_interaction_multi_combined.npz \
bash scripts/prepare_autobots_experiment.sh
```

The script creates:

```text
AUTOBOTS_DATASET_DIR/
  train_dataset.hdf5   # generated prediction/init samples
  val_dataset.hdf5     # real combined validation, if REAL_VAL_NPZ is set
  maps/*.osm
```

If `REAL_VAL_NPZ` is not set, it splits the generated data into train/val. Use that only for debugging or smoke tests, not for final paper numbers.

```bash
AUTOBOTS_ROOT=../AutoBots \
AUTOBOTS_DATASET_DIR=autobots_data/normgen_generated \
bash scripts/train_autobots.sh
```

Useful overrides:

```bash
AUTOBOTS_EPOCHS=10 AUTOBOTS_BATCH_SIZE=16 bash scripts/train_autobots.sh
```

`train_autobots.sh` creates a timestamped `EXP_ID` by default so repeated runs do not stop at AutoBots' overwrite prompt. Set `EXP_ID=my_run_name` when you want a fixed experiment directory.
`AUTOBOTS_NUM_WORKERS=0` is the default because AutoBots hard-codes 12 DataLoader workers, which can hang on some servers. Increase it if your server handles multiprocessing well.

To train AutoBots with map lanes:

```bash
USE_MAP_LANES=1 INTERACTION_MAPS_ROOT=/path/to/INTERACTION-Dataset-DR-multi-v1_2/maps \
bash scripts/prepare_autobots_dataset.sh /path/to/sample.npz
USE_MAP_LANES=1 bash scripts/train_autobots.sh
```

### 4. Evaluate an AutoBots Checkpoint

```bash
bash scripts/eval_autobots.sh /path/to/best_models_fde.pth
```

The evaluation script uses `val_dataset.hdf5` from `AUTOBOTS_DATASET_DIR`.

### Recommended Experiments

Train/test combinations that match the paper-style question:

1. Real-to-real baseline:
   Convert real preprocessed train/val data and train AutoBots normally.
2. Prediction-generated data:
   Use NormGen prediction samples, convert them, train AutoBots, and evaluate on real val HDF5.
3. Initialization-generated data:
   Use NormGen initialization samples, convert them, train AutoBots, and evaluate on real val HDF5.
4. Mixed real + generated:
   Convert generated data and real data separately, then concatenate HDF5 files or train in stages.

For strict evaluation, keep the AutoBots validation set real, not generated.
