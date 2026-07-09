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

To save visualization PNGs during sampling:

```bash
bash scripts/train.sh /data/interaction_multi_train_combined.npz --save_sample_images
```

Outputs:

- checkpoints: `results/model_interaction_combined.pt`, `results/optim_interaction_combined.pt`
- samples: `results/*_interaction_combined_samples.npz`
- TensorBoard logs: `runs/`
