# Triton paired fine-tuning

These templates are for paired noisy/clean data. They do not use the original
URGENT-style online noise/RIR simulation loader.

## 1. Prepare metadata

The noisy and clean directories should have matching relative paths:

```bash
python scripts/make_paired_csv.py \
  --noisy-dir /path/to/noisy_train \
  --clean-dir /path/to/clean_train \
  --output /path/to/metadata/train_pairs.csv

python scripts/make_paired_csv.py \
  --noisy-dir /path/to/noisy_valid \
  --clean-dir /path/to/clean_valid \
  --output /path/to/metadata/valid_pairs.csv
```

The generated CSV columns are:

```text
uid,sample_rate,noisy_filepath,clean_filepath,filename,audio_length
```

## 2. Set paths

```bash
export PASE_WAVLM_CKPT=/path/to/WavLM-Large.pt
export PASE_DEWAVLM_CKPT=/path/to/DeWavLM.tar
export PASE_VOCODER_L24_CKPT=/path/to/Vocoder_L24.tar
export PASE_VOCODER_DUAL_CKPT=/path/to/Vocoder_Dual.tar
export PASE_TRAIN_PAIRS_CSV=/path/to/metadata/train_pairs.csv
export PASE_VALID_PAIRS_CSV=/path/to/metadata/valid_pairs.csv
export PASE_EXPERIMENT_ROOT=/path/to/experiments/pase
```

## 3. Recommended first run

Start with inference using the official checkpoint before fine-tuning:

```bash
python -m inference.inference \
  -I /path/to/noisy_eval \
  -O /path/to/pase_eval \
  -D cuda:0 \
  --dewavlm_ckpt "$PASE_DEWAVLM_CKPT" \
  --vocoder_ckpt "$PASE_VOCODER_DUAL_CKPT"
```

For fine-tuning, the safer first target is the dual-stream vocoder:

```bash
python -m train.train_vocoder_dual \
  -C configs/triton/cfg_train_vocoder_dual_paired.yaml \
  -D 0
```

Only fine-tune DeWavLM if the baseline output shows systematic representation
errors on your data:

```bash
python -m train.train_dewavlm \
  -C configs/triton/cfg_train_dewavlm_paired.yaml \
  -D 0
```

Use `-D 0,1,2,3` for multi-GPU runs.
