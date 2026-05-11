# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from torch.utils import data

from utils import simulate_utils


class PairedSpeechDataset(data.Dataset):
    def __init__(
        self,
        pair_csvs: List,
        wav_len=4,
        num_per_epoch=10000,
        random_start=False,
        default_fs=16000,
        mode="train",
        normalize=True,
    ):
        super().__init__()
        if mode not in ["train", "validation"]:
            raise ValueError(f"mode must be train or validation, got {mode}")

        self.wav_len = wav_len
        self.num_per_epoch = num_per_epoch
        self.random_start = random_start
        self.default_fs = default_fs
        self.mode = mode
        self.normalize = normalize
        self.meta = []

        for csv_path in pair_csvs:
            df = pd.read_csv(csv_path).dropna(subset=["noisy_filepath", "clean_filepath"])
            for row in df.itertuples():
                uid = str(getattr(row, "uid", Path(row.noisy_filepath).stem))
                self.meta.append(
                    {
                        "id": uid,
                        "noisy_filepath": row.noisy_filepath,
                        "clean_filepath": row.clean_filepath,
                    }
                )

        if not self.meta:
            raise ValueError("No paired noisy/clean examples found in pair_csvs")

        print(f"Number of {mode} paired examples:", len(self.meta))
        self.sample_data_per_epoch(mode)

    def sample_data_per_epoch(self, mode="train"):
        if self.num_per_epoch <= 0 or self.num_per_epoch >= len(self.meta):
            self.meta_selected = list(self.meta)
            return

        if mode == "train":
            self.meta_selected = random.sample(self.meta, self.num_per_epoch)
        else:
            self.meta_selected = self.meta[: self.num_per_epoch]

    def _crop_or_pad_pair(self, noisy, clean, rng):
        length = min(noisy.shape[1], clean.shape[1])
        noisy = noisy[:, :length]
        clean = clean[:, :length]

        if self.wav_len == 0:
            return noisy, clean, length

        seg_len = int(self.wav_len * self.default_fs)
        if seg_len < length:
            if self.random_start:
                start = rng.integers(0, length - seg_len)
            else:
                start = 0
            noisy = noisy[:, start : start + seg_len]
            clean = clean[:, start : start + seg_len]
        elif seg_len > length:
            pad_points = seg_len - length
            noisy = np.pad(noisy, ((0, 0), (0, pad_points)), constant_values=0)
            clean = np.pad(clean, ((0, 0), (0, pad_points)), constant_values=0)

        return noisy, clean, length

    def __getitem__(self, idx):
        info = self.meta_selected[idx]
        noisy, _ = simulate_utils.read_audio(
            info["noisy_filepath"], force_1ch=True, fs=self.default_fs
        )
        clean, _ = simulate_utils.read_audio(
            info["clean_filepath"], force_1ch=True, fs=self.default_fs
        )

        if self.mode == "train":
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(idx)
        noisy, clean, orig_len = self._crop_or_pad_pair(noisy, clean, rng)

        if self.normalize:
            scale = 0.9 / (max(np.max(np.abs(noisy)), np.max(np.abs(clean))) + 1e-12)
            noisy = noisy * scale
            clean = clean * scale

        item_info = {"id": info["id"], "fs": self.default_fs, "length": orig_len}
        return noisy.astype(np.float32).squeeze(), clean.astype(np.float32).squeeze(), item_info

    def __len__(self):
        return len(self.meta_selected)
