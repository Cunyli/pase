# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

from pathlib import Path
import sys


def _add_use_simulation_to_path(use_simulation_root):
    root = Path(use_simulation_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"USE_simulation repo not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


class UseSimulationOnlineSpeechDataset:
    def __new__(
        cls,
        use_simulation_root,
        clean_json,
        noise_json,
        rir_json,
        wind_noise_json=None,
        simulation_config=None,
        quality_json=None,
        wav_len=4,
        num_per_epoch=10000,
        random_start=False,
        default_fs=16000,
        mode="train",
        normalize=True,
        vqscore_threshold=None,
        use_dnsmos_filter=False,
        missing_scores="keep",
        seed=0,
    ):
        _add_use_simulation_to_path(use_simulation_root)

        from use_simulation_datasets import OnlineDegradationDataset

        return OnlineDegradationDataset(
            clean_json=clean_json,
            noise_json=noise_json,
            rir_json=rir_json,
            wind_noise_json=wind_noise_json,
            simulation_config=simulation_config,
            quality_json=quality_json,
            wav_len=wav_len,
            num_per_epoch=num_per_epoch,
            random_start=random_start,
            target_sample_rate=default_fs,
            mode=mode,
            normalize=normalize,
            vqscore_threshold=vqscore_threshold,
            use_dnsmos_filter=use_dnsmos_filter,
            missing_scores=missing_scores,
            seed=seed,
        )


class UseSimulationFixedPairSpeechDataset:
    def __new__(
        cls,
        use_simulation_root,
        pair_manifest,
        wav_len=4,
        num_per_epoch=0,
        random_start=False,
        default_fs=16000,
        mode="train",
        normalize=True,
        seed=0,
    ):
        _add_use_simulation_to_path(use_simulation_root)

        from use_simulation_datasets import FixedPairDataset

        return FixedPairDataset(
            pair_manifest=pair_manifest,
            wav_len=wav_len,
            num_per_epoch=num_per_epoch,
            random_start=random_start,
            target_sample_rate=default_fs,
            mode=mode,
            normalize=normalize,
            seed=seed,
        )


class UseSimulationCleanSpeechDataset:
    def __new__(
        cls,
        use_simulation_root,
        speech_csvs=None,
        clean_json=None,
        wav_len=4,
        num_per_epoch=0,
        random_start=False,
        default_fs=16000,
        mode="train",
        normalize=True,
        seed=0,
        dnsmos_threshold=3.0,
    ):
        _add_use_simulation_to_path(use_simulation_root)

        from use_simulation_datasets import CleanSpeechDataset

        return CleanSpeechDataset(
            speech_csvs=speech_csvs,
            clean_json=clean_json,
            wav_len=wav_len,
            num_per_epoch=num_per_epoch,
            random_start=random_start,
            target_sample_rate=default_fs,
            mode=mode,
            normalize=normalize,
            seed=seed,
            dnsmos_threshold=dnsmos_threshold,
        )


UseSimulationSpeechDataset = UseSimulationOnlineSpeechDataset
