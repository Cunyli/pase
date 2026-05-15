# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

import os
import shutil
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf


def build_run_paths(trainer_config, experiment):
    if not trainer_config["resume"]:
        run_id = f"{experiment}_{datetime.now().strftime('%Y-%m-%d-%Hh%Mm')}"
    else:
        run_id = f"{experiment}_{trainer_config['resume_datetime']}"

    run_root = Path(trainer_config.get("run_root", os.environ.get("PASE_RUN_ROOT", "./runs")))
    exp_path = Path(trainer_config.get("exp_path", run_root / experiment))
    if exp_path.name == experiment:
        exp_path = exp_path.parent / run_id
    else:
        exp_path = Path(f"{exp_path}_{run_id.rsplit('_', 1)[-1]}")

    checkpoint_root = trainer_config.get(
        "checkpoint_root",
        os.environ.get("PASE_GLOBAL_CHECKPOINT_DIR"),
    )
    checkpoint_path = (
        Path(checkpoint_root) / exp_path.name
        if checkpoint_root
        else exp_path / "checkpoints"
    )

    paths = {
        "exp_path": exp_path,
        "log_path": exp_path / "logs",
        "checkpoint_path": checkpoint_path,
        "sample_path": exp_path / "val_samples",
        "code_path": exp_path / "codes",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def snapshot_run(script_file, config_path, exp_path, code_path):
    repo_root = Path(script_file).resolve().parent.parent
    shutil.copy2(script_file, exp_path / Path(script_file).name)
    shutil.copy2(config_path, exp_path / "config.yaml")

    for file in repo_root.iterdir():
        if file.is_file():
            shutil.copy2(file, code_path / file.name)
    for dirname in ["configs", "loaders", "models", "train", "inference", "utils"]:
        shutil.copytree(repo_root / dirname, code_path / dirname, dirs_exist_ok=True)


def init_wandb_run(config, args, exp_path, log_path, checkpoint_path, default_experiment):
    trainer_config = config["trainer"]
    use_wandb = bool(trainer_config.get("use_wandb", False)) or os.environ.get("WANDB_ENABLE") == "1"
    if not use_wandb:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is enabled but not installed in this environment. "
            "Install wandb or set trainer.use_wandb=false."
        ) from exc

    repo_name = config.get("repo_name", "pase")
    experiment = config.get("experiment", default_experiment)
    dataset_type = config.get("dataset_type", "urgent2")
    timestamp = Path(exp_path).name.rsplit("_", 1)[-1]
    run_name = trainer_config.get("wandb_run_name") or (
        f"{repo_name}__{experiment}__{dataset_type}__{timestamp}"
    )
    tags = list(
        dict.fromkeys(
            [repo_name, experiment, dataset_type] + list(trainer_config.get("wandb_tags", []))
        )
    )
    wandb_config = OmegaConf.to_container(config, resolve=True)
    wandb_config.update(
        {
            "repo_name": repo_name,
            "experiment": experiment,
            "config_path": args.config,
            "log_dir": str(log_path),
            "checkpoint_dir": str(checkpoint_path),
            "train_dataset_type": dataset_type,
            "val_dataset_type": dataset_type,
            "max_epochs": trainer_config["epochs"],
            "devices": args.device,
        }
    )

    return wandb.init(
        project=trainer_config.get("wandb_project", "pase"),
        entity=trainer_config.get("wandb_entity"),
        name=run_name,
        dir=str(exp_path),
        config=wandb_config,
        tags=tags,
        mode=trainer_config.get("wandb_mode", "online"),
    )


def log_wandb(wandb_run, metrics, step):
    if wandb_run is not None:
        wandb_run.log({"epoch": step, **metrics}, step=step)
