# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

import argparse

import torch
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description="Export a vocoder checkpoint for inference.")
    parser.add_argument("--train-config", required=True, help="Training config used for the checkpoint.")
    parser.add_argument("--train-checkpoint", required=True, help="Training checkpoint to export.")
    parser.add_argument("--output", required=True, help="Output checkpoint path.")
    parser.add_argument(
        "--config-key",
        default="decoder_config",
        help="Config key to store under cfg in the exported checkpoint.",
    )
    args = parser.parse_args()

    config = OmegaConf.to_container(OmegaConf.load(args.train_config), resolve=True)
    state_dict = torch.load(args.train_checkpoint, map_location="cpu")

    if "generator" in state_dict:
        model_dict = state_dict["generator"]
    elif "model" in state_dict:
        model_dict = state_dict["model"]
    else:
        raise ValueError("Checkpoint must contain a 'generator' or 'model' state dict.")

    exported = {
        "model": model_dict,
        "cfg": config[args.config_key],
    }
    torch.save(exported, args.output)
    print("Successfully created a pre-trained checkpoint:", args.output)
    print("Keys:", exported.keys())


if __name__ == "__main__":
    main()
