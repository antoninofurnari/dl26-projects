"""
Evaluation entry point — reproduces the numbers in the README summary table.

    python -m src.evaluation.evaluate --config experiments/configs/model_v1.yaml

Loads the checkpoint named by the config, runs target evaluation, and prints
accuracy + per-source influence (for msda) or zero-shot accuracy (baseline).
"""

import argparse
from pathlib import Path

import torch

from src.utils.common import load_config, set_seed
from src.data.datasets import build_loaders
from src.data.class_mapping import NUM_CLASSES
from src.models.multisource_da import MultiSourceDA
from src.training.train import BaselineModel, evaluate_baseline, evaluate_msda


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    _, _, target_eval, feat_dim = build_loaders(cfg)
    ckpt_dir = Path(cfg.get("ckpt_dir", "experiments/checkpoints"))
    kind = cfg["mode"]
    ckpt = ckpt_dir / f"{kind}_{cfg.get('run_name', 'run')}.pt"
    print(f"[eval] loading {ckpt}")

    if kind == "baseline":
        model = BaselineModel(feat_dim, NUM_CLASSES, cfg["embed_dim"]).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        evaluate_baseline(model, target_eval, device)
    else:
        model = MultiSourceDA(feat_dim, NUM_CLASSES, cfg["sources"],
                              embed_dim=cfg["embed_dim"], num_domains=3).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        evaluate_msda(model, target_eval, cfg["sources"], device,
                      cfg.get("ensemble_temperature", 0.1))


if __name__ == "__main__":
    main()
