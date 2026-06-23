"""
Per-class analysis on the target domain.

Loads a trained MSDA checkpoint, runs the weighted ensemble on the target, and
reports:
  - overall accuracy
  - per-class accuracy (which actions the model gets right/wrong)
  - a confusion matrix (printed as text + saved as CSV)

Run:
    python -m src.evaluation.confusion --config experiments/configs/model_v1.yaml

Outputs go to experiments/confusion_<run_name>.csv and per_class_<run_name>.csv
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.utils.common import load_config, set_seed
from src.data.datasets import build_loaders
from src.data.class_mapping import NUM_CLASSES, IDX_TO_CLASS
from src.models.multisource_da import MultiSourceDA


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    temperature = cfg.get("ensemble_temperature", 0.1)

    _, _, target_eval, feat_dim = build_loaders(cfg)
    model = MultiSourceDA(feat_dim, NUM_CLASSES, cfg["sources"],
                          embed_dim=cfg["embed_dim"], num_domains=3).to(device)
    ckpt_dir = Path(cfg.get("ckpt_dir", "experiments/checkpoints"))
    ckpt = ckpt_dir / f"msda_{cfg.get('run_name', 'run')}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"[confusion] loaded {ckpt}")

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)  # rows=true, cols=pred
    for x, y, _ in target_eval:
        x = x.to(device)
        logits = model.ensemble_predict(x, temperature=temperature)
        preds = logits.argmax(1).cpu().numpy()
        ys = y.numpy()
        for t, p in zip(ys, preds):
            confusion[t, p] += 1

    total = confusion.sum()
    correct = np.trace(confusion)
    print(f"[confusion] overall target accuracy = {correct/total:.4f} "
          f"({correct}/{total})")

    # per-class accuracy
    print("\nPer-class accuracy (target):")
    per_class_rows = []
    for c in range(NUM_CLASSES):
        n = confusion[c].sum()
        acc = confusion[c, c] / n if n > 0 else float("nan")
        name = IDX_TO_CLASS[c]
        print(f"  {name:12s}  acc={acc:.3f}  ({confusion[c, c]}/{n})")
        per_class_rows.append((name, n, confusion[c, c], round(float(acc), 4)))

    # save outputs
    out_dir = Path("experiments")
    run = cfg.get("run_name", "run")
    np.savetxt(out_dir / f"confusion_{run}.csv", confusion, fmt="%d", delimiter=",",
               header=",".join(IDX_TO_CLASS[c] for c in range(NUM_CLASSES)), comments="")
    with open(out_dir / f"per_class_{run}.csv", "w") as f:
        f.write("class,n_clips,correct,accuracy\n")
        for name, n, corr, acc in per_class_rows:
            f.write(f"{name},{n},{corr},{acc}\n")

    # text confusion matrix (compact)
    print("\nConfusion matrix (rows=true, cols=pred), class order:")
    print("  " + " ".join(f"{IDX_TO_CLASS[c][:4]:>4s}" for c in range(NUM_CLASSES)))
    for c in range(NUM_CLASSES):
        row = " ".join(f"{confusion[c, j]:4d}" for j in range(NUM_CLASSES))
        print(f"{IDX_TO_CLASS[c][:4]:>4s} {row}")

    print(f"\n[confusion] saved -> experiments/confusion_{run}.csv, "
          f"experiments/per_class_{run}.csv")


if __name__ == "__main__":
    main()
