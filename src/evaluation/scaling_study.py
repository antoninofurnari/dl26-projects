"""
Extra objective — scaling study.

Trains the multi-source DA model with different subsets of sources and reports
how target accuracy scales with the number / choice of sources:

    single source: [hmdb51]      (no real multi-source benefit)
    single source: [ucf101]
    both sources:  [hmdb51, ucf101]

Run:
    python -m src.evaluation.scaling_study --config experiments/configs/model_v1.yaml

Writes a small results table to stdout and to experiments/scaling_results.csv.
"""

import argparse
import copy
import csv
from pathlib import Path

import torch

from src.utils.common import load_config, set_seed
from src.training.train import train_msda


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="experiments/scaling_results.csv")
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    device = base_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Source subsets to compare. A single-source run still uses the msda machinery
    # (encoder + one head + discriminator over 2 domains: that source + target).
    all_sources = base_cfg["sources"]
    subsets = [[s] for s in all_sources] + [list(all_sources)]

    rows = []
    for subset in subsets:
        cfg = copy.deepcopy(base_cfg)
        cfg["sources"] = subset
        cfg["run_name"] = f"scaling_{'_'.join(subset)}"
        set_seed(cfg.get("seed", 42))
        print(f"\n=== scaling run: sources={subset} ===")
        acc, ratio = train_msda(cfg, device)
        rows.append({
            "sources": "+".join(subset),
            "num_sources": len(subset),
            "target_acc": round(acc, 4),
            "influence": ratio,
        })

    print("\n=== SCALING STUDY SUMMARY ===")
    for r in rows:
        print(f"  {r['sources']:20s} (n={r['num_sources']}) "
              f"target_acc={r['target_acc']:.4f}  influence={r['influence']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sources", "num_sources", "target_acc", "influence"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[scaling] results saved -> {out}")


if __name__ == "__main__":
    main()
