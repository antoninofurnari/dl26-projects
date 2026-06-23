"""
Run this FROM HOME (a machine with internet) to download backbone weights,
then copy the resulting file(s) to the offline cluster.

R3D-18 (Kinetics) -- the OLD backbone (has target leakage):
    python download_weights.py --backbone r3d_18 --out weights/r3d_18_kinetics.pth

ResNet-50 (ImageNet) -- the NEW backbone for the inflated I3D model (no leakage):
    python download_weights.py --backbone resnet50 --out weights/resnet50_imagenet.pth

On the cluster, pass the path via --weights-path to extract_features:
    python -m src.data.extract_features ... --backbone inflated_resnet50 \
        --weights-path weights/resnet50_imagenet.pth
"""

import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50", choices=["r3d_18", "resnet50"])
    ap.add_argument("--out", default="weights/resnet50_imagenet.pth")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.backbone == "r3d_18":
        from torchvision.models.video import r3d_18, R3D_18_Weights
        print("[download] fetching R3D-18 Kinetics-400 weights...")
        model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        torch.save(model.state_dict(), out)
    else:
        from torchvision.models import resnet50, ResNet50_Weights
        print("[download] fetching ResNet-50 ImageNet weights...")
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        torch.save(model.state_dict(), out)

    print(f"[download] saved -> {out}")
    print("[download] copy this file to the cluster and use --weights-path")


if __name__ == "__main__":
    main()
