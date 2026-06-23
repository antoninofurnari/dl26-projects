#!/bin/bash
# =============================================================================
# Feature extraction with the INFLATED IMAGENET ResNet-50 backbone (no leakage).
# Produces features into features_imagenet/ so they don't overwrite the old ones.
#   sbatch run_extract_imagenet.sbatch
# =============================================================================
#SBATCH --job-name=msda-extract-in
#SBATCH --output=logs/extract_in_%j.out
#SBATCH --error=logs/extract_in_%j.err
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

set -e
mkdir -p logs

SIF=/shared/sifs/latest.sif
PROJECT=$HOME/DomainAdaptation-Track9-DataLost

WEIGHTS=$PROJECT/weights/resnet50_imagenet.pth
UCF_ROOT=$PROJECT/data/UCF-101
HMDB_ROOT=$PROJECT/data/HMDB51
KIN_ROOT=$PROJECT/data/kinetics400_5per/kinetics400_5per/train

run() { apptainer exec --nv "$SIF" bash -lc "cd $PROJECT && python -m $*"; }

run src.data.extract_features --dataset ucf101 \
    --video-root "$UCF_ROOT" --layout folders \
    --backbone inflated_resnet50 --weights-path "$WEIGHTS" --out-root features_imagenet
run src.data.extract_features --dataset hmdb51 \
    --video-root "$HMDB_ROOT" \
    --backbone inflated_resnet50 --weights-path "$WEIGHTS" --out-root features_imagenet
run src.data.extract_features --dataset kinetics \
    --video-root "$KIN_ROOT" \
    --backbone inflated_resnet50 --weights-path "$WEIGHTS" --out-root features_imagenet

echo "=== extraction (imagenet backbone) done ==="
