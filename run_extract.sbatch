#!/bin/bash
# =============================================================================
# Feature extraction only — runs INSIDE the Apptainer container.
# The slow, GPU-heavy, run-once step. Run first; training reuses features/.
#   sbatch run_extract.sbatch
# =============================================================================
#SBATCH --job-name=msda-extract
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00

set -e
mkdir -p logs

SIF=/shared/sifs/latest.sif
PROJECT=$HOME/DomainAdaptation-Track9-DataLost

WEIGHTS=$PROJECT/weights/r3d_18_kinetics.pth
UCF_ROOT=$PROJECT/data/UCF-101
HMDB_ROOT=$PROJECT/data/HMDB51
KIN_ROOT=$PROJECT/data/kinetics400_5per/kinetics400_5per/train

run() { apptainer exec --nv "$SIF" bash -lc "cd $PROJECT && python -m $*"; }

# UCF-101 full uses class-per-folder layout (NOT csv).
run src.data.extract_features --dataset ucf101 \
    --video-root "$UCF_ROOT" --layout folders \
    --weights-path "$WEIGHTS" --out-root features
run src.data.extract_features --dataset hmdb51 \
    --video-root "$HMDB_ROOT" \
    --weights-path "$WEIGHTS" --out-root features
run src.data.extract_features --dataset kinetics \
    --video-root "$KIN_ROOT" \
    --weights-path "$WEIGHTS" --out-root features

echo "=== extraction done ==="
