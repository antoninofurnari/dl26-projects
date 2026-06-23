#!/bin/bash
# =============================================================================
# Full multi-source DA pipeline — runs INSIDE the Apptainer container.
#   sbatch run_pipeline.sbatch
#   squeue -u $USER ; tail -f logs/msda_<jobid>.out
# =============================================================================
#SBATCH --job-name=msda-pipeline
#SBATCH --output=logs/msda_%j.out
#SBATCH --error=logs/msda_%j.err
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

WEIGHTS=$PROJECT/weights/r3d_18_kinetics.pth
UCF_ROOT=$PROJECT/data/UCF-101
HMDB_ROOT=$PROJECT/data/HMDB51
KIN_ROOT=$PROJECT/data/kinetics400_5per/kinetics400_5per/train

run() { apptainer exec --nv "$SIF" bash -lc "cd $PROJECT && python -m $*"; }

echo "=== gpu check ==="
apptainer exec --nv "$SIF" python -c "import torch; print('cuda:', torch.cuda.is_available())"

echo "=== verifying class mappings ==="
run src.data.verify_classes --dataset ucf101   --video-root "$UCF_ROOT" --layout folders
run src.data.verify_classes --dataset hmdb51   --video-root "$HMDB_ROOT"
run src.data.verify_classes --dataset kinetics --video-root "$KIN_ROOT"

echo "=== extracting features (skips datasets already cached) ==="
[ -f "$PROJECT/features/ucf101/index.csv" ]   || run src.data.extract_features --dataset ucf101 \
    --video-root "$UCF_ROOT" --layout folders --weights-path "$WEIGHTS" --out-root features
[ -f "$PROJECT/features/hmdb51/index.csv" ]   || run src.data.extract_features --dataset hmdb51 \
    --video-root "$HMDB_ROOT" --weights-path "$WEIGHTS" --out-root features
[ -f "$PROJECT/features/kinetics/index.csv" ] || run src.data.extract_features --dataset kinetics \
    --video-root "$KIN_ROOT" --weights-path "$WEIGHTS" --out-root features

echo "=== training baseline (objective 1) ==="
run src.training.train --config experiments/configs/baseline.yaml

echo "=== training multi-source DA (objectives 2-3) ==="
run src.training.train --config experiments/configs/model_v1.yaml

echo "=== evaluation (objective 4: source influence) ==="
run src.evaluation.evaluate --config experiments/configs/model_v1.yaml

echo "=== extra: source-drop robustness ==="
run src.training.train --config experiments/configs/model_drop.yaml

echo "=== extra: scaling study ==="
run src.evaluation.scaling_study --config experiments/configs/model_v1.yaml

echo "=== PIPELINE DONE ==="
