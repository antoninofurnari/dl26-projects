# Running on the OFFLINE cluster (Apptainer container)

The cluster is air-gapped, but the execution environment is a **prebuilt
Apptainer/Singularity container** (`/shared/sifs/latest.sif`) that already has
everything: Python 3.11, torch 2.7.1 (CUDA 11.8), torchvision 0.22.1, numpy,
pyyaml, av. **You do NOT need to build or transfer any Python environment.**

You only need to transfer two things from home:

1. The backbone weights (`r3d_18_kinetics.pth`) — see section A.
2. The datasets — unless they are already on `/shared/data` (check first!).

---

## A. Backbone weights (transfer from home)

The container is offline, so torchvision can't download the pretrained weights.
**At home:**
```bash
pip install torch torchvision        # if needed
python download_weights.py --out weights/r3d_18_kinetics.pth
```
Copy the file to the cluster, e.g.:
```bash
scp weights/r3d_18_kinetics.pth USER@gcluster:~/DomainAdaptation-Track9-DataLost/weights/
```
Every extract call uses it via `--weights-path` (already wired into the sbatch scripts).

---

## B. The environment — nothing to do

Everything runs inside the container. Interactively you can test with:
```bash
srun --account=dl-course-q2 --partition=dl-course-q2 --qos=gpu-xlarge \
     --gres=gpu:1 --pty bash
apptainer exec --nv /shared/sifs/latest.sif \
     python -c "import torch; print(torch.cuda.is_available())"
```
For real runs, use the sbatch scripts (section D) — they call
`apptainer exec --nv` for you. The `--nv` flag exposes the GPU to the container.

---

## C. Datasets

First check whether the course already placed them on the shared mount:
```bash
ls -la /shared/data
find /shared/data -maxdepth 3 -type d | head -40
```
If they're there, point the sbatch path variables at `/shared/data/...` and skip
the transfer. Otherwise copy them into `~/DomainAdaptation-Track9-DataLost/data/raw/`:
```
data/raw/UCF-101/train/*.avi   + data/raw/UCF-101/train.csv (and test)
data/raw/HMDB51/<class>/*.avi
data/raw/kinetics400_5per/train/<class>/*
```

---

## D. Running

Put the project in your home, edit the path variables at the top of the two
sbatch scripts if needed, then from the **login node**:

```bash
cd ~/DomainAdaptation-Track9-DataLost

# slow run-once feature extraction
sbatch run_extract.sbatch

# once features/ exists, the full pipeline (fast part) — or just submit
# run_pipeline.sbatch which also extracts if features are missing
sbatch run_pipeline.sbatch

squeue -u $USER                 # watch the queue
tail -f logs/msda_*.out         # follow progress
```

The sbatch scripts already contain your SLURM coordinates
(`--account=dl-course-q2 --partition=dl-course-q2 --qos=gpu-xlarge --gres=gpu:1`).
Adjust `--time` if extraction needs longer.

### Interactive debugging
To poke around before committing a job:
```bash
srun --account=dl-course-q2 --partition=dl-course-q2 --qos=gpu-xlarge \
     --gres=gpu:1 --pty bash
apptainer shell --nv /shared/sifs/latest.sif
cd ~/DomainAdaptation-Track9-DataLost
python -m src.data.verify_classes --dataset ucf101 --csv data/raw/UCF-101/train.csv
```
