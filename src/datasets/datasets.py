"""
datasets.py — Multi-Source Domain Adaptation Dataset Pipeline
═══════════════════════════════════════════════════════════════
Source 1 : HMDB-51   → pre-extracted JPG frames
           data/HMDB51/<class>/<video_folder>/*.jpg
Source 2 : UCF-101   → raw .avi video files + CSV
           data/train/*.avi  +  data/train.csv
Target   : Kinetics  → raw videos organised by class
           data/kinetics400_5per/kinetics400_5per/train/<class>/

Each __getitem__ returns:
    frames  : Tensor (T, C, H, W)   T=NUM_FRAMES
    label   : int                    per-dataset label id
    domain  : int                    0=HMDB51 | 1=UCF101 | 2=Kinetics
"""

import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── Optional fast video backend ───────────────────────────────────────────────
try:
    from decord import VideoReader, cpu as decord_cpu
    _DECORD = True
except ImportError:
    try:
        import torchvision.io as _tvio
        _DECORD = False
    except ImportError as e:
        raise ImportError("Install decord or torchvision for video reading") from e


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DOMAIN_IDS: Dict[str, int] = {"hmdb51": 0, "ucf101": 1, "kinetics": 2}
NUM_FRAMES:  int = 16        # frames uniformly sampled per clip
FRAME_SIZE:  Tuple[int, int] = (112, 112)
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".webm", ".mov"}


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

def get_transform(train: bool = True) -> transforms.Compose:
    """Standard ImageNet-normalised transform for a single PIL frame."""
    if train:
        return transforms.Compose([
            transforms.Resize(128),
            transforms.RandomCrop(FRAME_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(FRAME_SIZE),
        transforms.CenterCrop(FRAME_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# LOW-LEVEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _sample_indices(total: int, n: int) -> List[int]:
    """Uniformly sample n frame indices from a clip of length total."""
    if total <= n:
        idx = list(range(total)) + [total - 1] * (n - total)
    else:
        idx = np.linspace(0, total - 1, n, dtype=int).tolist()
    return idx


def _load_frames_from_folder(folder: Path, n: int) -> List[Image.Image]:
    """
    HMDB-51 format: folder contains pre-extracted *.jpg frames.
    Returns n uniformly sampled PIL images.
    """
    files = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No image files found in {folder}")
    indices = _sample_indices(len(files), n)
    return [Image.open(files[i]).convert("RGB") for i in indices]


def _load_frames_from_video(path: Path, n: int) -> List[Image.Image]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(_sample_indices(max(total, 1), n))
    frames, idx = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        idx += 1
    cap.release()
    while len(frames) < n:
        frames.append(frames[-1] if frames else Image.new("RGB", FRAME_SIZE))
    return frames[:n]


def _frames_to_tensor(frames: List[Image.Image],
                      transform: transforms.Compose) -> torch.Tensor:
    """Apply transform to each frame → stack → (T, C, H, W)."""
    return torch.stack([transform(f) for f in frames])


# ══════════════════════════════════════════════════════════════════════════════
# LABEL-MAP BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_label_maps(
    hmdb_root: Path,
    ucf_csv:   Path,
    kin_root:  Path,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Returns three independent label-to-int dicts, one per dataset.
    Each dict maps class-name → integer id (alphabetically sorted).
    """
    # HMDB-51: class names from subdirectory names
    hmdb_classes = sorted(d.name for d in hmdb_root.iterdir() if d.is_dir())
    hmdb_map = {c: i for i, c in enumerate(hmdb_classes)}

    # UCF-101: class names from the 'tag' column of the CSV
    ucf_classes: set = set()
    with open(ucf_csv, newline="") as f:
        for row in csv.DictReader(f):
            ucf_classes.add(row["tag"])
    ucf_map = {c: i for i, c in enumerate(sorted(ucf_classes))}

    # Kinetics: class names from subdirectory names inside /train
    kin_train = kin_root / "train"
    kin_classes = sorted(d.name for d in kin_train.iterdir() if d.is_dir())
    kin_map = {c: i for i, c in enumerate(kin_classes)}

    return hmdb_map, ucf_map, kin_map


# ══════════════════════════════════════════════════════════════════════════════
# DATASET CLASSES
# ══════════════════════════════════════════════════════════════════════════════

class HMDB51Dataset(Dataset):
    """
    Source 1 — HMDB-51
    ─────────────────────────────────────────────────────────────
    root/
      <class>/
        <video_folder>/
          10000.jpg
          10002.jpg
          ...
    """
    DOMAIN_ID = DOMAIN_IDS["hmdb51"]

    def __init__(
        self,
        root:       str,
        label_map:  Dict[str, int],
        num_frames: int  = NUM_FRAMES,
        train:      bool = True,
    ):
        self.root       = Path(root)
        self.label_map  = label_map
        self.num_frames = num_frames
        self.transform  = get_transform(train)
        self.samples: List[Tuple[Path, int]] = []
        self._scan()

    def _scan(self):
        for cls_dir in sorted(self.root.iterdir()):
            if not cls_dir.is_dir() or cls_dir.name not in self.label_map:
                continue
            label = self.label_map[cls_dir.name]
            for vid_dir in cls_dir.iterdir():
                if vid_dir.is_dir() and any(vid_dir.glob("*.jpg")):
                    self.samples.append((vid_dir, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        vid_dir, label = self.samples[idx]
        frames = _load_frames_from_folder(vid_dir, self.num_frames)
        return _frames_to_tensor(frames, self.transform), label, self.DOMAIN_ID


class UCF101Dataset(Dataset):
    """
    Source 2 — UCF-101
    ─────────────────────────────────────────────────────────────
    video_dir/
      v_CricketShot_g08_c01.avi
      ...
    csv_path:  video_name,tag
    """
    DOMAIN_ID = DOMAIN_IDS["ucf101"]

    def __init__(
        self,
        video_dir:  str,
        csv_path:   str,
        label_map:  Dict[str, int],
        num_frames: int  = NUM_FRAMES,
        train:      bool = True,
    ):
        self.video_dir  = Path(video_dir)
        self.label_map  = label_map
        self.num_frames = num_frames
        self.transform  = get_transform(train)
        self.samples: List[Tuple[Path, int]] = []
        self._scan(csv_path)

    def _scan(self, csv_path: str):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                tag  = row["tag"]
                path = self.video_dir / row["video_name"]
                if path.exists() and tag in self.label_map:
                    self.samples.append((path, self.label_map[tag]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        frames = _load_frames_from_video(path, self.num_frames)
        return _frames_to_tensor(frames, self.transform), label, self.DOMAIN_ID


class KineticsDataset(Dataset):
    """
    Target — Kinetics (5 % subset)
    ─────────────────────────────────────────────────────────────
    root/
      train/
        <class>/
          xxxxxxx.mp4
          ...
      val/          (optional)
    """
    DOMAIN_ID = DOMAIN_IDS["kinetics"]

    def __init__(
        self,
        root:       str,
        label_map:  Dict[str, int],
        split:      str  = "train",   # "train" | "val"
        num_frames: int  = NUM_FRAMES,
        train:      bool = True,
    ):
        self.split_root = Path(root) / split
        self.label_map  = label_map
        self.num_frames = num_frames
        self.transform  = get_transform(train)
        self.samples: List[Tuple[Path, int]] = []
        self._scan()

    def _scan(self):
        for cls_dir in sorted(self.split_root.iterdir()):
            if not cls_dir.is_dir() or cls_dir.name not in self.label_map:
                continue
            label = self.label_map[cls_dir.name]
            for f in cls_dir.iterdir():
                if f.suffix.lower() in VIDEO_EXTS:
                    self.samples.append((f, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        frames = _load_frames_from_video(path, self.num_frames)
        return _frames_to_tensor(frames, self.transform), label, self.DOMAIN_ID


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-SOURCE DATALOADER
# ══════════════════════════════════════════════════════════════════════════════

class MultiSourceDataLoader:
    """
    Round-robin iterator over three DataLoaders.
    Each step yields a triple:
        (batch_s1, batch_s2, batch_target)
    where each batch is (frames, labels, domain_ids).

    Length = min(len(s1_loader), len(s2_loader), len(target_loader)).
    """

    def __init__(
        self,
        hmdb_ds:     HMDB51Dataset,
        ucf_ds:      UCF101Dataset,
        kinetics_ds: KineticsDataset,
        batch_size:  int = 8,
        num_workers: int = 4,
    ):
        _kw = dict(batch_size=batch_size, num_workers=num_workers,
                   pin_memory=True, drop_last=True)

        self.loader_s1  = DataLoader(hmdb_ds,     shuffle=True,  **_kw)
        self.loader_s2  = DataLoader(ucf_ds,      shuffle=True,  **_kw)
        self.loader_tgt = DataLoader(kinetics_ds, shuffle=False, **_kw)

    def __iter__(self):
        for b_s1, b_s2, b_tgt in zip(
            self.loader_s1, self.loader_s2, self.loader_tgt
        ):
            yield b_s1, b_s2, b_tgt

    def __len__(self):
        return min(
            len(self.loader_s1),
            len(self.loader_s2),
            len(self.loader_tgt),
        )

    # ── Convenience: separate target-only DataLoader for evaluation ───────────
    def get_target_eval_loader(self, batch_size: int = 16, num_workers: int = 4):
        return DataLoader(
            self.loader_tgt.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_dataloaders(
    data_root:   str,
    batch_size:  int = 8,
    num_workers: int = 4,
):
    """
    Main entry point — builds all datasets and returns a ready-to-use loader.

    Returns
    -------
    loader      : MultiSourceDataLoader
    hmdb_map    : Dict[str, int]  — HMDB-51  label → id
    ucf_map     : Dict[str, int]  — UCF-101  label → id
    kin_map     : Dict[str, int]  — Kinetics label → id
    """
    root     = Path(data_root)
    hmdb_dir = root / "HMDB51"
    ucf_dir  = root / "train"
    ucf_csv  = root / "train.csv"
    kin_dir  = root / "kinetics400_5per" / "kinetics400_5per"

    # ── Sanity checks ─────────────────────────────────────────────────────────
    for p in [hmdb_dir, ucf_dir, ucf_csv, kin_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Expected path not found: {p}")

    # ── Label maps ────────────────────────────────────────────────────────────
    hmdb_map, ucf_map, kin_map = build_label_maps(hmdb_dir, ucf_csv, kin_dir)

    print("=" * 50)
    print(f"  HMDB-51   : {len(hmdb_map):>4} classes")
    print(f"  UCF-101   : {len(ucf_map):>4} classes")
    print(f"  Kinetics  : {len(kin_map):>4} classes")
    print("=" * 50)

    # ── Datasets ──────────────────────────────────────────────────────────────
    hmdb_ds = HMDB51Dataset(str(hmdb_dir), hmdb_map, train=True)
    ucf_ds  = UCF101Dataset(str(ucf_dir),  str(ucf_csv), ucf_map, train=True)
    kin_ds  = KineticsDataset(str(kin_dir), kin_map, split="train", train=False)

    print(f"  HMDB-51   : {len(hmdb_ds):>6} clips loaded")
    print(f"  UCF-101   : {len(ucf_ds):>6} clips loaded")
    print(f"  Kinetics  : {len(kin_ds):>6} clips loaded")
    print("=" * 50)

    # ── Multi-source loader ───────────────────────────────────────────────────
    loader = MultiSourceDataLoader(
        hmdb_ds, ucf_ds, kin_ds,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    print(f"  Steps / epoch : {len(loader)}")
    print("=" * 50)

    return loader, hmdb_map, ucf_map, kin_map


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SMOKE-TEST
# Run from project root:
#   python src/datasets/datasets.py ../../data
#   python src/datasets/datasets.py  (defaults to ../../data)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    data_root = sys.argv[1] if len(sys.argv) > 1 else "../../data"

    loader, hmdb_map, ucf_map, kin_map = build_dataloaders(
        data_root, batch_size=4, num_workers=0  # num_workers=0 for debugging
    )

    print("\n[Smoke test] Fetching one triple-batch …")
    (f1, l1, d1), (f2, l2, d2), (ft, lt, dt) = next(iter(loader))

    print(f"  S1  frames={tuple(f1.shape)}  labels={l1.tolist()}  domain={d1.tolist()}")
    print(f"  S2  frames={tuple(f2.shape)}  labels={l2.tolist()}  domain={d2.tolist()}")
    print(f"  TGT frames={tuple(ft.shape)}  labels={lt.tolist()}  domain={dt.tolist()}")
    print("\n✅  datasets.py smoke test passed.")