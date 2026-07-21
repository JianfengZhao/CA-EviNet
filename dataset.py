"""Patient-level dataset for the JOINT two-task setting. Each item is one patient:
the available chamber views (each with T frames), BOTH task labels (etiology, MAC),
and the M normalized clinical measurements (with a NaN mask).

Frames were already sector-isolated, letterboxed, and resized to 224x224 by the
preprocessing pipeline. Here we only load and normalize them.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from monai.transforms import (
    Compose, ScaleIntensity, NormalizeIntensity, RandFlip, RandRotate, EnsureType,
)
import config as C


def _view_transform(train: bool):
    tfs = [ScaleIntensity(),  # -> [0,1]
           NormalizeIntensity(subtrahend=0.5, divisor=0.5)]  # -> ~[-1,1]
    if train:
        tfs = [RandFlip(prob=0.5, spatial_axis=2),
               RandRotate(range_z=0.087, prob=0.3, keep_size=True)] + tfs
    tfs.append(EnsureType(dtype=torch.float32))
    return Compose(tfs)


class CAEDataset(Dataset):
    def __init__(self, df, ids, quant_mean, quant_std, train=False):
        self.df = df.set_index(C.ID_COL)
        self.ids = [int(i) for i in ids]
        self.mean = quant_mean
        self.std = quant_std
        self.tf = _view_transform(train)

    def __len__(self):
        return len(self.ids)

    def _load_view(self, pid, view):
        d = os.path.join(C.DATA_ROOT, f"{pid:03d}", view)
        frames = []
        for t in range(C.T):
            p = os.path.join(d, f"frame_{t:02d}.png")
            img = np.asarray(Image.open(p).convert("L"), dtype=np.float32)  # H,W
            frames.append(img)
        arr = np.stack(frames, 0)[:, None, :, :]      # T,1,H,W
        return self.tf(arr)                            # tensor T,1,H,W

    def __getitem__(self, idx):
        pid = self.ids[idx]
        row = self.df.loc[pid]
        views = torch.stack([self._load_view(pid, v) for v in C.VIEWS], 0)  # V,T,1,H,W
        labels = {t["name"]: torch.tensor(int(row[f"label_{t['name']}"]), dtype=torch.long)
                  for t in C.TASKS}
        q = row[C.QUANT_COLS].to_numpy(dtype="float32")     # M (may contain NaN)
        mask = (~np.isnan(q)).astype("float32")
        qn = (np.nan_to_num(q, nan=0.0) - self.mean) / self.std
        qn = qn * mask                                       # zero-out missing
        item = {
            "views": views,                                  # V,T,1,H,W
            "quant": torch.tensor(qn, dtype=torch.float32),  # M (normalized)
            "quant_mask": torch.tensor(mask, dtype=torch.float32),
            "pid": pid,
        }
        for name, lab in labels.items():
            item[f"label_{name}"] = lab
        return item
