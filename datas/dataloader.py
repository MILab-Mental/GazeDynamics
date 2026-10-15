"""One sample = one scanpath, its matching native NIfTI, and finding labels.

ctscangaze.csv paths are resolved relative to its directory. CT tensors use
[C, X, Y, Z] and scanpaths use [N, 4] (x_norm, y_norm, z_norm, seconds).
No axis canonicalization, physical-spacing resampling, or trajectory truncation
is performed. An optional CT resize preserves the normalized coordinate frame.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from collections.abc import Callable, Sequence

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

DEFAULT_INDEX = Path(__file__).resolve().parent / "DATASET" / "ctscangaze.csv"


class CTScanGazeDataset(Dataset):
    """Load a complete trajectory and CT; unknown findings have label_mask=False.

    Args:
        csv_path: index containing exactly ct,split,traj,label columns.
        split: train, val, test, or None to include all splits.
        ct_size: optional (X,Y,Z) output size; None keeps native dimensions.
        intensity_window: (-175,250) matches the repository feature extractor.
            Clip and scale to [0,1]; None returns stored intensities unchanged.
            Published CTs already have series-specific window clipping.
        class_names: official finding vocabulary. If omitted, load the adjacent
            label_schema.json. Unknown labels become -1 with a False mask.
        transform: optional JOINT transform of the sample dictionary. Spatial
            flips/rotations must update trajectory coordinates together with CT.

    Each sample contains ct, traj, traj_voxel, traj_ct_voxel, duration, t_start,
    t_end, label, label_mask, label_known, label_names, affine, native_affine,
    native_shape, ct_id, ct_prefix,
    split, ct_path, traj_path, length, subject, and source_group.
    """

    def __init__(self, csv_path: str | Path = DEFAULT_INDEX, split: str | None = "train",
                 ct_size: Sequence[int] | None = None,
                 intensity_window: tuple[float, float] | None = (-175., 250.),
                 class_names: Sequence[str] | None = None,
                 transform: Callable | None = None):
        self.csv_path = Path(csv_path).resolve()
        if split not in (None, "train", "val", "test"):
            raise ValueError("split must be train, val, test, or None")
        self.ct_size = tuple(ct_size) if ct_size is not None else None
        if self.ct_size is not None and (len(self.ct_size) != 3 or
                    any(not isinstance(n, int) or isinstance(n, bool) or n < 2 for n in self.ct_size)):
            raise ValueError("ct_size must contain three integers >= 2")
        if intensity_window is not None:
            lo, hi = intensity_window
            if not np.isfinite([lo, hi]).all() or hi <= lo:
                raise ValueError("intensity_window must contain finite increasing bounds")
        self.intensity_window = intensity_window
        self.transform = transform
        with self.csv_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ["ct", "split", "traj", "label"]:
                raise ValueError("CSV header must be ct,split,traj,label")
            rows = list(reader)
        schema = self.csv_path.parent / "label_schema.json"
        if class_names is None:
            if schema.exists():
                class_names = json.loads(schema.read_text())["classes"]
            else:
                # Vocabulary comes from the ENTIRE index, never a single split.
                class_names = sorted({name for r in rows for name in (json.loads(r["label"]) or [])})
        self.class_names = tuple(class_names)
        if any(not isinstance(c, str) or not c for c in self.class_names) or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must be unique nonempty strings")
        self.class_to_index = {name: i for i, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        prefix_state, recording_splits, seen_trajectories = {}, {}, set()
        self.rows = []
        for row in rows:
            if row["split"] not in ("train", "val", "test"):
                raise ValueError(f"Invalid split: {row['split']}")
            ct_path = self._resolve(row["ct"])
            traj_path = self._resolve(row["traj"])
            name = ct_path.name.removesuffix(".nii.gz").removesuffix(".nii")
            parts = name.split("_")
            if len(parts) != 3 or not all(p.isdigit() for p in parts):
                raise ValueError(f"Unexpected CT filename: {ct_path.name}")
            prefix = "_".join(parts[:2])
            labels = json.loads(row["label"])
            if labels is not None:
                if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
                    raise ValueError("label must be JSON [finding names], or null")
                labels = sorted(set(labels))
                unknown = set(labels) - self.class_to_index.keys()
                if unknown:
                    raise ValueError(f"Findings absent from class vocabulary: {sorted(unknown)}")
            state = (row["split"], labels)
            if prefix in prefix_state and prefix_state[prefix] != state:
                raise ValueError(f"Split or label differs across resolutions of CT {prefix}")
            prefix_state[prefix] = state
            if parts[0] in recording_splits and recording_splits[parts[0]] != row["split"]:
                raise ValueError(f"Recording {parts[0]} leaks across splits")
            recording_splits[parts[0]] = row["split"]
            if traj_path in seen_trajectories:
                raise ValueError(f"Duplicate trajectory in index: {traj_path}")
            seen_trajectories.add(traj_path)
            if not ct_path.is_file() or not traj_path.is_file():
                raise FileNotFoundError(f"Missing CT or trajectory: {ct_path}, {traj_path}")
            if not traj_path.name.startswith(name + "__"):
                raise ValueError(f"Trajectory filename does not match CT: {traj_path.name}")
            if split is None or row["split"] == split:
                self.rows.append({**row, "ct_path": ct_path, "traj_path": traj_path,
                                  "ct_id": name, "ct_prefix": prefix, "label_names": labels})
        # Verify DICOM study grouping when generated audit metadata is present.
        groups_file = self.csv_path.parent / "processing_audit" / "ct_groups_and_labels.csv"
        if groups_file.exists():
            study_splits = {}
            with groups_file.open(newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r["ct_prefix"] not in prefix_state:
                        continue
                    actual_split = prefix_state[r["ct_prefix"]][0]
                    if actual_split != r["split"]:
                        raise ValueError(f"Split disagrees with group audit: {r['ct_prefix']}")
                    uid = r["study_uid"]
                    if uid and uid in study_splits and study_splits[uid] != actual_split:
                        raise ValueError("One DICOM study leaks across splits")
                    if uid:
                        study_splits[uid] = actual_split

    def _resolve(self, value):
        p = Path(value)
        return p.resolve() if p.is_absolute() else (self.csv_path.parent / p).resolve()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = nib.load(row["ct_path"])
        shape = np.asarray(image.shape)
        if len(shape) != 3 or np.any(shape < 2):
            raise ValueError(f"Invalid CT shape: {image.shape}")
        fields = ["x", "y", "z", "duration", "t_start", "t_end",
                  "x_norm", "y_norm", "z_norm", "source_start", "source_end"]
        with row["traj_path"].open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != fields:
                raise ValueError(f"Invalid processed trajectory header: {row['traj_path']}")
            values = np.asarray([[float(r[k]) for k in fields] for r in reader], dtype=np.float64)
        if values.ndim != 2 or len(values) < 3 or not np.isfinite(values).all():
            raise ValueError(f"Invalid trajectory: {row['traj_path']}")
        xyz, duration = values[:, :3], values[:, 3]
        normalized = values[:, 6:9]
        if (xyz < 0).any() or (xyz > shape - 1).any() or (duration <= 0).any():
            raise ValueError("Trajectory has out-of-volume coordinates or invalid durations")
        if not np.allclose(normalized, xyz / (shape - 1), rtol=1e-6, atol=1e-6):
            raise ValueError("Normalized trajectory coordinates disagree with CT dimensions")
        if (values[:, 4] < 0).any() or (np.diff(values[:, 4]) < 0).any() or not np.allclose(
                values[:, 5] - values[:, 4], duration, rtol=1e-6, atol=1e-6):
            raise ValueError("Trajectory dwell-time intervals are inconsistent")
        array = image.get_fdata(dtype=np.float32, caching="unchanged")
        if not np.isfinite(array).all():
            raise ValueError(f"CT contains nonfinite intensities: {row['ct_path']}")
        if self.intensity_window is not None:
            lo, hi = self.intensity_window
            np.clip(array, lo, hi, out=array)
            array -= lo
            array /= hi - lo
        ct = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)
        if self.ct_size is not None and tuple(shape) != self.ct_size:
            ct = F.interpolate(ct.unsqueeze(0), size=self.ct_size, mode="trilinear",
                               align_corners=True).squeeze(0)
        output_shape = np.asarray(ct.shape[1:])
        affine = image.affine.copy()
        affine[:3, :3] = affine[:3, :3] @ np.diag((shape-1) / (output_shape-1))
        known = row["label_names"] is not None
        label = torch.zeros(self.num_classes) if known else torch.full((self.num_classes,), -1.)
        if known:
            for name in row["label_names"]:
                label[self.class_to_index[name]] = 1.
        stem = row["traj_path"].stem.split("__")
        sample = {"ct": ct,
                  "traj": torch.from_numpy(np.column_stack((normalized, duration)).astype(np.float32)),
                  "traj_voxel": torch.from_numpy(xyz.astype(np.float32)),
                  "traj_ct_voxel": torch.from_numpy((normalized * (output_shape-1)).astype(np.float32)),
                  "duration": torch.from_numpy(duration.astype(np.float32)),
                  "t_start": torch.from_numpy(values[:, 4].astype(np.float32)),
                  "t_end": torch.from_numpy(values[:, 5].astype(np.float32)),
                  "source_spans": torch.from_numpy(values[:, 9:11].astype(np.int64)),
                  "label": label, "label_mask": torch.full((self.num_classes,), known, dtype=torch.bool),
                  "label_known": known, "label_names": row["label_names"],
                  "affine": torch.from_numpy(affine),
                  "native_affine": torch.from_numpy(image.affine.copy()),
                  "native_shape": torch.tensor(shape, dtype=torch.long), "length": len(values),
                  "subject": stem[2].removeprefix("s"), "source_group": stem[1].removeprefix("g"),
                  **{k: row[k] for k in ("ct_id", "ct_prefix", "split")},
                  "ct_path": str(row["ct_path"]), "traj_path": str(row["traj_path"])}
        return self.transform(sample) if self.transform is not None else sample

    @staticmethod
    def collate_fn(batch):
        """Pad complete scanpaths; CTs stack if equal-shaped, otherwise stay a list.

        traj_mask denotes valid fixations; padding is zero and must be masked.
        Use ct_size for a dense CT batch. ct_list=True signals variable CT shapes.
        """
        if not batch:
            raise ValueError("Cannot collate an empty batch")
        result = {}
        lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
        for key in ("traj", "traj_voxel", "traj_ct_voxel", "duration", "t_start", "t_end", "source_spans"):
            result[key] = pad_sequence([b[key] for b in batch], batch_first=True, padding_value=0)
        result["lengths"] = lengths
        result["traj_mask"] = torch.arange(int(lengths.max()))[None, :] < lengths[:, None]
        cts = [b["ct"] for b in batch]
        equal = len({tuple(t.shape) for t in cts}) == 1
        result["ct"] = torch.stack(cts) if equal else cts
        result["ct_list"] = not equal
        for key in ("label", "label_mask", "affine", "native_affine", "native_shape"):
            result[key] = torch.stack([b[key] for b in batch])
        result["label_known"] = torch.tensor([b["label_known"] for b in batch], dtype=torch.bool)
        for key in ("label_names", "ct_id", "ct_prefix", "split", "ct_path", "traj_path", "subject", "source_group"):
            result[key] = [b[key] for b in batch]
        return result


# Convenient spelling matching the reference repository.
CTScanGaze = CTScanGazeDataset
