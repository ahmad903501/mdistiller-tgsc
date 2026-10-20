"""Versioned, teacher-specific TGSC crop banks; read-only inputs are supported."""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import uuid

import torch
from torchvision import datasets, transforms

from .oracle import (CropTensorDataset, TeacherUnionCache, build_teacher_union_cache,
                     build_bestcrop_dataset_batchwise)
from .utils import pick_target_layer

SCHEMA = 2
ORACLE_FIELDS = ("use_gt_target", "w_cam", "w_sal", "temp", "gamma", "pct_lo", "pct_hi",
                 "stride", "k_per_img", "min_frac", "min_mean", "top_k_crops", "low_k_crops")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def writer_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Cache writer lock exists: {path}. Check for another preparation job or a stale lock.")
    try:
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def cifar_dataset(exp, train=True, download=False, limit=None):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(exp.data.mean, exp.data.std)])
    ds = datasets.CIFAR100(exp.data.root, train=train, download=download, transform=transform)
    if limit:
        if limit < 2:
            raise ValueError("Subset limit must be at least 2")
        ds.data = ds.data[:limit]
        ds.targets = ds.targets[:limit]
    return ds


def dataset_hash(dataset):
    h = hashlib.sha256(dataset.data.tobytes())
    h.update(json.dumps(dataset.targets).encode())
    return h.hexdigest()


def manifest(exp, teacher_checkpoint, dataset):
    from torchvision import __version__ as torchvision_version
    # Source identity prevents silent use of crops from different model/oracle behavior.
    source = hashlib.sha256()
    paths = [Path(__file__).with_name("oracle.py"), Path(__file__).with_name("utils.py")]
    paths += sorted((Path(__file__).parents[1] / "models" / "cifar").glob("*.py"))
    for path in paths:
        source.update(path.name.encode()); source.update(path.read_text(encoding="utf-8-sig").encode())
    return {"schema": SCHEMA, "teacher": exp.teacher_model,
            "checkpoint_sha256": file_hash(teacher_checkpoint),
            "dataset_sha256": dataset_hash(dataset), "count": len(dataset), "split": "train",
            "mean": list(exp.data.mean), "std": list(exp.data.std),
            "oracle": {k: getattr(exp.tgsc, k) for k in ORACLE_FIELDS},
            "source_sha256": source.hexdigest(), "torch": str(torch.__version__),
            "torchvision": str(torchvision_version)}


class CropBank:
    def __init__(self, identity, read_root, write_root=None):
        self.identity = identity
        self.key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        self.read_root = Path(read_root)
        self.write_root = Path(write_root) if write_root else None

    def path(self, resolution, write=False):
        root = self.write_root if write else self.read_root
        if root is None:
            raise ValueError("A writable cache root is required for preparation")
        return root / self.key / f"res{resolution}.pt"

    def find(self, resolution):
        candidates = [self.path(resolution)]
        if self.write_root is not None:
            candidates.append(self.path(resolution, write=True))
        return next((p for p in candidates if p.is_file()), None)

    def load(self, resolution):
        path = self.find(resolution)
        if path is None:
            raise FileNotFoundError(f"Missing TGSC cache {self.key}/res{resolution}.pt. Run run_tgsc.py prepare with this teacher/config and cache root.")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("manifest") != self.identity or payload.get("resolution") != resolution:
            raise ValueError(f"Cache identity/schema mismatch: {path}; regenerate rather than reuse legacy q-only data")
        state = payload["dataset"]
        required = {"crops", "labels", "scores", "source_indices", "quality_flags", "components", "boxes"}
        if not required.issubset(state):
            raise ValueError("Cache lacks raw r1/r2/r3 or crop identity; run prepare")
        n = state["labels"].numel()
        if n == 0 or any(v.shape[0] != n for v in state.values()):
            raise ValueError("Cache record counts differ or are empty")
        if state["crops"].shape != (n, 3, resolution, resolution) or state["components"].shape != (n, 3) or state["boxes"].shape != (n, 4):
            raise ValueError("Invalid crop/metric/box shapes")
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError("Non-finite cache values")
        if (state["components"] < 0).any() or ((state["scores"] < 0) | (state["scores"] > 1)).any():
            raise ValueError("Invalid oracle score ranges")
        if ((state["source_indices"] < 0) | (state["source_indices"] >= self.identity["count"])).any():
            raise ValueError("Invalid original source indices")
        if ((state["labels"] < 0) | (state["labels"] >= 100)).any():
            raise ValueError("Invalid CIFAR-100 labels")
        boxes = state["boxes"]
        if (boxes < 0).any() or (boxes > 32).any() or not torch.all(boxes[:, 2:] - boxes[:, :2] == resolution):
            raise ValueError("Crop boxes do not match the cached resolution")
        if not torch.allclose(state["components"].mean(1).clamp(0, 1), state["scores"], rtol=1e-5, atol=1e-6):
            raise ValueError("Stored q is inconsistent with its raw metric components")
        return CropTensorDataset.from_state_dict(state)

    def save(self, resolution, dataset):
        path = self.path(resolution, write=True)
        atomic_save({"manifest": self.identity, "resolution": resolution, "dataset": dataset.state_dict()}, path)
        # Human-readable identity belongs to the bank, independent of method/student.
        manifest_path = path.parent / "manifest.json"
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.identity, indent=2), encoding="utf-8")
        os.replace(temporary, manifest_path)


def prepare_bank(exp, teacher, dataset, bank):
    resolutions = sorted({r for r, _ in exp.data.stages})
    missing = []
    for size in resolutions:
        if bank.find(size):
            ds = bank.load(size)
            print(f"Reusing {bank.key}/res{size}: {len(ds)} crops", flush=True)
            del ds
        else:
            missing.append(size)
    if not missing:
        return
    estimate = len(dataset) * (exp.tgsc.top_k_crops + exp.tgsc.low_k_crops) * sum(3*r*r*4 for r in missing)
    print(f"Preparing sizes {missing}; crop-pixel upper estimate {estimate / 2**30:.2f} GiB", flush=True)
    output_dir = bank.path(missing[0], write=True).parent
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=exp.tgsc.precompute_batch_size, shuffle=False,
                        num_workers=exp.tgsc.precompute_workers, pin_memory=exp.data.pin_memory)
    oracle = asdict(exp.tgsc)
    full_keys = ("use_gt_target", "w_cam", "w_sal", "temp", "gamma", "pct_lo", "pct_hi")
    with writer_lock(output_dir / ".prepare.lock"):
        ref_path = output_dir / "reference.pt"
        if ref_path.exists():
            ref = torch.load(ref_path, map_location="cpu", weights_only=True)
            if ref["manifest"] != bank.identity:
                raise ValueError("Teacher reference cache identity mismatch")
            reference = TeacherUnionCache(ref["union_maps"], ref["logits"])
        else:
            reference = build_teacher_union_cache(loader, teacher, pick_target_layer(teacher),
                                                   **{k: oracle[k] for k in full_keys})
            atomic_save(dict(manifest=bank.identity, union_maps=reference.union_maps, logits=reference.logits), ref_path)
        crop_keys = (*ORACLE_FIELDS, "eval_chunk")
        for size in missing:
            # Another process may have completed this bank before we acquired the lock.
            if bank.find(size):
                continue
            ds = build_bestcrop_dataset_batchwise(loader, teacher, pick_target_layer(teacher), crop_size=size,
                    teacher_cache=reference, **{k: oracle[k] for k in crop_keys})
            bank.save(size, ds)
            print(f"Saved {len(ds)} crops: {bank.path(size, write=True)}", flush=True)
            del ds

