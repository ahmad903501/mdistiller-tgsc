"""Training, epoch-boundary recovery, and method-aware inference exports."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, Sampler
from torchvision import transforms

from ..models import cifar_model_dict
from ..distillers import distiller_dict
from ..distillers.SimKD import SimKDProjector, classifier, simkd_predict
from .cache import atomic_save
from .experiment import method_config
from .utils import build_stage_plan


def build_model(name, num_classes=100):
    if name not in cifar_model_dict:
        raise ValueError(f"Unknown CIFAR model {name}; choose from {sorted(cifar_model_dict)}")
    return cifar_model_dict[name][0](num_classes=num_classes)


def load_teacher(model, path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("model", payload.get("state_dict", payload))
    state = {k.removeprefix("module.").removeprefix("model."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


def build_distiller(exp, method, options, teacher):
    student = build_model(exp.student_model, exp.num_classes)
    if exp.student_checkpoint:
        load_teacher(student, exp.student_checkpoint)
        student.requires_grad_(True)
    cfg = method_config(exp, method, options, student, teacher)
    return distiller_dict[method](student, teacher, cfg)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def environment():
    from torchvision import __version__ as tv
    from tqdm import __version__ as tqdm_version
    import sys
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[2],
                                           stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    return {"python": sys.version, "torch": str(torch.__version__), "torchvision": str(tv),
            "numpy": np.__version__, "tqdm": tqdm_version,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "code_revision": revision}


class MergeSingletonBatches(Sampler):
    """Keep all crops while avoiding a final singleton in pooled-feature BatchNorm."""
    def __init__(self, sampler, batch_size):
        self.sampler, self.batch_size = sampler, batch_size

    def __iter__(self):
        pending = []
        for index in self.sampler:
            pending.append(index)
            if len(pending) == self.batch_size + 2:
                yield pending[:self.batch_size]
                pending = pending[self.batch_size:]
        if pending:
            yield pending

    def __len__(self):
        n = len(self.sampler)
        return n // self.batch_size if n % self.batch_size == 1 and n > 1 else (n + self.batch_size-1)//self.batch_size


def training_loader(dataset, exp, resolution, epoch):
    operations = []
    if exp.tgsc.aug_padding:
        operations.append(transforms.RandomCrop(resolution, padding=exp.tgsc.aug_padding, padding_mode="reflect"))
    if exp.tgsc.flip_prob:
        operations.append(transforms.RandomHorizontalFlip(exp.tgsc.flip_prob))
    dataset.set_transform(transforms.Compose(operations) if operations else None)
    gen = torch.Generator().manual_seed(exp.train.seed + epoch)
    batches = MergeSingletonBatches(RandomSampler(dataset, generator=gen), exp.data.batch_size)
    return DataLoader(dataset, batch_sampler=batches, num_workers=exp.data.num_workers,
                      pin_memory=exp.data.pin_memory, generator=gen)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n, top1, top5, loss_sum = 0, 0, 0, 0.
    for image, labels in loader:
        image, labels = image.to(device), labels.to(device)
        out = model(image=image) if hasattr(model, "forward_train") else model(image)
        if isinstance(out, tuple):
            out = out[0]
        loss_sum += float(nn.functional.cross_entropy(out, labels, reduction="sum"))
        pred = out.topk(5, dim=1).indices
        top1 += int((pred[:, 0] == labels).sum())
        top5 += int((pred == labels[:, None]).any(1).sum())
        n += labels.numel()
    if not n:
        raise ValueError("Empty evaluation dataset")
    return {"top1": 100*top1/n, "top5": 100*top5/n, "ce": loss_sum/n, "samples": n}


def rng_state():
    np_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def export_payload(distiller, exp, method, options):
    def cpu_state(module):
        return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
    result = {"format": "tgsc-inference-v1", "method": method, "student_model": exp.student_model,
              "teacher_model": exp.teacher_model, "num_classes": exp.num_classes,
              "student": cpu_state(distiller.student), "mean": list(exp.data.mean), "std": list(exp.data.std)}
    if method == "SIMKD":
        result.update(projector=cpu_state(distiller.projector), head=cpu_state(distiller.head),
                      teacher_shapes=distiller.teacher_shapes, factor=options.get("factor", 2))
    return result


class SimKDInference(nn.Module):
    def __init__(self, payload):
        super().__init__()
        self.student = build_model(payload["student_model"], payload["num_classes"])
        self.student.load_state_dict(payload["student"])
        weights = payload["projector"]
        self.projector = SimKDProjector(s_n=weights["transfer.0.weight"].shape[1],
                         t_n=weights["transfer.6.weight"].shape[0], factor=payload["factor"])
        self.projector.load_state_dict(weights)
        # Construct only the architecture's classifier; no teacher weights/forward are required.
        self.head = copy.deepcopy(classifier(build_model(payload["teacher_model"], payload["num_classes"])))
        self.head.load_state_dict(payload["head"])
        self.teacher_shapes = payload["teacher_shapes"]

    def forward(self, image):
        return simkd_predict(self.student, self.projector, self.head, self.teacher_shapes, image)


def load_export(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "tgsc-inference-v1":
        raise ValueError("Expected a TGSC inference export, not a training checkpoint")
    if payload["method"] == "SIMKD":
        model = SimKDInference(payload)
    else:
        model = build_model(payload["student_model"], payload["num_classes"])
        model.load_state_dict(payload["student"])
    return model.eval(), payload


def resume_contract(config):
    """Paths may relocate between Kaggle sessions; scientific settings must not."""
    config = copy.deepcopy(config)
    exp = config["experiment"]
    for key in ("teacher_checkpoint", "student_checkpoint", "device"):
        exp.pop(key, None)
    exp["data"].pop("root", None)
    for key in ("save_path", "log_dir", "run_name"):
        exp["train"].pop(key, None)
    for key in ("cache_dir", "cache_to_disk", "cache"):
        exp["tgsc"].pop(key, None)
    return config


def train(exp, method, options, distiller, bank, test_dataset, device, output,
          resume=None, stop_after_epoch=None, launch_args=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "latest.pt").exists() and not resume:
        raise FileExistsError("Output already has a checkpoint; use --resume or a new output directory")
    distiller.to(device)
    parameters = [p for p in distiller.get_learnable_parameters() if p.requires_grad]
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("Optimizer parameter registration contains duplicates")
    optimizer = torch.optim.SGD(parameters, lr=exp.optim.base_lr, momentum=exp.optim.momentum,
                                weight_decay=exp.optim.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(exp.optim.milestones), exp.optim.milestone_gamma)
    run_config = {"experiment": asdict(exp), "method": method, "options": options}
    student_parameters = sum(p.numel() for p in distiller.student.parameters())
    extra_parameters = distiller.get_extra_parameters()
    provenance = {"config": run_config, "cache": bank.identity, "environment": environment(),
                  "launch_args": launch_args,
                  "parameters": {"student": student_parameters,
                                 "extra": extra_parameters,
                                 "trainable": sum(p.numel() for p in parameters),
                                 "deployed": student_parameters + (extra_parameters if method == "SIMKD" else 0)}}
    start, best, best_inference = 1, -1., None
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if resume_contract(state["config"]) != resume_contract(run_config) or state["cache"] != bank.identity:
            raise ValueError("Resume config/cache differs; only output/data/cache/checkpoint path relocation is allowed")
        distiller.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, best = state["epoch"] + 1, state["best_top1"]
        best_inference = state.get("best_inference")
        if best_inference is not None:
            atomic_save(best_inference, output / "inference_best.pt")
        restore_rng(state["rng"])
    (output / "run.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    test_loader = DataLoader(test_dataset, batch_size=exp.data.batch_size, num_workers=exp.data.num_workers)
    plan = build_stage_plan(exp.data.stages)
    current, crops = None, None
    for epoch in range(start, exp.train.epochs + 1):
        resolution = plan[epoch-1]
        if current != resolution:
            if crops is not None:
                del crops
            crops, current = bank.load(resolution), resolution
        loader = training_loader(crops, exp, resolution, epoch)
        distiller.train()
        totals, diagnostics, n, gate_sum = {}, {}, 0, 0.
        saturated, q_min, q_max = 0, float("inf"), float("-inf")
        begin = time.monotonic()
        for batch in loader:
            image, labels = batch["image"].to(device), batch["target"].to(device)
            metadata = {k: batch[k].to(device) for k in ("q", "r1", "r2", "r3")}
            optimizer.zero_grad(set_to_none=True)
            _, losses = distiller(image=image, target=labels, epoch=epoch, tgsc=metadata,
                                  gate_threshold=exp.tgsc.score_high, gate_mode=options.get("gate_mode", "unified"))
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, crop size {resolution}")
            loss.backward(); optimizer.step()
            count = len(labels)
            for k, value in losses.items():
                totals[k] = totals.get(k, 0.) + float(value.detach()) * count
            for k, value in getattr(distiller, "tgsc_diagnostics", {}).items():
                diagnostics[k] = diagnostics.get(k, 0.) + float(value) * count
            for k in ("r1", "r2", "r3", "q"):
                diagnostics[k+"_mean"] = diagnostics.get(k+"_mean", 0.) + float(metadata[k].sum())
            saturated += int((metadata["q"] >= exp.tgsc.score_high).sum())
            q_min = min(q_min, float(metadata["q"].min()))
            q_max = max(q_max, float(metadata["q"].max()))
            n += count
            gate_sum += float((metadata["q"] / exp.tgsc.score_high).clamp(max=1).sum())
        metrics = evaluate(distiller, test_loader, device)
        improved = metrics["top1"] > best
        best = max(best, metrics["top1"])
        row = {"epoch": epoch, "resolution": resolution, "lr": optimizer.param_groups[0]["lr"],
               "train": {k:v/n for k,v in totals.items()}, "gate_mean": gate_sum/n,
               "diagnostics": {k:v/n for k,v in diagnostics.items()},
               "gate_saturated_fraction": saturated/n, "q_min": q_min, "q_max": q_max,
               "unique_source_images": int(crops.source_indices.unique().numel()),
               "source_draws": n, "optimizer_steps": len(loader), "test": metrics,
               "seconds": time.monotonic()-begin}
        scheduler.step()
        inference = export_payload(distiller, exp, method, options)
        if improved:
            best_inference = inference
        state = {**provenance, "epoch": epoch, "best_top1": best, "best_inference": best_inference,
                 "model": distiller.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": rng_state()}
        atomic_save(state, output / "latest.pt")
        atomic_save(inference, output / "inference_latest.pt")
        if improved:
            atomic_save(state, output / "best.pt")
            atomic_save(inference, output / "inference_best.pt")
        if epoch % 40 == 0:
            atomic_save(state, output / f"epoch_{epoch}.pt")
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if stop_after_epoch and epoch >= stop_after_epoch:
            break
    return best
