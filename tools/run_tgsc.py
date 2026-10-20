"""One entry point for local and notebook TGSC runs; select CUDA before torch import."""
import argparse
import json
import os
from pathlib import Path
import sys
from dataclasses import asdict


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("check", "prepare", "train", "evaluate"))
    p.add_argument("--config", required=True)
    p.add_argument("--gpu", type=int, default=0, help="Index within inherited CUDA_VISIBLE_DEVICES, or host CUDA-visible index")
    p.add_argument("--cpu-check", action="store_true", help="Allow CPU for check only, never training")
    p.add_argument("--teacher-checkpoint")
    p.add_argument("--data-root")
    p.add_argument("--cache-root", default="tgsc_cache", help="Read-only input bank parent, not the bank's hash directory")
    p.add_argument("--cache-write-root", help="Writable cache parent for prepare; defaults to cache-root")
    p.add_argument("--output-dir")
    p.add_argument("--seed", type=int)
    p.add_argument("--resume")
    p.add_argument("--export", help="Inference export for evaluate")
    p.add_argument("--download", action="store_true", help="Explicitly download CIFAR into data-root if absent")
    p.add_argument("--limit", type=int, help="Small data prefix for smoke checks; creates a distinct cache")
    p.add_argument("--stop-after-epoch", type=int, help="Save/stop at this completed epoch without changing scientific config")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.gpu < 0:
        raise ValueError("--gpu must be a nonnegative visible-device index")
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = inherited.split(",") if inherited is not None else None
    if args.cpu_check:
        selected = "cpu"
    elif devices is not None:
        if args.gpu >= len(devices) or not devices[args.gpu].strip() or devices[args.gpu].strip() == "-1":
            raise ValueError("Selected GPU is outside inherited CUDA_VISIBLE_DEVICES")
        selected = devices[args.gpu].strip()
    else:
        selected = str(args.gpu)
    if args.cpu_check:
        if args.command != "check":
            raise ValueError("--cpu-check is allowed only with check")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = selected
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from torch.utils.data import DataLoader
    from mdistiller.tgsc.experiment import load_run
    from mdistiller.tgsc.cache import cifar_dataset, manifest, CropBank, prepare_bank
    from mdistiller.tgsc.runtime import (seed_everything, environment, build_model, load_teacher,
                                      build_distiller, load_export, evaluate, train)
    exp, method, options, raw = load_run(args.config)
    if args.data_root:
        exp.data.root = str(Path(args.data_root).resolve())
    if args.seed is not None:
        exp.train.seed = args.seed
    seed_everything(exp.train.seed)
    if not args.cpu_check and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA GPU {selected!r} is unavailable; training never falls back to CPU. Use check --cpu-check for offline validation.")
    device = torch.device("cpu" if args.cpu_check else "cuda:0")
    if args.cpu_check:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    print(json.dumps({"inherited_visibility": inherited, "selected": selected,
                      "device": str(device), "environment": environment()}), flush=True)
    if args.command == "evaluate":
        if not args.export:
            raise ValueError("evaluate requires --export inference_best.pt or inference_latest.pt")
        model, meta = load_export(args.export)
        if meta["method"] != method or meta["student_model"] != exp.student_model:
            raise ValueError("Export does not match selected method/student")
        exp.data.mean, exp.data.std = tuple(meta["mean"]), tuple(meta["std"])
        ds = cifar_dataset(exp, train=False, download=args.download, limit=args.limit)
        result = evaluate(model.to(device), DataLoader(ds, batch_size=exp.data.batch_size,
                          num_workers=exp.data.num_workers), device)
        print(json.dumps(result), flush=True)
        return
    teacher_path = args.teacher_checkpoint or exp.teacher_checkpoint
    if not teacher_path or not Path(teacher_path).is_file():
        raise FileNotFoundError("Provide the existing --teacher-checkpoint file; model registry paths are not assumed")
    teacher = load_teacher(build_model(exp.teacher_model, exp.num_classes), teacher_path)
    ds = cifar_dataset(exp, download=args.download, limit=args.limit)
    identity = manifest(exp, teacher_path, ds)
    bank = CropBank(identity, args.cache_root, args.cache_write_root or (args.cache_root if args.command == "prepare" else None))
    print(json.dumps({"method": method, "teacher": exp.teacher_model, "student": exp.student_model,
                      "cache_key": bank.key, "profile_source": raw.get("profile_source"),
                      "config": asdict(exp), "method_options": options}), flush=True)
    if args.command == "prepare":
        prepare_bank(exp, teacher.to(device), ds, bank)
        return
    distiller = build_distiller(exp, method, options, teacher).to(device)
    # Eval shape pass first, then method loss/backward with a real-size two-sample batch.
    shape_report = []
    for size in sorted({r for r, _ in exp.data.stages} | {32}):
        distiller.train()
        image = torch.randn(2, 3, size, size, device=device)
        target = torch.tensor([0, 1], device=device)
        metadata = {k:torch.ones(2, device=device) for k in ("q", "r1", "r2", "r3")}
        _, losses = distiller(image=image, target=target, epoch=1, tgsc=metadata,
                             gate_mode=options.get("gate_mode", "unified"), gate_threshold=exp.tgsc.score_high)
        sum(losses.values()).backward()
        distiller.zero_grad(set_to_none=True)
        shape_report.append({"resolution": size, "losses": {k:float(v.detach()) for k,v in losses.items()}})
    print(json.dumps({"shape_checks": shape_report}), flush=True)
    # Construct a fresh model below: preflight must not alter initial BN/RNG state for training.
    for size in sorted({r for r, _ in exp.data.stages}):
        loaded = bank.load(size)
        if not torch.equal(loaded.labels, torch.tensor(ds.targets)[loaded.source_indices]):
            raise ValueError("Cached labels do not align with original CIFAR sources")
        print(f"Validated {bank.key}/res{size}: {len(loaded)} crops", flush=True)
        del loaded
    if args.command == "check":
        print("Preflight passed; no training performed.", flush=True)
        return
    seed_everything(exp.train.seed)
    del distiller
    distiller = build_distiller(exp, method, options, teacher.cpu())
    test_ds = cifar_dataset(exp, train=False, download=args.download, limit=args.limit)
    run_suffix = f"seed_{exp.train.seed}" + ("_" + options.get("gate_mode", "unified") if method == "DKD" else "")
    output = args.output_dir or str(Path(exp.train.save_path).parent / run_suffix)
    train(exp, method, options, distiller, bank, test_ds, device, output,
          resume=args.resume, stop_after_epoch=args.stop_after_epoch, launch_args=vars(args))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, FloatingPointError) as error:
        print(f"TGSC error: {error}", file=sys.stderr)
        sys.exit(1)
