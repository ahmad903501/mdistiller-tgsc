"""Download the official mdistiller CIFAR teacher archive and verify a selected teacher."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import urllib.request

URL = "https://github.com/megvii-research/mdistiller/releases/download/checkpoints/cifar_teachers.tar"


def select_member(archive, teacher):
    """Match real archive entries against the registry's directory/filename convention."""
    expected = (f"{teacher}_vanilla", "ckpt_epoch_240.pth")
    matches = [m for m in archive.getmembers()
               if m.isfile() and PurePosixPath(m.name).parts[-2:] == expected]
    if len(matches) != 1:
        available = [m.name for m in archive.getmembers() if m.isfile() and m.name.endswith(".pth")]
        raise ValueError(f"Expected exactly one {expected}, found {len(matches)}. Actual checkpoints: {available}")
    return matches[0]


def download(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"Using existing archive: {path}", flush=True)
        return
    partial = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(URL, headers={"User-Agent": "TGSC-checkpoint-downloader"})
    size, reported = 0, 0
    with urllib.request.urlopen(request, timeout=60) as source, partial.open("wb") as output:
        expected = source.headers.get("Content-Length")
        while block := source.read(1024 * 1024):
            output.write(block)
            size += len(block)
            if size - reported >= 32 * 1024**2:
                print(f"Downloaded {size / 1024**2:.0f} MiB", flush=True)
                reported = size
    if expected is not None and size != int(expected):
        raise OSError(f"Incomplete download: {size}/{expected} bytes; incomplete file remains at {partial}")
    os.replace(partial, path)


def verify_checkpoint(teacher, checkpoint):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from mdistiller.tgsc.runtime import build_model, load_teacher
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = load_teacher(build_model(teacher), checkpoint)
    with torch.no_grad():
        logits = model(torch.zeros(2, 3, 32, 32))[0]
    if logits.shape != (2, 100) or not torch.isfinite(logits).all():
        raise ValueError("Teacher checkpoint did not produce finite CIFAR-100 logits")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="TGSC JSON config; select teacher_model automatically")
    source.add_argument("--teacher", help="Exact mdistiller teacher registry key")
    p.add_argument("--output-dir", default="download_ckpts", help="Writable checkpoint/archive directory")
    p.add_argument("--archive", help="Use an existing archive without downloading")
    args = p.parse_args()
    teacher = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))["teacher_model"] if args.config else args.teacher
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mdistiller.models import cifar_model_dict
    if teacher not in cifar_model_dict or cifar_model_dict[teacher][1] is None:
        raise ValueError(f"No official CIFAR teacher path registered for {teacher!r}")
    # Use the actual registry suffix, then independently verify it exists in the archive.
    registered = Path(cifar_model_dict[teacher][1])
    destination = Path(args.output_dir).resolve() / "cifar_teachers" / registered.parent.name / registered.name
    archive_path = Path(args.archive) if args.archive else Path(args.output_dir) / "cifar_teachers.tar"
    if not args.archive:
        download(archive_path)
    with tarfile.open(archive_path, "r:*") as archive:
        member = select_member(archive, teacher)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_suffix(destination.suffix + ".part")
            # Copy one regular member to a controlled path, never extract archive paths verbatim.
            with archive.extractfile(member) as source, partial.open("wb") as output:
                shutil.copyfileobj(source, output)
            if partial.stat().st_size != member.size:
                raise OSError("Extracted checkpoint length differs from archive member")
            os.replace(partial, destination)
        with destination.open("rb") as checkpoint_file:
            digest = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
        with archive.extractfile(member) as official_file:
            official_digest = hashlib.file_digest(official_file, "sha256").hexdigest()
        if digest != official_digest:
            raise ValueError(f"Existing checkpoint differs from the official archive: {destination}; use another output directory")
        verify_checkpoint(teacher, destination)
        report = {"teacher": teacher, "source": URL, "archive_member": member.name,
                          "checkpoint": str(destination), "sha256": digest,
                          "relative_checkpoint": destination.relative_to(Path(args.output_dir).resolve()).as_posix(),
                          "verification": "strict state_dict load and finite 100-class forward passed"}
        # The relative path and hash remain useful after uploading this folder to Kaggle.
        manifest = Path(args.output_dir) / "teacher_manifest.json"
        manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
