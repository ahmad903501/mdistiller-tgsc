from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Optional


@dataclass
class DataConfig:
    root: str = "./data"
    mean: tuple[float, float, float] = (0.5071, 0.4867, 0.4408)
    std: tuple[float, float, float] = (0.2675, 0.2565, 0.2761)
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    stages: tuple[tuple[int, int], ...] = (
        (16, 48),
        (20, 48),
        (24, 48),
        (28, 48),
        (32, 48),
    )

    def __post_init__(self) -> None:
        self.mean = tuple(self.mean)
        self.std = tuple(self.std)
        self.stages = tuple((int(res), int(epochs)) for res, epochs in self.stages)


@dataclass
class KDConfig:
    T: float = 4.0
    alpha: float = 0.9


@dataclass
class OptimConfig:
    base_lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_half_on_stage_change: bool = True
    milestones: tuple[int, int, int] = (150, 180, 210)
    milestone_gamma: float = 0.1

    def __post_init__(self) -> None:
        self.milestones = tuple(int(value) for value in self.milestones)


@dataclass
class TrainConfig:
    epochs: int = 240
    seed: int = 27
    save_path: str = "outputs/student_kd.pth"
    log_every: int = 0
    log_dir: str = "logs"
    run_name: Optional[str] = None


@dataclass
class TGSCConfig:
    precompute_batch_size: int = 256
    precompute_workers: int = 2
    use_gt_target: bool = True
    w_cam: float = 0.7
    w_sal: float = 0.3
    temp: float = 1.5
    gamma: float = 1.2
    pct_lo: float = 1.0
    pct_hi: float = 99.0
    stride: int = 8
    k_per_img: int = 6
    min_frac: float = 0.20
    min_mean: float = 0.20
    eval_chunk: int = 256
    cache: bool = True
    cache_to_disk: bool = False
    cache_dir: str = ".cache"
    flip_prob: float = 0.5
    aug_padding: int = 2
    top_k_crops: int = 2
    low_k_crops: int = 1
    score_high: float = 0.7
    score_low: float = 0.3
    easy_threshold: float = 0.75
    hard_threshold: float = 0.25
    difficulty_step: float = 0.05


@dataclass
class ExperimentConfig:
    teacher_model: str = "vgg13"
    teacher_checkpoint: Optional[str] = None
    student_model: str = "vgg8"
    student_checkpoint: Optional[str] = None
    num_classes: int = 100
    device: Optional[str] = None
    data: DataConfig = field(default_factory=DataConfig)
    kd: KDConfig = field(default_factory=KDConfig)
    optim: OptimConfig = field(
        default_factory=lambda: OptimConfig(
            lr_half_on_stage_change=False,
            milestones=(48, 144, 192),
            milestone_gamma=0.1,
        )
    )
    train: TrainConfig = field(
        default_factory=lambda: TrainConfig(
            epochs=240,
            seed=27,
            save_path="outputs/vgg13_to_vgg8_tgsc_kd.pth",
            log_every=0,
        )
    )
    tgsc: TGSCConfig = field(
        default_factory=lambda: TGSCConfig(
            score_high=0.75,
            score_low=0.35,
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_section(defaults: dict[str, Any], updates: Optional[dict[str, Any]]) -> dict[str, Any]:
    merged = dict(defaults)
    if updates:
        merged.update(updates)
    return merged


def experiment_config_from_dict(raw: dict[str, Any], base: Optional[ExperimentConfig] = None) -> ExperimentConfig:
    base_cfg = base or ExperimentConfig()
    base_dict = base_cfg.to_dict()

    merged: dict[str, Any] = {}
    for key in ("teacher_model", "teacher_checkpoint", "student_model", "student_checkpoint", "num_classes", "device"):
        merged[key] = raw.get(key, base_dict[key])

    merged["data"] = DataConfig(**_merge_section(base_dict["data"], raw.get("data")))
    merged["kd"] = KDConfig(**_merge_section(base_dict["kd"], raw.get("kd")))
    merged["optim"] = OptimConfig(**_merge_section(base_dict["optim"], raw.get("optim")))
    merged["train"] = TrainConfig(**_merge_section(base_dict["train"], raw.get("train")))
    merged["tgsc"] = TGSCConfig(**_merge_section(base_dict["tgsc"], raw.get("tgsc")))
    return ExperimentConfig(**merged)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    raw = json.loads(config_path.read_text())
    return experiment_config_from_dict(raw)
