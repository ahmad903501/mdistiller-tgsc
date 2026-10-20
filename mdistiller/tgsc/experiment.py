"""JSON run configuration: existing TGSC profiles plus official method settings."""
import json
from pathlib import Path
from types import SimpleNamespace

from .config import experiment_config_from_dict


def load_run(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    method = raw.get("method", "").upper()
    if method not in ("DKD", "SIMKD", "REVIEWKD"):
        raise ValueError("method must be DKD, SIMKD, or REVIEWKD")
    exp = experiment_config_from_dict(raw)
    if sum(d for _, d in exp.data.stages) != exp.train.epochs:
        raise ValueError("Curriculum durations must sum to train.epochs")
    if any(r < 16 or r > 32 or n <= 0 for r, n in exp.data.stages):
        raise ValueError("Supported CIFAR stages are sizes 16..32 with positive durations")
    if exp.data.batch_size < 2:
        raise ValueError("Feature adapters require a training batch size of at least 2")
    if exp.tgsc.score_high <= 0:
        raise ValueError("TGSC gate threshold score_high must be positive")
    if not 0 <= exp.tgsc.flip_prob <= 1 or exp.tgsc.aug_padding < 0:
        raise ValueError("Invalid TGSC augmentation settings")
    if exp.optim.lr_half_on_stage_change:
        raise ValueError("Run profiles must explicitly disable stage LR halving; audited profiles use milestones")
    options = raw.get("method_options", {})
    if options.get("gate_mode", "unified") not in ("unified", "decoupled"):
        raise ValueError("Unknown gate mode")
    return exp, method, options, raw


def method_config(exp, method, options, student, teacher):
    """Discover channel dimensions in eval mode without changing BN buffers."""
    import torch
    cfg = SimpleNamespace()
    cfg.DKD = SimpleNamespace(CE_WEIGHT=1., ALPHA=options.get("alpha", 1.),
                              BETA=options.get("beta", 8.), T=options.get("temperature", 4.),
                              WARMUP=options.get("warmup", 20))
    cfg.SIMKD = SimpleNamespace(FACTOR=options.get("factor", 2), FEAT_WEIGHT=options.get("feat_weight", 1.),
                                RESOLUTIONS=sorted({32, *(r for r, _ in exp.data.stages)}))
    if method == "REVIEWKD":
        if "review_weight" not in options:
            raise ValueError("ReviewKD requires an explicitly audited pair review_weight")
        state_s, state_t = student.training, teacher.training
        student.eval(); teacher.eval()
        with torch.no_grad():
            sf = student(torch.zeros(2, 3, 32, 32))[1]
            tf = teacher(torch.zeros(2, 3, 32, 32))[1]
        student.train(state_s); teacher.train(state_t)
        if "preact_feats" not in tf:
            raise ValueError("ReviewKD teacher needs preactivation features")
        s = sf["feats"][1:] + [sf["pooled_feat"][:, :, None, None]]
        t = tf["preact_feats"][1:] + [tf["pooled_feat"][:, :, None, None]]
        if len(s) != len(t):
            raise ValueError("No audited ReviewKD stage correspondence for this pair")
        cfg.REVIEWKD = SimpleNamespace(
            IN_CHANNELS=[f.shape[1] for f in s], OUT_CHANNELS=[f.shape[1] for f in t],
            SHAPES=[f.shape[-1] for f in s[::-1]], OUT_SHAPES=[f.shape[-1] for f in t[::-1]],
            CE_WEIGHT=1., REVIEWKD_WEIGHT=options["review_weight"], WARMUP_EPOCHS=options.get("warmup",20),
            STU_PREACT=False, MAX_MID_CHANNEL=512, TGSC_DYNAMIC=True)
    return cfg
