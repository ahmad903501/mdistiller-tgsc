import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ..tgsc.gates import weighted_mean


def dkd_components(logits_student, logits_teacher, target, temperature):
    """Official KL components before batch reduction, for sample-wise gating."""
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = ~gt_mask
    ps = cat_mask(F.softmax(logits_student / temperature, dim=1), gt_mask, other_mask)
    pt = cat_mask(F.softmax(logits_teacher / temperature, dim=1), gt_mask, other_mask)
    tckd = F.kl_div(ps.log(), pt, reduction="none").sum(1) * temperature**2
    nckd = F.kl_div(
        F.log_softmax(logits_student / temperature - 1000.0 * gt_mask, dim=1),
        F.softmax(logits_teacher / temperature - 1000.0 * gt_mask, dim=1),
        reduction="none",
    ).sum(1) * temperature**2
    return tckd, nckd


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class DKD(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(DKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        metadata = kwargs.get("tgsc")
        if metadata is not None:
            tckd, nckd = dkd_components(logits_student, logits_teacher, target, self.temperature)
            self.tgsc_diagnostics = {"raw_tckd": tckd.detach().mean(), "raw_nckd": nckd.detach().mean()}
            threshold = kwargs.get("gate_threshold", 0.75)
            mode = kwargs.get("gate_mode", "unified")
            if mode == "decoupled":
                gates = dict(metadata, target_quality=(metadata["r1"] + metadata["r3"]) / 2)
                loss_dkd = self.alpha * weighted_mean(tckd, gates, threshold, "target_quality")
                loss_dkd += self.beta * weighted_mean(nckd, gates, threshold, "r2")
            elif mode == "unified":
                loss_dkd = weighted_mean(self.alpha * tckd + self.beta * nckd, metadata, threshold)
            else:
                raise ValueError("DKD gate mode must be unified or decoupled")
            loss_dkd *= min(kwargs["epoch"] / self.warmup, 1.0) if self.warmup else 1.0
            return logits_student, {"loss_ce": loss_ce, "loss_kd": loss_dkd}
        loss_dkd = (min(kwargs["epoch"] / self.warmup, 1.0) if self.warmup else 1.0) * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student, losses_dict
