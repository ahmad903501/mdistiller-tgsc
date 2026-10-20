from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from .config import DataConfig, TGSCConfig
from .utils import forward_logits, pick_target_layer


@dataclass
class TeacherUnionCache:
    union_maps: torch.Tensor
    logits: torch.Tensor

    def to(self, device: torch.device | str) -> "TeacherUnionCache":
        return TeacherUnionCache(
            union_maps=self.union_maps.to(device),
            logits=self.logits.to(device),
        )


def _top2_batch(p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    vals, idx = torch.sort(p, dim=-1, descending=True)
    return idx[:, 0], vals[:, 0], idx[:, 1], vals[:, 1]


def _kl_div_batch(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    p_ = p.clamp_min(eps)
    q_ = q.clamp_min(eps)
    return (p_ * (p_.log() - q_.log())).sum(dim=1)


class _LayerTaps:
    def __init__(self, target_layer: nn.Module):
        self.A: Optional[torch.Tensor] = None
        self.dY: Optional[torch.Tensor] = None
        self._h1 = target_layer.register_forward_hook(self._store_act)
        self._h2 = target_layer.register_full_backward_hook(self._store_grad)

    def _store_act(self, _module: nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        self.A = out

    def _store_grad(
        self,
        _module: nn.Module,
        _gin: tuple[Optional[torch.Tensor], ...],
        gout: tuple[Optional[torch.Tensor], ...],
    ) -> None:
        self.dY = gout[0]

    def close(self) -> None:
        self._h1.remove()
        self._h2.remove()


def _gradcam_pp_from_taps(A: torch.Tensor, dY: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    dY2 = dY * dY
    dY3 = dY2 * dY
    sumA = A.sum(dim=(2, 3), keepdim=True)
    sumdY3 = dY3.sum(dim=(2, 3), keepdim=True)
    alpha = dY2 / (2.0 * dY2 + sumA * sumdY3 + eps)
    w = (alpha * dY.clamp_min(0)).sum(dim=(2, 3), keepdim=True)
    cam = (w * A).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    return cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)


def _soft_or_t(cam: torch.Tensor, sal: torch.Tensor, *, w_cam: float, w_sal: float, temp: float) -> torch.Tensor:
    cam = cam.clamp(0, 1)
    sal = sal.clamp(0, 1)
    if temp != 1.0:
        cam = cam.clamp(1e-8, 1 - 1e-8).pow(1.0 / temp)
        sal = sal.clamp(1e-8, 1 - 1e-8).pow(1.0 / temp)
    return 1.0 - (1.0 - cam).pow(w_cam) * (1.0 - sal).pow(w_sal)


def _percentile_scale_t(x: torch.Tensor, *, lo: float, hi: float) -> torch.Tensor:
    batch = x.size(0)
    flat = x.view(batch, -1)
    q = torch.tensor([lo / 100.0, hi / 100.0], device=x.device, dtype=x.dtype)
    values = torch.quantile(flat, q, dim=1)
    lo_v = values[0].view(batch, 1, 1, 1)
    hi_v = values[1].view(batch, 1, 1, 1)
    denom = (hi_v - lo_v).clamp_min(1e-8)
    return ((x - lo_v).clamp_min(0.0) / denom).clamp(0.0, 1.0)


def _build_windows_by_pool(
    union: torch.Tensor,
    crop_h: int,
    crop_w: int,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    thr = _percentile_scale_t(union, lo=75, hi=75)
    mask = (union >= thr).float()
    mean_union_map = F.avg_pool2d(union, kernel_size=(crop_h, crop_w), stride=stride, padding=0)
    frac_above_map = F.avg_pool2d(mask, kernel_size=(crop_h, crop_w), stride=stride, padding=0)
    return mean_union_map, frac_above_map


def _gather_top_windows(
    mean_union_map: torch.Tensor,
    frac_map: torch.Tensor,
    *,
    k_per_img: int,
    min_frac: float,
    min_mean: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, _, grid_h, grid_w = mean_union_map.shape
    flat_score = mean_union_map.view(batch, -1)
    flat_frac = frac_map.view(batch, -1)
    valid = (flat_frac >= min_frac) & (flat_score >= min_mean)
    score_masked = torch.where(valid, flat_score, torch.full_like(flat_score, -1.0))
    topk = min(k_per_img, grid_h * grid_w)
    vals, idxs = torch.topk(score_masked, k=topk, dim=1)
    yh = idxs // grid_w
    xw = idxs % grid_w
    return yh, xw, vals


def _boxes_from_grid(
    yh: torch.Tensor,
    xw: torch.Tensor,
    crop_h: int,
    crop_w: int,
    stride: int,
    height: int,
    width: int,
) -> torch.Tensor:
    y0 = yh * stride
    x0 = xw * stride
    y1 = (y0 + crop_h).clamp(max=height)
    x1 = (x0 + crop_w).clamp(max=width)
    return torch.stack([y0, x0, y1, x1], dim=-1)


def _score_candidates_with_teacher(
    model: nn.Module,
    full_logits: torch.Tensor,
    class_idx: torch.Tensor,
    x: torch.Tensor,
    boxes: torch.Tensor,
    *,
    eval_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = x.device
    batch, num_candidates, _ = boxes.shape
    if num_candidates == 0:
        return (
            boxes[:, :0],
            torch.empty((batch, 0), device=device),
            torch.empty((batch, 0, full_logits.size(1)), device=device),
            torch.empty((batch, 0, 3), device=device),
        )

    p_full = torch.softmax(full_logits, dim=-1)
    _, pc1_f, _, pc2_f = _top2_batch(p_full)

    crops: list[torch.Tensor] = []
    owners: list[int] = []
    candidate_indices: list[int] = []
    for batch_idx in range(batch):
        for candidate_idx in range(num_candidates):
            y0, x0, y1, x1 = boxes[batch_idx, candidate_idx].tolist()
            crops.append(x[batch_idx : batch_idx + 1, :, y0:y1, x0:x1])
            owners.append(batch_idx)
            candidate_indices.append(candidate_idx)

    crops_cat = torch.cat(crops, dim=0)
    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, crops_cat.size(0), eval_chunk):
            logits_k = forward_logits(model, crops_cat[start : start + eval_chunk])
            preds.append(torch.softmax(logits_k, dim=-1))
    p_k_all = torch.cat(preds, dim=0)

    scores = torch.empty(p_k_all.size(0), device=device)
    conf_ratio = torch.empty_like(scores)
    kl_norm = torch.empty_like(scores)
    margin_rat = torch.empty_like(scores)

    alpha = math.log(p_k_all.size(1))
    owners_t = torch.tensor(owners, device=device, dtype=torch.long)
    candidate_t = torch.tensor(candidate_indices, device=device, dtype=torch.long)

    for flat_idx in range(p_k_all.size(0)):
        owner = int(owners_t[flat_idx].item())
        pk = p_k_all[flat_idx]
        pf = p_full[owner]
        cls = int(class_idx[owner].item())

        pc_k = pk[cls]
        pc_f = pf[cls]
        conf_ratio[flat_idx] = pc_k / (pc_f + 1e-8)

        kl = _kl_div_batch(pf.unsqueeze(0), pk.unsqueeze(0))[0]
        kl_norm[flat_idx] = 1.0 - torch.clamp(kl / alpha, max=1.0)

        vals_k, _ = torch.sort(pk, descending=True)
        pc1_k, pc2_k = vals_k[0], vals_k[1]
        margin_rat[flat_idx] = (pc1_k - pc2_k) / (pc1_f[owner] - pc2_f[owner] + 1e-8)

    scores[:] = (conf_ratio + kl_norm + margin_rat) / 3.0

    component_matrix = torch.zeros((batch, num_candidates, 3), device=device)
    score_matrix = torch.full((batch, num_candidates), -1.0, device=device)
    prob_matrix = torch.zeros((batch, num_candidates, p_k_all.size(1)), dtype=p_k_all.dtype, device=device)
    for flat_idx in range(scores.numel()):
        owner = int(owners_t[flat_idx].item())
        candidate_idx = int(candidate_t[flat_idx].item())
        component_matrix[owner, candidate_idx] = torch.stack((conf_ratio[flat_idx], kl_norm[flat_idx], margin_rat[flat_idx]))
        score_matrix[owner, candidate_idx] = scores[flat_idx]
        prob_matrix[owner, candidate_idx] = p_k_all[flat_idx]

    sorted_scores, order = torch.sort(score_matrix, dim=1, descending=True)
    order_boxes = order.unsqueeze(-1).expand(-1, -1, 4)
    boxes_sorted = boxes.gather(1, order_boxes)
    order_probs = order.unsqueeze(-1).expand(-1, -1, prob_matrix.size(-1))
    probs_sorted = prob_matrix.gather(1, order_probs)
    components_sorted = component_matrix.gather(1, order.unsqueeze(-1).expand(-1, -1, 3))
    return boxes_sorted, sorted_scores, probs_sorted, components_sorted


@torch.inference_mode(False)
def build_teacher_union_cache(
    base_loader: DataLoader,
    teacher: nn.Module,
    target_layer: nn.Module,
    *,
    use_gt_target: bool,
    w_cam: float,
    w_sal: float,
    temp: float,
    gamma: float,
    pct_lo: float,
    pct_hi: float,
) -> TeacherUnionCache:
    device = next(teacher.parameters()).device
    teacher.eval()

    union_maps_all: list[torch.Tensor] = []
    logits_all: list[torch.Tensor] = []

    print("[TGSC] Building teacher union cache (one-time)...")

    for xb, yb in tqdm(base_loader, desc="Teacher cache", leave=False):
        xb = xb.to(device).requires_grad_(True)
        yb = yb.to(device)
        _, _, height, width = xb.shape

        taps = _LayerTaps(target_layer)
        teacher.zero_grad(set_to_none=True)

        logits = forward_logits(teacher, xb)
        probs_full = torch.softmax(logits.detach(), dim=-1)
        top1 = probs_full.argmax(dim=1)
        target_idx = yb if use_gt_target else top1

        oh = torch.zeros_like(logits)
        oh.scatter_(1, target_idx.view(-1, 1), 1.0)
        logits.backward(gradient=oh, retain_graph=False)

        g = xb.grad.detach().abs().amax(dim=1, keepdim=True)
        sal = (g - g.amin(dim=(2, 3), keepdim=True)) / (
            g.amax(dim=(2, 3), keepdim=True) - g.amin(dim=(2, 3), keepdim=True) + 1e-8
        )

        A, dY = taps.A, taps.dY
        if A is None or dY is None:
            taps.close()
            raise RuntimeError("Target layer hooks did not capture activations or gradients.")

        cam_small = _gradcam_pp_from_taps(A, dY)
        cam = F.interpolate(cam_small, size=(height, width), mode="bilinear", align_corners=False)
        cam = _percentile_scale_t(cam, lo=pct_lo, hi=pct_hi)
        sal = _percentile_scale_t(sal, lo=pct_lo, hi=pct_hi)

        union = _soft_or_t(cam, sal, w_cam=w_cam, w_sal=w_sal, temp=temp)
        if gamma != 1.0:
            union = union.clamp(0, 1).pow(1.0 / max(gamma, 1e-6))

        union_maps_all.append(union.detach().cpu())
        logits_all.append(logits.detach().cpu())

        taps.close()
        xb.grad = None
        teacher.zero_grad(set_to_none=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cache = TeacherUnionCache(
        union_maps=torch.cat(union_maps_all, dim=0),
        logits=torch.cat(logits_all, dim=0),
    )

    print(
        f"[TGSC] Teacher cache built: {cache.union_maps.shape[0]} images, "
        f"~{(cache.union_maps.numel() + cache.logits.numel()) * 4 / 1024**2:.1f} MB"
    )
    return cache


@torch.inference_mode(False)
def build_bestcrop_dataset_batchwise(
    base_loader: DataLoader,
    teacher: nn.Module,
    target_layer: nn.Module,
    *,
    crop_size: int | tuple[int, int],
    use_gt_target: bool,
    w_cam: float,
    w_sal: float,
    temp: float,
    gamma: float,
    pct_lo: float,
    pct_hi: float,
    stride: int,
    k_per_img: int,
    min_frac: float,
    min_mean: float,
    eval_chunk: int,
    top_k_crops: int,
    low_k_crops: int,
    teacher_cache: Optional[TeacherUnionCache] = None,
) -> "CropTensorDataset":
    device = next(teacher.parameters()).device
    teacher.eval()

    if isinstance(crop_size, int):
        crop_h = crop_w = int(crop_size)
    else:
        crop_h, crop_w = int(crop_size[0]), int(crop_size[1])

    crops_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    scores_all: list[torch.Tensor] = []
    sources_all: list[torch.Tensor] = []
    quality_all: list[torch.Tensor] = []
    components_all: list[torch.Tensor] = []
    boxes_all: list[torch.Tensor] = []

    progress_desc = f"TGSC precompute {crop_h}x{crop_w}"

    if teacher_cache is not None:
        print(f"[TGSC] Using cached teacher outputs for {crop_h}x{crop_w} crops")
        seen = 0
        for xb, yb in tqdm(base_loader, desc=progress_desc, leave=False):
            xb = xb.to(device)
            yb = yb.to(device)

            batch = xb.size(0)
            start_idx = seen
            end_idx = start_idx + batch
            seen = end_idx

            union = teacher_cache.union_maps[start_idx:end_idx].to(device)
            logits = teacher_cache.logits[start_idx:end_idx].to(device)

            _, _, height, width = xb.shape
            base_indices = torch.arange(start_idx, end_idx, dtype=torch.long)

            probs_full = torch.softmax(logits, dim=-1)
            top1 = probs_full.argmax(dim=1)
            target_idx = yb if use_gt_target else top1

            mean_union_map, frac_map = _build_windows_by_pool(union, crop_h, crop_w, stride)
            yh, xw, _ = _gather_top_windows(
                mean_union_map,
                frac_map,
                k_per_img=k_per_img,
                min_frac=min_frac,
                min_mean=min_mean,
            )
            boxes = _boxes_from_grid(yh, xw, crop_h, crop_w, stride, height, width)
            boxes_sorted, scores_sorted, _, components_sorted = _score_candidates_with_teacher(
                teacher,
                logits,
                target_idx,
                xb,
                boxes,
                eval_chunk=eval_chunk,
            )

            for sample_idx in range(batch):
                _collect_crop_candidates(
                    xb=xb,
                    yb=yb,
                    sample_idx=sample_idx,
                    boxes_sorted=boxes_sorted,
                    scores_sorted=scores_sorted,
                    crop_h=crop_h,
                    crop_w=crop_w,
                    base_indices=base_indices,
                    top_k_crops=top_k_crops,
                    low_k_crops=low_k_crops,
                    crops_all=crops_all,
                    labels_all=labels_all,
                    scores_all=scores_all,
                    sources_all=sources_all,
                    quality_all=quality_all,
                    components_sorted=components_sorted, components_all=components_all, boxes_all=boxes_all,
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        print(f"[TGSC] Computing from scratch (no cache) for {crop_h}x{crop_w} crops")
        seen = 0
        for xb, yb in tqdm(base_loader, desc=progress_desc, leave=False):
            xb = xb.to(device).requires_grad_(True)
            yb = yb.to(device)

            batch, _, height, width = xb.shape
            base_indices = torch.arange(seen, seen + batch, dtype=torch.long)
            seen += batch

            taps = _LayerTaps(target_layer)
            teacher.zero_grad(set_to_none=True)

            logits = forward_logits(teacher, xb)
            probs_full = torch.softmax(logits.detach(), dim=-1)
            top1 = probs_full.argmax(dim=1)
            target_idx = yb if use_gt_target else top1

            oh = torch.zeros_like(logits)
            oh.scatter_(1, target_idx.view(-1, 1), 1.0)
            logits.backward(gradient=oh, retain_graph=False)

            g = xb.grad.detach().abs().amax(dim=1, keepdim=True)
            sal = (g - g.amin(dim=(2, 3), keepdim=True)) / (
                g.amax(dim=(2, 3), keepdim=True) - g.amin(dim=(2, 3), keepdim=True) + 1e-8
            )

            A, dY = taps.A, taps.dY
            if A is None or dY is None:
                taps.close()
                raise RuntimeError("Target layer hooks did not capture activations or gradients.")

            cam_small = _gradcam_pp_from_taps(A, dY)
            cam = F.interpolate(cam_small, size=(height, width), mode="bilinear", align_corners=False)
            cam = _percentile_scale_t(cam, lo=pct_lo, hi=pct_hi)
            sal = _percentile_scale_t(sal, lo=pct_lo, hi=pct_hi)
            union = _soft_or_t(cam, sal, w_cam=w_cam, w_sal=w_sal, temp=temp)
            if gamma != 1.0:
                union = union.clamp(0, 1).pow(1.0 / max(gamma, 1e-6))

            taps.close()
            xb.grad = None

            mean_union_map, frac_map = _build_windows_by_pool(union, crop_h, crop_w, stride)
            yh, xw, _ = _gather_top_windows(
                mean_union_map,
                frac_map,
                k_per_img=k_per_img,
                min_frac=min_frac,
                min_mean=min_mean,
            )
            boxes = _boxes_from_grid(yh, xw, crop_h, crop_w, stride, height, width)
            boxes_sorted, scores_sorted, _, components_sorted = _score_candidates_with_teacher(
                teacher,
                logits.detach(),
                target_idx,
                xb,
                boxes,
                eval_chunk=eval_chunk,
            )

            for sample_idx in range(batch):
                _collect_crop_candidates(
                    xb=xb,
                    yb=yb,
                    sample_idx=sample_idx,
                    boxes_sorted=boxes_sorted,
                    scores_sorted=scores_sorted,
                    crop_h=crop_h,
                    crop_w=crop_w,
                    base_indices=base_indices,
                    top_k_crops=top_k_crops,
                    low_k_crops=low_k_crops,
                    crops_all=crops_all,
                    labels_all=labels_all,
                    scores_all=scores_all,
                    sources_all=sources_all,
                    quality_all=quality_all,
                    components_sorted=components_sorted, components_all=components_all, boxes_all=boxes_all,
                )

            teacher.zero_grad(set_to_none=True)
            xb = xb.detach()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not crops_all:
        raise RuntimeError("No TGSC crops were generated; please verify teacher checkpoints and settings.")

    X = torch.cat(crops_all, dim=0)
    Y = torch.cat(labels_all, dim=0).long()
    S = torch.cat(scores_all, dim=0).float().clamp(0.0, 1.0)
    src = torch.cat(sources_all, dim=0).long()
    qflag = torch.cat(quality_all, dim=0).long()
    print(f"[TGSC] Built best-crop tensor {tuple(X.shape)} (C,H,W) with {S.numel()} crops.")
    return CropTensorDataset(X, Y, S, src, qflag, torch.cat(components_all), torch.cat(boxes_all))


def _collect_crop_candidates(
    *,
    xb: torch.Tensor,
    yb: torch.Tensor,
    sample_idx: int,
    boxes_sorted: torch.Tensor,
    scores_sorted: torch.Tensor,
    crop_h: int,
    crop_w: int,
    base_indices: torch.Tensor,
    top_k_crops: int,
    low_k_crops: int,
    crops_all: list[torch.Tensor],
    labels_all: list[torch.Tensor],
    scores_all: list[torch.Tensor],
    sources_all: list[torch.Tensor],
    quality_all: list[torch.Tensor],
    components_sorted: torch.Tensor, components_all: list, boxes_all: list,
) -> None:
    sample_scores = scores_sorted[sample_idx]
    valid_mask = sample_scores >= 0
    valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        return

    valid = valid_idx.tolist()
    top_sel = valid[: max(0, min(top_k_crops, len(valid)))]
    low_sel_candidates = valid[-max(0, min(low_k_crops, len(valid))) :] if low_k_crops > 0 else []
    used = set(top_sel)
    low_sel: list[int] = []
    for candidate_idx in reversed(low_sel_candidates):
        if candidate_idx not in used:
            low_sel.append(candidate_idx)
            used.add(candidate_idx)

    selections = [(candidate_idx, rank, 1) for rank, candidate_idx in enumerate(top_sel)]
    selections.extend((candidate_idx, rank, -1) for rank, candidate_idx in enumerate(low_sel))

    for candidate_idx, rank, quality_flag in selections:
        score_val = sample_scores[candidate_idx]
        if score_val < 0:
            continue
        y0, x0, y1, x1 = boxes_sorted[sample_idx, candidate_idx].long().tolist()
        crop = xb[sample_idx, :, y0:y1, x0:x1].detach()
        height, width = crop.shape[-2:]
        if height != crop_h or width != crop_w:
            pad_t = max(0, crop_h - height)
            pad_l = max(0, crop_w - width)
            crop = F.pad(crop, (0, pad_l, 0, pad_t), mode="reflect")[:, :crop_h, :crop_w]

        components_all.append(components_sorted[sample_idx, candidate_idx].detach().cpu().unsqueeze(0))
        boxes_all.append(boxes_sorted[sample_idx, candidate_idx].detach().cpu().long().unsqueeze(0))
        crops_all.append(crop.cpu().unsqueeze(0))
        labels_all.append(yb[sample_idx].detach().cpu().unsqueeze(0))
        scores_all.append(score_val.detach().cpu().unsqueeze(0))
        sources_all.append(torch.tensor([int(base_indices[sample_idx].item())], dtype=torch.long))
        quality_all.append(torch.tensor([quality_flag * (rank + 1)], dtype=torch.long))


class CropTensorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        crops: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        source_indices: torch.Tensor,
        quality_flags: torch.Tensor,
        components: torch.Tensor, boxes: torch.Tensor,
    ):
        self.crops = crops
        self.labels = labels
        self.scores = scores
        self.source_indices = source_indices
        self.quality_flags = quality_flags
        self.components = components
        self.boxes = boxes
        self.transform = None

    def __len__(self) -> int:
        return self.crops.size(0)

    def __getitem__(self, idx: int):
        x = self.crops[idx]
        if self.transform is not None:
            x = self.transform(x.clone())
        return {"image": x, "target": self.labels[idx], "q": self.scores[idx],
                "r1": self.components[idx, 0], "r2": self.components[idx, 1],
                "r3": self.components[idx, 2], "source_index": self.source_indices[idx],
                "crop_id": idx, "box": self.boxes[idx], "resolution": self.crops.shape[-1],
                "quality_flag": self.quality_flags[idx]}


    def set_transform(self, transform) -> None:
        self.transform = transform

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "crops": self.crops,
            "labels": self.labels,
            "scores": self.scores,
            "source_indices": self.source_indices,
            "quality_flags": self.quality_flags,
            "components": self.components, "boxes": self.boxes,
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, torch.Tensor]) -> "CropTensorDataset":
        return cls(
            crops=payload["crops"],
            labels=payload["labels"],
            scores=payload["scores"],
            source_indices=payload["source_indices"],
            quality_flags=payload["quality_flags"],
            components=payload["components"], boxes=payload["boxes"],
        )


