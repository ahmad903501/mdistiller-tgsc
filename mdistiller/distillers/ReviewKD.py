import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb

from ._base import Distiller
from ..tgsc.gates import weighted_mean


def hcl_loss(fstudent, fteacher, per_sample=False):
    if len(fstudent) != len(fteacher) or not fstudent:
        raise ValueError("ReviewKD stage counts must match and be nonempty")
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n, c, h, w = fs.shape
        if fs.shape != ft.shape:
            raise ValueError("ReviewKD aligned feature shapes differ")
        loss = (fs - ft).square().flatten(1).mean(1) if per_sample else F.mse_loss(fs, ft)
        cnt = 1.0
        tot = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
            tmpft = F.adaptive_avg_pool2d(ft, (l, l))
            cnt /= 2.0
            loss += ((tmpfs - tmpft).square().flatten(1).mean(1) if per_sample else F.mse_loss(tmpfs, tmpft)) * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all


class ReviewKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(ReviewKD, self).__init__(student, teacher)
        self.shapes = cfg.REVIEWKD.SHAPES
        self.out_shapes = cfg.REVIEWKD.OUT_SHAPES
        in_channels = cfg.REVIEWKD.IN_CHANNELS
        out_channels = cfg.REVIEWKD.OUT_CHANNELS
        self.ce_loss_weight = cfg.REVIEWKD.CE_WEIGHT
        self.reviewkd_loss_weight = cfg.REVIEWKD.REVIEWKD_WEIGHT
        self.warmup_epochs = cfg.REVIEWKD.WARMUP_EPOCHS
        self.stu_preact = cfg.REVIEWKD.STU_PREACT
        self.max_mid_channel = cfg.REVIEWKD.MAX_MID_CHANNEL
        self.dynamic_shapes = getattr(cfg.REVIEWKD, "TGSC_DYNAMIC", False)

        abfs = nn.ModuleList()
        mid_channel = min(512, in_channels[-1])
        for idx, in_channel in enumerate(in_channels):
            abfs.append(
                ABF(
                    in_channel,
                    mid_channel,
                    out_channels[idx],
                    idx < len(in_channels) - 1,
                )
            )
        self.abfs = abfs[::-1]

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.abfs.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.abfs.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        logits_student, features_student = self.student(image)
        with torch.no_grad():
            logits_teacher, features_teacher = self.teacher(image)

        if self.dynamic_shapes:
            key = "preact_feats" if self.stu_preact else "feats"
            x = features_student[key][1:] + [features_student["pooled_feat"][:, :, None, None]]
            targets = features_teacher["preact_feats"][1:] + [features_teacher["pooled_feat"][:, :, None, None]]
            if len(x) != len(targets) or len(x) != len(self.abfs):
                raise ValueError("ReviewKD feature/ABF counts differ")
            results, residual = [], None
            for fs, ft, abf in zip(x[::-1], targets[::-1], self.abfs):
                out, residual = abf(fs, residual, shape=fs.shape[-2:], out_shape=ft.shape[-2:])
                results.insert(0, out)
            ramp = min(kwargs["epoch"] / self.warmup_epochs, 1.0) if self.warmup_epochs else 1.0
            hcl = hcl_loss(results, targets, per_sample=True)
            self.tgsc_diagnostics = {"raw_hcl": hcl.detach().mean()}
            return logits_student, {
                "loss_ce": self.ce_loss_weight * F.cross_entropy(logits_student, target),
                "loss_kd": self.reviewkd_loss_weight * ramp * weighted_mean(
                    hcl, kwargs.get("tgsc"), kwargs.get("gate_threshold", .75)
                ),
            }

        # get features
        if self.stu_preact:
            x = features_student["preact_feats"] + [
                features_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
        else:
            x = features_student["feats"] + [
                features_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
        x = x[::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0], out_shape=self.out_shapes[0])
        results.append(out_features)
        for features, abf, shape, out_shape in zip(
            x[1:], self.abfs[1:], self.shapes[1:], self.out_shapes[1:]
        ):
            out_features, res_features = abf(features, res_features, shape, out_shape)
            results.insert(0, out_features)
        features_teacher = features_teacher["preact_feats"][1:] + [
            features_teacher["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
        ]
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_reviewkd = (
            self.reviewkd_loss_weight
            * min(kwargs["epoch"] / self.warmup_epochs, 1.0)
            * hcl_loss(results, features_teacher)
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_reviewkd,
        }
        return logits_student, losses_dict


class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None, out_shape=None):
        n, _, h, w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            size = (shape, shape) if isinstance(shape, int) else shape
            y = F.interpolate(y, size, mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w)
        # output
        output_size = (out_shape, out_shape) if isinstance(out_shape, int) else out_shape
        if x.shape[-2:] != output_size:
            x = F.interpolate(x, output_size, mode="nearest")
        y = self.conv2(x)
        return y, x
