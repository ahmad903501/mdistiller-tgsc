"""Projector adapted verbatim from DefangChen/SimKD models/util.py (CVPR 2022)."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from ._base import Distiller
from ..tgsc.gates import weighted_mean

class SimKDProjector(nn.Module):
    """CVPR-2022: Knowledge Distillation with the Reused Teacher Classifier"""
    def __init__(self, *, s_n, t_n, factor=2):
        super(SimKDProjector, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        def conv3x3(in_channels, out_channels, stride=1, groups=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False, groups=groups)

        # A bottleneck design to reduce extra parameters
        setattr(self, 'transfer', nn.Sequential(
            conv1x1(s_n, t_n//factor),
            nn.BatchNorm2d(t_n//factor),
            nn.ReLU(inplace=True),
            conv3x3(t_n//factor, t_n//factor),
            # depthwise convolution
            #conv3x3(t_n//factor, t_n//factor, groups=t_n//factor),
            nn.BatchNorm2d(t_n//factor),
            nn.ReLU(inplace=True),
            conv1x1(t_n//factor, t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),
            ))

    def forward(self, feat_s, feat_t, cls_t):

        # Spatial Dimension Alignment
        s_H, t_H = feat_s.shape[2], feat_t.shape[2]
        if s_H > t_H:
            source = F.adaptive_avg_pool2d(feat_s, (t_H, t_H))
            target = feat_t
        else:
            source = feat_s
            target = F.adaptive_avg_pool2d(feat_t, (s_H, s_H))

        trans_feat_t=target

        # Channel Alignment
        trans_feat_s = getattr(self, 'transfer')(source)

        # Prediction via Teacher Classifier
        temp_feat = self.avg_pool(trans_feat_s)
        temp_feat = temp_feat.view(temp_feat.size(0), -1)
        pred_feat_s = cls_t(temp_feat)

        return trans_feat_s, trans_feat_t, pred_feat_s

def final_feature(features):
    return features["final_feat"] if "final_feat" in features else features["feats"][-1]


def classifier(model):
    for name in ("fc", "classifier", "linear"):
        if hasattr(model, name):
            return getattr(model, name)
    raise ValueError("No supported teacher classifier")


class SimKD(Distiller):
    """Official feature-only SimKD, with optional sample-wise TGSC weights."""
    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        previous = student.training
        student.eval()
        self.teacher_shapes = {}
        with torch.no_grad():
            for size in cfg.SIMKD.RESOLUTIONS:
                data = torch.zeros(2, 3, size, size)
                sf = final_feature(student(data)[1])
                tf = final_feature(teacher(data)[1])
                self.teacher_shapes[int(size)] = tuple(tf.shape[-2:])
            self.projector = SimKDProjector(s_n=sf.shape[1], t_n=tf.shape[1], factor=cfg.SIMKD.FACTOR)
        student.train(previous)
        self.head = copy.deepcopy(classifier(teacher)).requires_grad_(False)
        for name in ("fc", "classifier", "linear"):
            if hasattr(student, name):
                getattr(student, name).requires_grad_(False)
        self.feat_weight = cfg.SIMKD.FEAT_WEIGHT

    def train(self, mode=True):
        super().train(mode)
        self.head.eval()
        return self

    def get_learnable_parameters(self):
        return [p for p in self.student.parameters() if p.requires_grad] + list(self.projector.parameters())

    def get_extra_parameters(self):
        return sum(p.numel() for p in self.projector.parameters()) + sum(p.numel() for p in self.head.parameters())

    def forward_train(self, image, target=None, **kwargs):
        _, sf = self.student(image)
        with torch.no_grad():
            _, tf = self.teacher(image)
        source, target_feat, logits = self.projector(final_feature(sf), final_feature(tf), self.head)
        mse = (source - target_feat).square().flatten(1).mean(1)
        self.tgsc_diagnostics = {"raw_feature_mse": mse.detach().mean()}
        loss = self.feat_weight * weighted_mean(mse, kwargs.get("tgsc"), kwargs.get("gate_threshold", .75))
        return logits, {"loss_kd": loss}

    def forward_test(self, image):
        return simkd_predict(self.student, self.projector, self.head, self.teacher_shapes, image)


def simkd_predict(student, projector, head, teacher_shapes, image):
    if image.shape[-2] != image.shape[-1] or image.shape[-1] not in teacher_shapes:
        raise ValueError("SimKD inference resolution was not audited")
    source = final_feature(student(image)[1])
    teacher_size = teacher_shapes[image.shape[-1]]
    if source.shape[-2] > teacher_size[0]:
        source = F.adaptive_avg_pool2d(source, teacher_size)
    source = projector.transfer(source)
    return head(projector.avg_pool(source).flatten(1))
