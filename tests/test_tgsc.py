"""Focused contracts; no downloads or GPU are required for these checks."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import contextmanager
import shutil
import uuid
import io
import tarfile
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from mdistiller.distillers.DKD import DKD, dkd_components, dkd_loss
from mdistiller.distillers.ReviewKD import hcl_loss, ABF
from mdistiller.distillers.SimKD import final_feature
from mdistiller.tgsc.config import ExperimentConfig
from mdistiller.tgsc.experiment import load_run
from mdistiller.tgsc.cache import CropBank, manifest, prepare_bank
from mdistiller.tgsc.gates import quality_gate, weighted_mean
from mdistiller.tgsc.runtime import (build_model, build_distiller, export_payload, load_export,
                                    MergeSingletonBatches, train, seed_everything, resume_contract, load_teacher)

torch.set_num_threads(2)


@contextmanager
def test_directory():
    # Python 3.13's tempfile mode 0700 has incompatible ACLs in this Windows sandbox.
    root = (Path.cwd() / ".test-artifacts").resolve()
    path = root / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield str(path)
    finally:
        if path.resolve().is_relative_to(root):
            shutil.rmtree(path)


class MiniCifar(Dataset):
    def __init__(self):
        self.data = np.random.default_rng(7).integers(0, 256, (6, 32, 32, 3), dtype=np.uint8)
        self.targets = [0, 1, 2, 3, 4, 5]
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            (.5071, .4867, .4408), (.2675, .2565, .2761))])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.transform(self.data[index]), self.targets[index]


class TGSCContracts(unittest.TestCase):
    def test_legacy_teacher_numpy_metadata_loads_strictly(self):
        model = build_model("resnet8")
        with test_directory() as directory:
            checkpoint = Path(directory) / "legacy.pth"
            torch.save({"model": model.state_dict(), "best_acc": np.float64(72.3)}, checkpoint)
            restored = load_teacher(build_model("resnet8"), checkpoint)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key])

    def test_teacher_archive_member_selection_preserves_case(self):
        from tools.download_teachers import select_member
        memory = io.BytesIO()
        with tarfile.open(fileobj=memory, mode="w") as archive:
            info = tarfile.TarInfo("cifar_teachers/ResNet50_vanilla/ckpt_epoch_240.pth")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
        memory.seek(0)
        with tarfile.open(fileobj=memory, mode="r") as archive:
            self.assertEqual(select_member(archive,"ResNet50").name,
                             "cifar_teachers/ResNet50_vanilla/ckpt_epoch_240.pth")
            with self.assertRaises(ValueError):
                select_member(archive,"resnet50")

    def test_gates_weight_each_sample_and_detach(self):
        quality = torch.tensor([0., .375, .75, 3.], requires_grad=True)
        losses = torch.tensor([4., 8., 12., 16.], requires_grad=True)
        actual = weighted_mean(losses, {"q": quality})
        self.assertEqual(float(actual.detach()), 8.)
        actual.backward()
        torch.testing.assert_close(losses.grad, torch.tensor([0., .125, .25, .25]))
        self.assertIsNone(quality.grad)
        with self.assertRaises(ValueError):
            quality_gate(torch.tensor([float("nan")]))

    def test_dkd_components_loss_and_gradients(self):
        torch.manual_seed(1)
        s = torch.randn(4, 100, requires_grad=True)
        t, target = torch.randn(4, 100), torch.tensor([1, 2, 3, 4])
        native = dkd_loss(s, t, target, 1., 8., 4.)
        tc, nc = dkd_components(s, t, target, 4.)
        gated = weighted_mean(tc + 8*nc, {"q": torch.ones(4)})
        torch.testing.assert_close(native, gated)
        torch.testing.assert_close(torch.autograd.grad(native, s, retain_graph=True)[0],
                                   torch.autograd.grad(gated, s)[0])

    def test_hcl_per_sample_preserves_native_loss_and_gradient(self):
        s = [torch.randn(3, 8, r, r, requires_grad=True) for r in (8, 4, 1)]
        t = [torch.randn_like(x) for x in s]
        original, vector = hcl_loss(s,t), hcl_loss(s,t,per_sample=True)
        torch.testing.assert_close(original, vector.mean())
        for a,b in zip(torch.autograd.grad(original,s,retain_graph=True), torch.autograd.grad(vector.mean(),s)):
            torch.testing.assert_close(a,b)
        with self.assertRaises(ValueError):
            hcl_loss(s,t[:-1])

    def test_dkd_decoupled_routes_components_without_gating_ce(self):
        class Fixed(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.logits = torch.nn.Parameter(torch.randn(2,100))
            def forward(self, image):
                return self.logits, {}
        student,teacher=Fixed(),Fixed().requires_grad_(False)
        cfg=SimpleNamespace(DKD=SimpleNamespace(CE_WEIGHT=1.,ALPHA=1.,BETA=8.,T=4.,WARMUP=20))
        d=DKD(student,teacher,cfg)
        target=torch.tensor([0,1]);tc,nc=dkd_components(student.logits,teacher.logits,target,4.)
        for keep_target in (False,True):
            one,zero=torch.ones(2),torch.zeros(2)
            meta={"q":zero,"r1":one if keep_target else zero,"r3":one if keep_target else zero,
                  "r2":zero if keep_target else one}
            _,loss=d(image=None,target=target,epoch=20,tgsc=meta,gate_mode="decoupled")
            torch.testing.assert_close(loss["loss_kd"],tc.mean() if keep_target else 8*nc.mean())
            torch.testing.assert_close(loss["loss_ce"],torch.nn.functional.cross_entropy(student.logits,target))

    def test_abf_tuple_sizes_preserve_scalar_reference(self):
        abf = ABF(4, 8, 6, True).eval()
        x,y = torch.randn(2,4,8,8), torch.randn(2,8,4,4)
        with torch.no_grad():
            old = abf(x,y,8,16)
            dynamic = abf(x,y,(8,8),(16,16))
        for a,b in zip(old,dynamic):
            torch.testing.assert_close(a,b)

    def test_singleton_merge_preserves_all_samples(self):
        for n in (2, 3, 4, 5, 8, 9, 10):
            batches = MergeSingletonBatches(list(range(n)), 4)
            result = list(batches)
            self.assertEqual(len(result),len(batches))
            self.assertEqual([i for b in result for i in b],list(range(n)))
            self.assertTrue(all(len(b)>1 for b in result))

    def test_profiles_preserve_existing_values(self):
        for path in Path("configs/tgsc/cifar100").glob("*.json"):
            exp,method,options,_ = load_run(path)
            self.assertEqual(exp.tgsc.flip_prob,.5)
            self.assertEqual(exp.tgsc.aug_padding,2)
            self.assertEqual(exp.data.batch_size,128)
            self.assertEqual(sum(n for r,n in exp.data.stages),240)
            self.assertEqual(exp.tgsc.score_high,.75)

    def test_every_supplied_method_pair_at_each_curriculum_size(self):
        for path in Path("configs/tgsc/cifar100").glob("*.json"):
            exp,method,options,_ = load_run(path)
            teacher=build_model(exp.teacher_model).requires_grad_(False).eval()
            d=build_distiller(exp,method,options,teacher).train()
            with torch.no_grad():
                for size,_ in exp.data.stages:
                    logits,losses=d(image=torch.randn(2,3,size,size),target=torch.tensor([0,1]),epoch=1,
                                    tgsc={k:torch.ones(2) for k in ("q","r1","r2","r3")})
                    self.assertEqual(logits.shape,(2,100),msg=str(path))
                    self.assertTrue(torch.isfinite(sum(losses.values())),msg=f"{path} at {size}")

    def test_all_profile_models_have_valid_size_agnostic_pooling(self):
        names = ("resnet56", "resnet20", "resnet32x4", "resnet8x4", "wrn_40_2", "wrn_40_1",
                 "wrn_16_2", "vgg13", "vgg8", "ShuffleV1", "ShuffleV2", "MobileNetV2", "ResNet50")
        for name in names:
            model = build_model(name).eval()
            with torch.no_grad():
                for size in (16,20,24,28,32):
                    logits,feats = model(torch.randn(2,3,size,size))
                    self.assertEqual(logits.shape,(2,100),msg=f"{name} at {size}")
                    self.assertEqual(final_feature(feats).shape[0],2)

    def test_methods_all_sizes_and_simkd_export(self):
        for method in ("DKD","SIMKD","REVIEWKD"):
            exp = ExperimentConfig(teacher_model="resnet32x4", student_model="ShuffleV2")
            options = {"review_weight":8.}
            teacher = build_model(exp.teacher_model).requires_grad_(False).eval()
            d = build_distiller(exp,method,options,teacher)
            for size in (16,20,24,28,32):
                d.train()
                meta = {k:torch.tensor([.2,.8]) for k in ("q","r1","r2","r3")}
                logits,losses = d(image=torch.randn(2,3,size,size), target=torch.tensor([0,1]),
                                  epoch=2,tgsc=meta,gate_mode="decoupled")
                sum(losses.values()).backward()
                self.assertTrue(all(p.grad is None for p in teacher.parameters()))
                self.assertEqual(logits.shape,(2,100))
                d.zero_grad(set_to_none=True)
            if method == "SIMKD":
                d.eval()
                image=torch.randn(2,3,32,32)
                with torch.no_grad():
                    expected=d(image=image)
                    sf=final_feature(d.student(image)[1]);tf=final_feature(d.teacher(image)[1])
                    official_logits=d.projector(sf,tf,d.head)[2]
                torch.testing.assert_close(expected,official_logits)
                with test_directory() as directory:
                    path=Path(directory)/"export.pt"
                    torch.save(export_payload(d,exp,method,options),path)
                    restored,_=load_export(path)
                    with torch.no_grad():
                        torch.testing.assert_close(expected,restored(image))

    def test_cache_prepare_replay_identity_and_resume(self):
        exp=ExperimentConfig(teacher_model="resnet8",student_model="resnet8")
        exp.data.num_workers=0;exp.data.batch_size=4
        exp.data.stages=((16,1),(20,1));exp.train.epochs=2
        exp.tgsc.precompute_batch_size=3;exp.tgsc.precompute_workers=0
        dataset=MiniCifar()
        with test_directory() as directory:
            root=Path(directory)
            seed_everything(27)
            teacher=build_model("resnet8").requires_grad_(False).eval()
            checkpoint=root/"teacher.pt";torch.save({"model":teacher.state_dict()},checkpoint)
            identity=manifest(exp,checkpoint,dataset)
            bank=CropBank(identity,root/"cache",root/"cache")
            prepare_bank(exp,teacher,dataset,bank)
            crops=bank.load(16)
            torch.testing.assert_close(crops.scores,crops.components.mean(1).clamp(0,1))
            for i in range(len(crops)):
                source=int(crops.source_indices[i]); y0,x0,y1,x1=crops.boxes[i]
                torch.testing.assert_close(crops.crops[i],dataset[source][0][:,y0:y1,x0:x1])
            readonly=CropBank(identity,root/"cache")
            self.assertEqual(len(readonly.load(16)),len(crops))
            incompatible=CropBank(dict(identity,checkpoint_sha256="different"),root/"cache")
            with self.assertRaises(FileNotFoundError):incompatible.load(16)
            for method in ("DKD","SIMKD","REVIEWKD"):
                options={"review_weight":1.}
                seed_everything(27)
                distiller=build_distiller(exp,method,options,teacher)
                train(exp,method,options,distiller,readonly,dataset,torch.device("cpu"),root/method,stop_after_epoch=1)
                seed_everything(27)
                continued=build_distiller(exp,method,options,teacher)
                train(exp,method,options,continued,readonly,dataset,torch.device("cpu"),root/(method+"_resume"),resume=root/method/"latest.pt")
                state=torch.load(root/(method+"_resume")/"latest.pt",weights_only=True)
                self.assertEqual(state["epoch"],2)
                model,_=load_export(root/(method+"_resume")/"inference_latest.pt")
                with torch.no_grad():self.assertEqual(model(torch.randn(2,3,32,32))[0].shape if method!="SIMKD" else model(torch.randn(2,3,32,32)).shape,(2,100))
                seed_everything(27)
                uninterrupted=build_distiller(exp,method,options,teacher)
                train(exp,method,options,uninterrupted,readonly,dataset,torch.device("cpu"),root/(method+"_full"))
                for key,value in continued.state_dict().items():
                    torch.testing.assert_close(value,uninterrupted.state_dict()[key],msg=f"Resume mismatch: {method}/{key}")
            a={"experiment":asdict(exp),"method":"DKD","options":{}}
            b=copy.deepcopy(a);b["experiment"]["data"]["root"]="relocated"
            self.assertEqual(resume_contract(a),resume_contract(b))


if __name__ == "__main__":
    unittest.main()
