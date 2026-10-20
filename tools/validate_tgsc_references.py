"""Development-only comparisons against local reference repos; not a runtime dependency."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from mdistiller.distillers.SimKD import SimKDProjector
from mdistiller.distillers.ReviewKD import ABF, hcl_loss
from mdistiller.tgsc import oracle
from mdistiller.tgsc.runtime import build_model


def definitions(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    scope = {"torch":torch,"nn":nn,"F":F}
    exec(compile(tree,str(path),"exec"),scope)
    return scope


def compare(reference_root):
    torch.set_num_threads(2);torch.manual_seed(27)
    root = Path(reference_root)
    ref = definitions(root/"SimKD/models/util.py", {"SimKD"})["SimKD"](s_n=8,t_n=16,factor=2)
    port = SimKDProjector(s_n=8,t_n=16,factor=2)
    port.load_state_dict(ref.state_dict());ref.eval();port.eval()
    for sizes in ((8,4),(4,8),(4,4)):
        s=torch.randn(2,8,sizes[0],sizes[0],requires_grad=True)
        t=torch.randn(2,16,sizes[1],sizes[1]);head=nn.Linear(16,100)
        a,b=ref(s,t,head),port(s,t,head)
        for x,y in zip(a,b):torch.testing.assert_close(x,y)
        torch.testing.assert_close(torch.autograd.grad(a[0].square().mean(),s,retain_graph=True)[0],
                                   torch.autograd.grad(b[0].square().mean(),s)[0])
    review=definitions(root/"ReviewKD/CIFAR-100/model/reviewkd.py",{"ABF","hcl"})
    a=review["ABF"](8,16,32,True).eval();b=ABF(8,16,32,True).eval();b.load_state_dict(a.state_dict())
    s=torch.randn(2,8,8,8);r=torch.randn(2,16,4,4)
    for x,y in zip(a(s,r,8,16),b(s,r,(8,8),(16,16))):torch.testing.assert_close(x,y)
    s=[torch.randn(2,8,r,r,requires_grad=True) for r in (8,4,1)];t=[torch.randn_like(x) for x in s]
    expected=review["hcl"](s,t);actual=hcl_loss(s,t,per_sample=True).mean()
    torch.testing.assert_close(expected,actual)
    for x,y in zip(torch.autograd.grad(expected,s,retain_graph=True),torch.autograd.grad(actual,s)):
        torch.testing.assert_close(x,y)
    # Reference package imports its own config/utils, never the installed mdistiller port.
    pkg=types.ModuleType("tgsc_reference");pkg.__path__=[str(root/"TGSC/src/tgsc_kd")]
    sys.modules[pkg.__name__]=pkg
    spec=importlib.util.spec_from_file_location("tgsc_reference.tgsc",root/"TGSC/src/tgsc_kd/tgsc.py")
    legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
    for name in ("resnet8","vgg8","wrn_16_2"):
        teacher=build_model(name).eval().requires_grad_(False)
        data=torch.randn(2,3,32,32);target=torch.tensor([0,1])
        loader=DataLoader(TensorDataset(data,target),batch_size=2)
        options=dict(use_gt_target=True,w_cam=.7,w_sal=.3,temp=1.5,gamma=1.2,pct_lo=1.,pct_hi=99.)
        target_layer=oracle.pick_target_layer(teacher)
        refcache=legacy.build_teacher_union_cache(loader,teacher,target_layer,**options)
        newcache=oracle.build_teacher_union_cache(loader,teacher,target_layer,**options)
        torch.testing.assert_close(refcache.union_maps,newcache.union_maps)
        for size in (16,20,32):
            kwargs=dict(crop_size=size,stride=8,k_per_img=6,min_frac=.2,min_mean=.2,eval_chunk=2,
                        top_k_crops=2,low_k_crops=1,teacher_cache=refcache,**options)
            expected=legacy.build_bestcrop_dataset_batchwise(loader,teacher,target_layer,**kwargs)
            kwargs["teacher_cache"]=newcache
            actual=oracle.build_bestcrop_dataset_batchwise(loader,teacher,target_layer,**kwargs)
            for field in ("crops","labels","scores","source_indices","quality_flags"):
                torch.testing.assert_close(getattr(expected,field),getattr(actual,field))
    pooling={}
    for file,builder in (("resnet.py","resnet8"),("wrn.py","wrn_16_2"),
                         ("mobilenetv2.py","mobile_half"),("ShuffleNetv2.py","ShuffleV2"),
                         ("ShuffleNetv1.py","ShuffleV1")):
        source=subprocess.check_output(["git","show",f"HEAD:mdistiller/models/cifar/{file}"],cwd=Path(__file__).parents[1],text=True)
        scope={"__name__":"original_model"};exec(compile(source,file,"exec"),scope)
        original=scope[builder](num_classes=100).eval()
        name="MobileNetV2" if builder=="mobile_half" else builder
        adapted=build_model(name).eval();adapted.load_state_dict(original.state_dict())
        data=torch.randn(2,3,32,32)
        with torch.no_grad(): a=original(data)[0];b=adapted(data)[0]
        pooling[name]=float((a-b).abs().max())
        if name!="ShuffleV1":torch.testing.assert_close(a,b)
    return {"simkd_projector":"pass: outputs and input gradients", "review_abf_hcl":"pass: outputs/loss/gradients",
            "oracle":"pass: CAM and selected crop/q parity at 16,20,32 on three model families",
            "pooling_32_max_abs":pooling,"device":"cpu","torch":str(torch.__version__)}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--reference-root",required=True);p.add_argument("--output")
    args=p.parse_args();result=compare(args.reference_root)
    text=json.dumps(result,indent=2)
    if args.output:Path(args.output).write_text(text+"\n",encoding="utf-8")
    print(text)
