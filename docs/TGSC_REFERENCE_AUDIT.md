# Reference audit: DKD, SimKD, ReviewKD

2026-10-21. Source audit completed; numerical port checks are separate implementation tasks. PDFs and sibling repositories are development references only.

## DKD (task 01)

Reference: mdistiller revision `a08d46f10d6102bd6e3f258ca5ac880b020ea259`, `distillers/DKD.py`, `engine/cfg.py`, `configs/cifar100/dkd/`; supplied DKD PDF section on decoupling target/non-target knowledge.

TCKD is KL between two-bin target-versus-other probabilities. NCKD is KL after masking the target logit with -1000. Both use temperature-softened probabilities, class-sum KL, T squared, and batch division. Per-sample implementation must sum classes before gating and batch mean. Preserve alpha/beta independently; do not reintroduce classical KD's confidence coupling. CE uses ground truth. Loss ramp is `min(epoch/20,1)` with mdistiller's one-based epoch convention; zero warmup means 1.

Merged examples: resnet32x4 -> resnet8x4 inherits alpha=1,beta=8,T=4,warmup=20; resnet32x4 -> ShuffleV2 also inherits these, but its official YAML sets student LR .01 instead of .05. resnet56 -> resnet20 overrides beta=2; WRN -> WRN overrides beta=6; WRN -> ShuffleV1 has no override and inherits 8. TGSC optimizer/schedule are supplied separately from the existing pair profile.

Required TGSC changes: expose unreduced TCKD/NCKD; apply unified or component-specific gates; preserve CE and ramp. No new loss term.

## SimKD (task 02)

Reference: official clone revision `2b389c31ed7779aea31e7aaf0bb0f2d8b6ac2f01`, `models/util.py` SimKD class, `helper/loops.py`, `train_student.py`, `scripts/run_distill.sh`; supplied paper's feature regression/reused-classifier method and bottleneck discussion.

Official projector: 1x1(Cs,Ct/2), 3x3(Ct/2,Ct/2,pad=1), 1x1(Ct/2,Ct), no conv bias, each with BN/ReLU. Pool the larger feature grid to the smaller before regression. Loss is elementwise mean MSE; per-sample reduction must mean C/H/W then batch. The example command uses `-c 0 -d 0 -b 1`, factor 2: no CE/KL, feature coefficient 1. The unrelated Proj helper is not this projector.

Official `feat[-2]` is the final spatial feature before global pooling. In mdistiller, `feats[-1]` matches this for ResNet/VGG/ResNetV2. WRN's dictionary feature precedes its final BN/ReLU; ShuffleV2 and MobileNetV2 have a final conv after `feats[-1]`. Explicitly expose the actual prepool output as `final_feat` for these models rather than project a different layer. Student and teacher head dimensions must match the selected final spatial feature. Head attributes include fc/classifier/linear. The official SimKD clone does not include the WRN40-2 backbone; using the verified projector/objective with the supplied TGSC WRN architecture is a backbone extension, not a claim of reproducing a released WRN SimKD checkpoint.

Inference retains student backbone, projector, global pooling, and frozen teacher classifier. Its pooling target uses teacher spatial geometry in the reference; inference must reproduce that geometry without running teacher features (derive and store the supported-resolution target grids). This matters for heterogeneous pairs when the student final grid is larger. Teacher head parameters are frozen, not detached from the projected student computational graph. Student's unused classifier must not receive optimization.

## ReviewKD (task 03)

Reference: official clone revision `cede6ea6387ae9b6127de0e561507177bf19c11e`, `CIFAR-100/model/reviewkd.py`, `train.py`, `script/reviewKD.sh`; supplied paper sections 3.2/3.3 on residual review, ABF, and HCL. Compare local `distillers/ReviewKD.py`.

ABF uses 1x1/BN transform, two sigmoid attention maps for current/residual fusion, nearest interpolation, then 3x3/BN output. Native mid width is min(512, deepest student channels). HCL sums stage losses; each stage starts with full-grid MSE and adds valid adaptive pools [4,2,1] with successive half weights, normalizing by accumulated weights. Preserve this order per sample. Native objective is CE + coefficient * ramp * HCL; optional official KL flag is not active in the cited base scripts, so it is not added.

Homogeneous ResNet32x4 -> ResNet8x4: student postactivation feats[1:] plus pooled feature, teacher preactivation feats[1:] plus pooled feature. The current mdistiller defaults include an extra student stem but silently truncate it through zip; select the intended three stages explicitly instead. Reference channel vectors [64,128,256,256].

Heterogeneous ResNet32x4 -> ShuffleV2: three student stages [116,232,464] plus pooled 1024, teacher stages [64,128,256] plus pooled 256. At 32 input student grids [16,8,4,1], teacher [32,16,8,1]. Residual interpolation targets student grid before fusion; output interpolation targets teacher grid. Shapes are runtime values under TGSC.

Reference script weights: ResNet32x4 -> ResNet8x4=5; ->ShuffleV2=8; WRN40-2 -> WRN16-2/WRN40-1/ShuffleV1=5; ResNet56 -> ResNet20=.6. The single mdistiller default YAML only establishes weight 5 for its named pair. VGG/MobileNet combinations are described in factory branches but some imports are commented out in the released clone; shape compatibility can be implemented from those branches, but do not invent a verified pair weight.

Epoch discrepancy: official train.py loops from zero and uses epoch/20; mdistiller calls from one. Retain mdistiller one-based ramp for integration and document it. Official-script coefficient and original fusion/HCL are retained. No proof of bitwise official backbone parity is claimed for TGSC's modified internal ShuffleV1 pooling.

## Initial supported configs

Use the three existing TGSC profiles: ResNet32x4 -> ResNet8x4, WRN40-2 -> WRN40-1, VGG13 -> VGG8. VGG ReviewKD has no verified active script coefficient: do not issue a supposedly official VGG ReviewKD config until that coefficient is established. Add a clearly documented ResNet32x4 -> ShuffleV2 compatibility pilot using the ResNet32x4 teacher crop bank and official student LR .01; this is an explicit derived TGSC profile, not an existing JSON reproduction. New pairs must state profile provenance and verified method settings.

## Provenance

Upstream code identities: https://github.com/megvii-research/mdistiller , https://github.com/DefangChen/SimKD , https://github.com/JIA-Lab-research/ReviewKD . Ported files retain attribution. Check source license terms before distributing copied modules; do not claim a license grant where a reference repository contains none.
