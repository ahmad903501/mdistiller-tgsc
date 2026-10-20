# TGSC progress and handoff

Last updated: 2026-10-21 (Asia/Karachi).

## Read this first after compaction

1. Read this file for current state and accepted decisions.
2. Read [implementation tasks](../TGSC_IMPLEMENTATION_TASKS.md) for the next task's scope/checks.
3. Consult [integration plan](../TGSC_INTEGRATION_PLAN.md) for detailed contracts; FitNet notes are historical and deferred.
4. Check `git status --short` and relevant diffs in the mdistiller checkout before editing. Verify actual files/check results rather than treating this report as proof that code exists or still works.
5. Resume the next incomplete task. Do not repeat completed work unless new changes or evidence justify it.

Latest explicit user instructions take precedence over this report. When a decision changes, update the current snapshot and append the change to the history so older discussions are not mistaken for the current plan.

## Current snapshot

- **Phase:** implementation active, authorized by the user's latest request to begin work.
- **Completed:** reference audits (01-03), first implementation of self-contained TGSC runtime, model adaptations, separate oracle metrics, persistent crop banks, gates, three distillers, epoch resume/exports, runner, and four thin notebooks.
- **Next action:** task 15: real CIFAR/teacher-checkpoint GPU pilot and actual Kaggle execution. Start from `docs/TGSC_RUNNING.md` and the preparation notebook. This host has CPU-only torch, so do not report GPU acceptance as complete. Do not redo the completed audits/CPU checks without new changes.
- **Task status:** 01-14 implemented and checked at the source/CPU/isolated-notebook-structure level. Task 15 is partial: synthetic CPU end-to-end checks pass; real-data GPU and Kaggle acceptance remain pending.
- **Checks performed:** `python -m unittest discover -s tests -q`: 11 tests pass on CPU (all supplied method/pair profiles at 16/20/24/28/32, sample-weight gradients, crop save/replay, shared bank identity, all-method resume versus uninterrupted execution, and exports). `python tools/validate_tgsc_references.py --reference-root .. --output docs/TGSC_REFERENCE_CHECKS.json`: passes official SimKD/ReviewKD component comparisons and TGSC oracle crop parity on ResNet/VGG/WRN families.
- **Environment:** Windows PowerShell; Python 3.13, torch 2.6.0+cpu, torchvision 0.21.0+cpu. CUDA unavailable; YACS missing. New JSON-based runner deliberately does not require YACS. No real CIFAR/checkpoint training or Kaggle execution claimed.
- **Repository:** all changes are inside mdistiller; sibling references untouched. No commit/push/remote modification performed.
- **Working tree:** implementation, docs, profiles and notebooks are edited/untracked. Recheck live Git status/diffs.
- **Known numerical difference:** TGSC's existing internal ShuffleV1 pooling adaptation changes 32x32 predictions (random-weight reference max absolute difference .01352 in saved check); other audited pooling changes match within FP32 tolerance.
- **Known tooling issue:** Python 3.13 tempfile(mode 0700) created inaccessible test folders in this sandbox. Tests now use a validated workspace scratch directory with compatible inherited permissions; initial test failures were filesystem errors, not algorithm failures.

## Accepted decisions

| Topic | Current instruction |
| --- | --- |
| Scope | Official DKD, SimKD, ReviewKD plus TGSC. FitNet deferred. Future methods should reuse the shared framework. |
| Implementation location | Change only mdistiller; sibling repos/PDFs are reference material. Runtime must work from one clone without siblings. |
| TGSC behavior | Preserve score definitions, ranking/clipping order, crop selection/multiplicity, gate, and selected pair settings. No silent redesign. |
| Augmentation | Keep TGSC defaults: `flip_prob=0.5`, `aug_padding=2`. Earlier proposal to disable them is superseded. Do not add duplicate mdistiller augmentation. |
| Misclassified crops | Retain existing top-two margin definition; do not reject a crop solely because teacher classification is wrong. |
| Losses/alignment | Preserve official losses and alignment mechanisms. No added CE/KL hybrids, new 1x1 FitNet adapters, or coefficient experiments. |
| Gating | Apply detached gates to per-sample distillation losses before batch mean. Preserve CE where native; do not add CE to feature-only SimKD. |
| DKD | Unified and requested decoupled gating; cache raw r1/r2/r3 alongside existing stored q. |
| Cache reuse | Prepare once per dataset/teacher checkpoint/preprocessing configuration and resolution bank. Reuse across methods/students. Different teachers generally require different banks. |
| Cache versus augmentation | Cache base crops and scores; training applies default TGSC augmentation. Scores still describe base crops: retain this existing approximation. Teacher/student see the same augmented tensor. |
| Pooling | Reuse existing TGSC input-size adaptations in mdistiller. Preserve official intermediate-feature alignment. |
| Baseline experiments | No newly trained runs without TGSC. Use available baselines with provenance/protocol differences disclosed. Unity-gate numerical checks remain allowed. |
| Delivery | Preparation notebook plus one notebook each for DKD/SimKD/ReviewKD; all call a shared Python runner. |
| GPU scope | One selected GPU per process initially; independent jobs on separate GPUs. DDP is deferred. |
| Recovery/export | Epoch-boundary resume and persistent Kaggle artifacts. SimKD export includes projector and frozen teacher head. |
| Publishing | User explicitly authorized creating a new GitHub repo and pushing in the latest turn. Planned private mdistiller-tgsc repository; owner account selection pending among saved accounts. Kaggle dataset uploads remain user-managed. |

## Verified facts worth carrying forward

- TGSC model copies already use adaptive final pooling in ResNet, WRN, MobileNetV2, and ShuffleNet; VGG/ResNetV2 also have adaptive final pooling. TGSC ShuffleNetV1 has an additional internal adaptive pooling replacement. Verify behavior when porting; not every change guarantees exact 32x32 parity.
- Existing TGSC caches save cropped tensors, labels, composite q, source indices, and selection flags; separate metric components are not persisted. Existing cache identity lacks teacher checkpoint content hashing.
- TGSC provides optional disk caching. ResNet32x4 and VGG JSONs have it off; WRN40-2 -> WRN40-1 has it on.
- All three inspected TGSC JSONs use batch 128 and five 48-epoch stages at 16/20/24/28/32. ResNet32x4/VGG milestones are `[48,144,192]`; WRN pair uses `[48,144,192,210]`. Do not flatten these differences into one invented default.
- DKD/ReviewKD's 20-epoch warmups apply to their distillation loss, not LR.
- Local ReviewKD resnet32x4 -> resnet8x4 YAML overrides review weight to 5, although global default is 1.
- Local DKD WRN40-2 -> ShuffleV1 YAML inherits beta 8; the user's initially pasted blanket WRN beta 6 does not describe that merged config.
- Official SimKD example uses feature-loss weight 1, CE/KL zero, factor 2. The reused teacher classifier and projector are needed at inference.
- mdistiller's original requirements pin torch 1.9/torchvision 0.10; don't install these blindly over Kaggle's compatible GPU stack.

These observations are from source/config inspection, not numerical validation. Recheck affected code during its assigned task.

## Reference locations and revisions

Relative to the mdistiller checkout:

- DKD: local `mdistiller/distillers/DKD.py`, `configs/cifar100/dkd/`, `../DKD.pdf`.
- SimKD: `../SimKD/models/util.py`, `helper/loops.py`, `train_student.py`, `scripts/run_distill.sh`, `../SimKD.pdf`.
- ReviewKD: `../ReviewKD/CIFAR-100/` and local `mdistiller/distillers/ReviewKD.py`, `../ReviewKD.pdf`.
- TGSC: `../TGSC/src/tgsc_kd/`, `../TGSC/configs/`, `../TGSC/model/python/`.
- FitNet: `../fitnet.pdf`; future work only. Do not spend current implementation effort recovering its historical epoch counts.

Previously inspected revisions, to be verified live if needed:

| Repository | Revision |
| --- | --- |
| mdistiller | `a08d46f10d6102bd6e3f258ca5ac880b020ea259` |
| SimKD | `2b389c31ed7779aea31e7aaf0bb0f2d8b6ac2f01` |
| ReviewKD | `cede6ea6387ae9b6127de0e561507177bf19c11e` |

## Task tracker

| ID | Task | Status |
| --- | --- | --- |
| 01 | DKD reference audit | Complete (CPU checks) |
| 02 | SimKD reference audit | Complete (CPU checks) |
| 03 | ReviewKD reference audit | Complete (CPU checks) |
| 04 | Self-contained TGSC config/package | Complete (offline checks) |
| 05 | Variable-size model compatibility | Complete (CPU checks) |
| 06 | Crop generation and separate metrics | Complete (CPU checks) |
| 07 | Persistent teacher-specific crop banks | Complete (CPU checks) |
| 08 | Shared per-sample gates | Complete (CPU checks) |
| 09 | DKD integration | Complete (CPU checks) |
| 10 | SimKD integration and inference | Complete (CPU checks) |
| 11 | ReviewKD integration | Complete (CPU checks) |
| 12 | Curriculum trainer/checkpoint recovery | Complete (CPU checks) |
| 13 | Python runner/GPU selection | Complete (offline checks) |
| 14 | Kaggle notebooks/instructions | Complete (offline checks) |
| 15 | GPU acceptance/pair expansion | Partial: CPU pilot passed; GPU/Kaggle pending |

## History and verification log

### 2026-10-21: planning consolidated; handoff initialized

- Created the integration plan and then the 15-task breakdown; no implementation changes.
- Superseded early proposals: disabling default augmentation, altering oracle clipping, imposing a universal 20 crop floor, changing crop multiplicity, adding loss hybrids, and training fresh ungated baselines.
- Investigated original FitNet's two-stage procedure and uncertain epoch budget; user subsequently deferred FitNet entirely.
- Confirmed mdistiller is the sole implementation/push target and references remain guidance only.
- Confirmed newly available `DKD.pdf`, `SimKD.pdf`, `ReviewKD.pdf` titles using first-page text extraction. Full formal audits remain pending.
- Added this progress/handoff record at the user's explicit request to survive context compaction.
- Verification during this handoff: inspected live Git status and task list; checked document links/15 task entries. No training result is claimed.

## Update discipline

Keep the snapshot and task tracker current; append substantive progress entries instead of rewriting historical outcomes. Update after each completed task, significant decision, failure that changes the approach, or interrupted partial task.

Each entry should include:

- Task ID/status and concrete outcome.
- Changed paths and why.
- Reference revision and paper section/equation actually consulted.
- Exact focused check commands, environment, concise results, and evidence/output paths when saved.
- For failed checks: primary error, what was tried, and the remaining issue; no large passing logs.
- Next action and any unresolved decision. For partial work, identify edited files and checks not yet run.

Record commit hashes only after commits actually exist. Record GPU/Kaggle verification only after execution. Before compaction or ending an implementation turn, ensure the next action is concrete enough to resume without reconstructing the chat.

### 2026-10-21: first implementation and CPU reference checks

- Authorized by the user's request to begin work; implementation proceeded through tasks without repeated approval gates.
- Added `docs/TGSC_REFERENCE_AUDIT.md` after comparing official DKD code/configs, official SimKD projector/loop, official ReviewKD ABF/HCL/factory/scripts, and relevant supplied PDF method sections.
- Ported TGSC config/utilities/oracle into `mdistiller/tgsc/`; retained formulas/selection and default augmentation while adding raw components/boxes/source metadata.
- Added persistent schema-2 crop bank with teacher checkpoint + dataset + normalized source identity, atomic writes, separate read/write roots and writer lock. No metric redesign.
- Ported inspected TGSC pooling changes into CIFAR models; exposed final prepool features where needed for SimKD.
- Added shared gate helpers, per-sample DKD unified/decoupled paths, official SimKD projector/inference graph, dynamic-size ReviewKD with native HCL, and explicit method configs.
- Added runtime trainer/resume/inference exports and GPU-selecting Python runner. Added preparation + three method notebooks and running guide; final isolated validation is next.
- Commands: `python -m unittest discover -s tests -q` (10 tests, pass); `python tools/validate_tgsc_references.py --reference-root .. --output docs/TGSC_REFERENCE_CHECKS.json` (pass). Subprocess capture was used to filter repetitive output; original legacy DKD emits a deprecation warning.
- Resume checks compare full final state against uninterrupted CPU execution for DKD/SimKD/ReviewKD across 16 -> 20 crop transition. Synthetic pixels/random weights only; not accuracy evidence.
- Official source discrepancy retained/documented: ReviewKD released script starts warmup epoch at zero; mdistiller starts at one. Pair-specific weights are explicit. VGG ReviewKD config omitted pending verified pair coefficient; no guessed default.
- No CUDA device here. Task 15 cannot be called complete until real-data/GPU and Kaggle execution occur.

### 2026-10-21: isolated delivery validation completed

- Tasks 04/13/14 completed for offline delivery: self-contained package/configs, shared Python CLI, GPU selection/error handling, four thin notebooks and running guide. README links to the workflow.
- Exact delivery check: `python tools/validate_tgsc_delivery.py`. It copies only runtime/config/tools/tests into a workspace scratch tree, clears PYTHONPATH, verifies imported runtime comes from that tree, runs all 11 self-contained tests, parses all four notebooks and checks CLI input errors. Result: pass. Evidence: `docs/TGSC_VALIDATION.json`.
- Checks now include independently zeroing DKD target/non-target gates while CE stays unchanged, all 11 supplied configs at five resolutions, reference component gradients, crop/cache replay, and per-method CPU resume versus uninterrupted final state.
- Corrected resume portability: checkpoint carries cloned best inference weights so `inference_best.pt` is available in a fresh output directory even when no later epoch improves. Run metadata includes launch arguments, dependency versions and parameter counts.
- Normalized source hashing across Windows/Linux line endings. Added crop-box/q consistency checks, retained only existing TGSC transforms, and documented derived heterogeneous profile provenance.
- `git diff --check`: passed. The two inaccessible workspace scratch folders from the initial tempfile failure were removed successfully using approved native PowerShell cleanup; no remaining cleanup issue is known.
- No reference repository changes, Git commit/push, dataset download, full training, or external publication. No CUDA/Kaggle execution claimed. Real checkpoint loading/accuracy, GPU memory/performance, and actual Kaggle artifact/session behavior still require task 15.
- Supported initial profiles: four pairs each for DKD/SimKD (three existing TGSC pair profiles plus explicit derived ResNet32x4 -> ShuffleV2), three ReviewKD pairs. VGG ReviewKD is omitted until a pair coefficient is verified; other pairs require explicit audited configs.
- FLOPs and long-run accuracy are not measured. CPU timings/synthetic test accuracy are correctness diagnostics only.

### 2026-10-21: publication preparation and Kaggle T4 guidance

- User authorized push and then clarified that a new repository must be created. Current origin is still Megvii upstream; do not push there. Planned new private repository name: `mdistiller-tgsc`, preserving Megvii as upstream. Owner account selection is pending; multiple saved GitHub accounts exist.
- Added private-repo clone support to all four notebooks using Kaggle Secret `GITHUB_TOKEN` and temporary Git askpass code. No token is embedded in notebook source, Git URLs, or generated files.
- Expanded running guide with one-T4-per-job recommendation, two independent GPU 0/1 subprocess launches within one session, cache-output-to-Dataset steps, and one-epoch smoke/resume workflow.
- `python tools/validate_tgsc_delivery.py`: all 11 tests/isolated checks and four notebook parses pass after these updates. Embedded askpass Python also parses; credential-pattern scan found no embedded secrets.
- CUDA/Kaggle throughput is still unmeasured. DDP is not implemented; selecting T4 x2 alone does not make a single method use both GPUs.
