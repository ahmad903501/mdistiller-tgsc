# TGSC implementation tasks

Status: implementation active. Current completion/evidence is tracked in the linked progress report. Main specification: [TGSC_INTEGRATION_PLAN.md](TGSC_INTEGRATION_PLAN.md).

Current status and handoff: [docs/TGSC_IMPLEMENTATION_PROGRESS.md](docs/TGSC_IMPLEMENTATION_PROGRESS.md). Read that file first after context compaction; update its snapshot/tracker and append evidence as implementation progresses.

## Scope and working rules

- Implement only in this `mdistiller/` repository. The user will push and clone this repository for Kaggle/local runs.
- Treat sibling TGSC, SimKD, and ReviewKD repositories, and the attached PDFs, as read-only references. Port required code into this package with provenance; do not create runtime dependencies on reference folders.
- First delivery: DKD, SimKD, ReviewKD. FitNet is deferred.
- Preserve current TGSC behavior, including default augmentation, scoring, crop multiplicity, and selected pair settings. Preserve official method losses/alignment; only make the agreed TGSC weighting and necessary variable-size compatibility changes.
- Work through small checkpoints. Record files changed, reference evidence, commands/checks, result, and remaining limitations for each task. Do not call a task complete merely because code imports.
- Use tiny deterministic tensors or a small data subset for correctness checks. Full 240-epoch experiments are not a prerequisite for each task. Synthetic checks validate contracts, not final accuracy.
- Add focused tests where they establish numerical, cache, gradient, or resume correctness. Avoid tests that merely repeat implementation details. Filter output to failures and primary traces.
- No new ungated baseline training, FitNet work, DDP, AMP, score redesign, or unrequested publication/push is part of this delivery.

## References

Paths below are relative to this checkout and are development references, not runtime imports.

| Topic | Reference code | Paper |
| --- | --- | --- |
| TGSC | `../TGSC/src/tgsc_kd/{config,tgsc,trainer,modeling}.py`; `../TGSC/model/python/`; experiment JSONs | Existing TGSC implementation/settings supplied by the user |
| DKD | `mdistiller/distillers/DKD.py`; `configs/cifar100/dkd/` | `../DKD.pdf` |
| SimKD | `../SimKD/models/util.py`; `helper/loops.py`; `train_student.py`; `scripts/run_distill.sh` | `../SimKD.pdf` |
| ReviewKD | `../ReviewKD/CIFAR-100/model/reviewkd.py`; training/loss code and `script/reviewKD.sh`; local mdistiller implementation | `../ReviewKD.pdf` |

Paper titles were checked when creating this task list. Detailed method audits are tasks 01-03; availability of a paper does not mean the audit is complete. Use paper equations for the intended method and code/configs for exact released implementation behavior. Record conflicts explicitly rather than choosing silently.

## Task list

### 01. Audit DKD and freeze its reference behavior

- **Scope:** masks, TCKD/NCKD reductions, temperature scaling, loss warmup, CE, and effective pair coefficients. No functional code changes.
- **References:** local official DKD code/configs and DKD PDF loss derivation/experimental settings.
- **Deliverable:** a short DKD reference section in `docs/TGSC_REFERENCE_AUDIT.md`, including source revision and equation/config pointers.
- **Check:** trace one homogeneous and one heterogeneous pair through config inheritance. Identify exact scalar operations that must become per-sample operations.
- **Done when:** the baseline loss contract and pair settings are unambiguous; discrepancies are listed.
- **Depends on:** none.

### 02. Audit official SimKD

- **Scope:** final feature selection, bottleneck, factor, spatial alignment, feature loss coefficient, classifier reuse, train/eval behavior, and required inference artifacts.
- **References:** official clone files listed above and SimKD PDF architecture/objective/experimental settings.
- **Deliverable:** SimKD section in the reference audit, including how official feature lists map to mdistiller's dictionary.
- **Check:** distinguish the official `SimKD` class from the unrelated `Proj` helper; trace the official example's active losses through its command and loop.
- **Done when:** projector, loss, head extraction, and inference contracts are documented without adding optional CE/KL variants.
- **Depends on:** none.

### 03. Audit official ReviewKD

- **Scope:** ABF, residual feedback, stage order, pre/post-activation choices, pooled feature, HCL pooling coefficients, warmup, and pair weights.
- **References:** official ReviewKD CIFAR source and PDF; compare against mdistiller's adaptation.
- **Deliverable:** ReviewKD section in the reference audit with necessary dynamic-size substitutions distinguished from method changes.
- **Check:** compare one homogeneous and one heterogeneous mapping; locate fixed-size assumptions and confirm loss reduction order.
- **Done when:** each initial pair has a justified mapping or an explicit unresolved compatibility issue.
- **Depends on:** none.

### 04. Establish self-contained TGSC configuration and packaging

- **Scope:** package location, config namespace, explicit paths, lightweight dependencies, provenance, and no dependency on sibling folders.
- **References:** existing TGSC dataclasses/JSONs and mdistiller YACS config/setup.
- **Deliverable:** package/config scaffolding and selected TGSC pair settings; auxiliary requirements without blindly downgrading platform torch.
- **Check:** import from an isolated copy/install of this checkout with reference folders unavailable; verify effective config reproduces original JSON values. Do not guess settings for missing pair configs.
- **Done when:** config round-trip works, default augmentation is 0.5/2, and runtime imports resolve entirely inside the clone or declared dependencies.
- **Depends on:** 01-03 for agreed method config contracts.

### 05. Port existing TGSC model compatibility

- **Scope:** carry over verified adaptive pooling/model changes; explicit feature/head metadata. Avoid importing TGSC model copies at runtime.
- **References:** corresponding TGSC and mdistiller model files; method audit mappings.
- **Deliverable:** variable-size-compatible mdistiller models and a per-pair feature-shape report.
- **Check:** supported architectures at 16/20/24/28/32; logits/classifier dimensions; teacher checkpoint loading; 32x32 parity where operations are equivalent. Inspect internal ShuffleNet changes separately.
- **Done when:** declared pairs pass shape checks or fail clearly with a precise unsupported case. Do not silently raise crop sizes or replace official alignment mechanisms.
- **Depends on:** 04.

### 06. Port TGSC crop generation and expose all metrics

- **Scope:** current CAM/saliency proposals, candidate selection/ranking, raw r1/r2/r3, stored q, crop/source identity, and default training augmentation.
- **References:** TGSC `tgsc.py`, `config.py`, and trainer gate conversion.
- **Deliverable:** internal TGSC preprocessing/data module with stable sample metadata.
- **Check:** compare reference versus port on a small fixed source batch with the same model/seed: candidate coordinates, selected pixels, labels, components, and q. Preserve existing composite clipping location and ranking behavior.
- **Done when:** numerical/selection parity holds within stated tolerances; teacher CAM gradients work; no duplicate mdistiller augmentation is applied.
- **Depends on:** 04-05.

### 07. Persist and reuse teacher-specific crop banks

- **Scope:** cache schema/manifest, checkpoint hash, per-resolution files, separate read/write roots, atomic saves, and reuse of completed entries.
- **References:** existing TGSC cache implementation and plan section 5.
- **Deliverable:** prepare-once/load-many cache API with clear legacy-cache errors and storage estimates.
- **Check:** save/load crop and metric parity; altered teacher hash invalidates cache; two methods/students reuse one bank; read-only cache loading performs no writes; incomplete preparation builds only missing compatible entries.
- **Done when:** cached crops reproduce preprocessing output without rerunning the teacher oracle, and components cannot be silently fabricated from old q-only files.
- **Depends on:** 06.

### 08. Implement per-sample TGSC gate helpers

- **Scope:** shared detached gates, unity/disabled diagnostic mode, per-sample reductions, and metadata validation.
- **References:** current TGSC gate and audited method reduction contracts.
- **Deliverable:** one shared gate implementation usable by present/future methods.
- **Check:** hand-computed mixed-quality examples, gate threshold boundaries, detached gradients, zero/unity behavior, and no batch-mean multiplication or sum-of-weights normalization.
- **Done when:** checks establish correct sample weighting and preserve original method scaling with unity gates. Diagnostic modes are correctness tools, not added baseline experiments.
- **Depends on:** 04 and metric contract from 06.

### 09. Integrate DKD + TGSC

- **Scope:** expose per-sample TCKD/NCKD; unified and decoupled gate options; preserve method loss warmup and CE.
- **References:** task 01 audit and DKD PDF/code whenever a formula changes shape/reduction.
- **Deliverable:** gated DKD implementation and initial validated config.
- **Check:** fixed-input original-versus-unity-gated scalar loss/gradient parity; independently zero TCKD/NCKD gates; verify raw component routing for decoupling and correct epoch ramp.
- **Done when:** both gate modes obey the specified formulas without an added objective term.
- **Depends on:** 05, 08.

### 10. Integrate SimKD + TGSC and its inference model

- **Scope:** port official projector, retain official pooling/MSE, freeze/reuse teacher head, and weight feature loss per sample.
- **References:** task 02 audit, official projector/loop, and SimKD PDF.
- **Deliverable:** registered SimKD distiller, learnable-parameter registration, correct eval graph, export reconstruction contract, and initial config.
- **Check:** identical projector weights/tensors give official-port output/loss/gradient parity; teacher/head remain frozen; projected student gets gradients; eval runs without teacher feature extraction. Check student classifier is not accidentally used.
- **Done when:** prediction path is backbone -> projector -> pool -> frozen head, including exported artifacts.
- **Depends on:** 05, 08.

### 11. Integrate ReviewKD + TGSC

- **Scope:** runtime spatial targets, official stage mapping/fusion/HCL, per-sample gating, and original loss ramp.
- **References:** task 03 audit, official ReviewKD source/PDF, and selected mdistiller pair config.
- **Deliverable:** dynamic-size ReviewKD and initial configs without added KL.
- **Check:** at 32x32 and identical weights compare reference outputs/HCL/gradients; check every curriculum size; assert full stage counts/channels; check HCL coefficients at small grids.
- **Done when:** original method behavior is retained where comparable and current TGSC stages run without silent stage truncation or a new crop floor.
- **Depends on:** 05, 08.

### 12. Integrate curriculum training and checkpoint recovery

- **Scope:** shared batch metadata transport, active-resolution loader switching, actual TGSC pair schedule, logging, optimizer registration, resumable state, and method-aware exports.
- **References:** existing TGSC trainer and mdistiller trainer/checkpoint behavior.
- **Deliverable:** reusable trainer support for all three methods; local logs, resolved config, latest/best/periodic checkpoints.
- **Check:** short training sequence crossing a crop boundary; verify optimizer/adapters persist; compare uninterrupted versus epoch-boundary resumed execution under controlled RNG settings; load inference exports. Do not promise bitwise equality with nondeterministic kernels.
- **Done when:** each method can train, stop, resume at the correct stage, and evaluate its export. SimKD export requires no teacher-backbone weights.
- **Depends on:** 07, 09-11.

### 13. Expose Python runner and GPU selection

- **Scope:** `check`, `prepare`, `train`, `evaluate`; config/path overrides; one selected GPU per process; main guard; subprocess-safe initialization and errors.
- **References:** plan section 11; existing entry points and installed torch environment.
- **Deliverable:** `tools/run_tgsc.py` and help/examples using the same runtime as notebooks.
- **Check:** argument/path errors; selecting a valid/invalid visible GPU; inherited visibility behavior; cache miss instructions; explicit resume from a read-only source into a writable destination. Never silently select every GPU or fall back to CPU training.
- **Done when:** one local command can prepare/train/resume/evaluate a declared pair with clear paths and GPU selection. CPU may support checks; real GPU checks are separately recorded.
- **Depends on:** 12; preparation/check portions can be built after 07.

### 14. Package Kaggle notebooks and instructions

- **Scope:** one preparation notebook plus one each for DKD, SimKD, ReviewKD; shared runner invocation; attached artifacts and output persistence instructions.
- **References:** plan section 11 and current official Kaggle documentation when platform behavior needs verification.
- **Deliverable:** four thin notebooks, `docs/TGSC_RUNNING.md`, documented tested dependency setup, and ignored generated artifacts.
- **Check:** parse notebook JSON; verify configuration cells produce correct runner arguments; run setup/preflight with reference folders unavailable; exercise a Kaggle-style read-only input/writable output layout. No embedded credentials, hard-coded personal paths, or duplicated training code.
- **Done when:** instructions cover clone/revision, data/teacher/cache paths, GPU, prepare/reuse, train, resume, and evaluate for each method.
- **Depends on:** 13.

### 15. End-to-end GPU acceptance and pair expansion

- **Scope:** a tiny real-data prepare -> train -> save -> resume -> export -> evaluate cycle per method, then declared pair coverage. This is a correctness pilot, not a full accuracy experiment.
- **References:** method audit and reproducible commands from task 14.
- **Deliverable:** validation record with code/config/cache identifiers, environment/GPU, tested pairs, concise results, and explicit pending checks.
- **Check:** one homogeneous and one heterogeneous pair first; verify cache reuse across methods and independent runs on different GPUs when hardware exists. Actual Kaggle execution is required before claiming Kaggle-tested status.
- **Done when:** the declared matrix passes relevant shape/numerical checks and each method completes a real GPU cycle. If hardware is unavailable, mark the GPU portion pending and state which offline checks passed; do not substitute notebook parsing for execution.
- **Depends on:** 14.

## Checkpoint record template

For each completed task, append a short entry to `docs/TGSC_IMPLEMENTATION_PROGRESS.md`:

- Task ID and status: complete / partial / pending.
- Changed files and purpose.
- Reference file/revision and paper section/equation used; any discrepancy and its resolution.
- Exact focused check command, environment, and concise outcome.
- Remaining limitations or downstream dependency.

Do not create empty progress claims before work is done. Recheck only affected behavior after subsequent changes, broadening checks when new failures or concerns justify it.

## Suggested implementation batches

| Batch | Tasks | Reviewable outcome |
| --- | --- | --- |
| A | 01-03 | Reference contracts established |
| B | 04-05 | Self-contained config/models accept curriculum sizes |
| C | 06-08 | Reusable crop bank and correct shared weighting |
| D | 09-11 | Each official method has verified TGSC loss/inference support |
| E | 12-13 | Local runnable training/resume/evaluation |
| F | 14-15 | Kaggle notebooks and verified delivery |

Tasks are sequential review checkpoints, not automatic user-approval gates. Once implementation is authorized, continue through routine verified steps and report meaningful progress. Ask only when a reference conflict or missing experiment setting requires a scientific decision that cannot be inferred from the agreed plan.
