# TGSC integration into mdistiller

Updated: 2026-10-21. Status: implementation active; see the linked progress report for delivered code and validation.

Implementation work is divided into independently checkable tasks in [TGSC_IMPLEMENTATION_TASKS.md](TGSC_IMPLEMENTATION_TASKS.md). All implementation changes belong inside this mdistiller checkout. Sibling repositories and attached PDFs are reference material only.

For current status, accepted decisions, verification history, and the next action after context compaction, read [docs/TGSC_IMPLEMENTATION_PROGRESS.md](docs/TGSC_IMPLEMENTATION_PROGRESS.md) first.

## 1. Agreed scope

Integrate the existing TGSC framework with official distillation methods, beginning with DKD, SimKD, and ReviewKD. FitNet is deferred by the user's latest decision; its earlier research notes below are retained for future reference and do not block implementation. Design the shared pipeline so more methods can be added later.

The user's latest decisions replace earlier proposals:

- Preserve TGSC's default augmentation: `flip_prob=0.5`, `aug_padding=2`. Do not add a second mdistiller augmentation pipeline on top.
- Preserve current TGSC score formulas, ranking, clipping locations, selection settings, stage schedule, sample multiplicity, gate threshold, and training settings. Do not substitute newly proposed settings silently.
- A confidently misclassified crop can still contain useful teacher information. Preserve the current top-two margin score; do not replace it with a ground-truth-margin score or filter crops solely for teacher misclassification.
- Use each official method's losses with TGSC weighting on its distillation terms. Do not add CE, KL, new adapters, or other losses that would create a different method.
- Preserve each official method's feature alignment mechanism. FitNet is outside the first implementation; its two-stage procedure is a future integration.
- No new training runs without TGSC are in the requested experiment scope. Use available baseline results, recording their source and settings.
- Reuse TGSC's existing support for variable input sizes and persistent caching. Extend it where needed rather than designing another crop framework.
- Store r1/r2/r3 as well as q to support DKD's separate gates.
- Prepare crops and oracle information once per compatible teacher/cache configuration; reuse across distillers and students.

Earlier suggestions to bound each score component, switch all runs to batch 64, change LR milestones, force every method to start at 20, remove selected crops, run new ungated baselines, or test hybrid losses are withdrawn from this implementation scope.

## 2. What the local code actually does

References available locally:

| Reference | Location / inspected revision |
| --- | --- |
| Official DKD and shared training framework | `mdistiller`, revision `a08d46f10d6102bd6e3f258ca5ac880b020ea259` |
| Official SimKD clone | `../SimKD`, revision `2b389c31ed7779aea31e7aaf0bb0f2d8b6ac2f01` |
| Official ReviewKD clone | `../ReviewKD`, revision `cede6ea6387ae9b6127de0e561507177bf19c11e` |
| Original FitNet paper | `../fitnet.pdf`, especially sections 2.2-2.3 and Algorithm 1 |
| DKD paper | `../DKD.pdf`, Decoupled Knowledge Distillation |
| SimKD paper | `../SimKD.pdf`, Knowledge Distillation with the Reused Teacher Classifier |
| ReviewKD paper | `../ReviewKD.pdf`, Distilling Knowledge via Knowledge Review |
| Current TGSC code | `../TGSC/src/tgsc_kd/` and `../TGSC/model/python/` |

The references are available, but full method-by-method numerical comparisons have not yet been performed.

### Existing augmentation

`TGSCPreprocessor.get_loader()` currently supports `RandomCrop` with reflected padding and `RandomHorizontalFlip`. Inspected configs set `flip_prob=0.5` and `aug_padding=2`. The user's latest decision is to retain these defaults.

Keep the existing order: load the scored base crop, apply TGSC's padding/random crop and horizontal flip, and use that same resulting image for teacher and student training forwards. Preserve tensor conversion/normalization and dataset shuffling. Do not also apply mdistiller's separate random image transforms. Cached q/r1/r2/r3 describe the base crop, so their use after augmentation remains the current TGSC approximation; preserving defaults does not make those scores exact for every augmented view. Do not change scoring or recompute it online without a separate decision.

### Existing variable-size support

TGSC's own ResNet, WRN, MobileNetV2, and ShuffleNet variants already use adaptive final average pooling; VGG and ResNetV2 also have adaptive final pooling. TGSC ShuffleNetV1 additionally defines `AdaptiveAvgPoolStride2` for an internal pooling path.

The unmodified mdistiller copies still contain fixed final pooling in several networks. Therefore this is primarily a task of carrying over and verifying the existing TGSC model adaptations. Inspect model differences carefully rather than replacing every pooling operation indiscriminately.

### Existing cache support

`TGSCPreprocessor` already supports in-memory caches and optional disk saves/loads of per-resolution `CropTensorDataset` objects. These include crop tensors, labels, q, original source indices, and selection flags. The ResNet32x4 and VGG JSON configs disable disk caching; the WRN40-2 -> WRN40-1 config already enables it. Full-image teacher logits/union maps are currently retained in memory by the preprocessor.

Reuse this design. Enable persistent preprocessing for cross-run reuse and extend its saved information and identity checks. Existing q-only files cannot provide the separate DKD components.

## 3. The two size problems, explained separately

### A. Final pooling: already addressed in TGSC models

A model turns an image into an internal grid. At a 32x32 input, that grid might be 8x8. With a 16x16 crop it might be 4x4. A fixed 8x8 pooling window no longer fits that 4x4 grid.

Adaptive global pooling means: average whatever grid exists down to one value per channel. It handles both 8x8 and 4x4 without changing the classifier's required vector length.

Action: port the relevant TGSC model changes into mdistiller and verify supported resolutions and checkpoint loading. Compare original and adapted outputs at 32x32 where the pooling is mathematically equivalent. Other TGSC changes, such as an internal ShuffleNet pooling replacement, need their own behavior checks; do not assume every modification is numerically identical.

### B. Intermediate feature matching: still needs method-specific checks

Final pooling does not fix intermediate features. For example, a student may produce `[B,64,10,10]` while the selected teacher feature is `[B,128,5,5]`. Direct MSE cannot compare these tensors. A 1x1 convolution can turn 64 channels into 128, but the student's grid remains 10x10.

This is already a concern for official feature-distillation methods even without TGSC. Their adapters, feature selectors, or pooling rules handle it. Our task is to retain those rules and ensure they still work as crop sizes change.

- DKD compares class logits and has no intermediate-feature alignment requirement.
- SimKD already pools to the smaller feature grid and projects channels. Preserve that behavior.
- FitNet uses a learned regressor. Cross-check its selected features and regressor against the reference instead of automatically replacing it with a new 1x1 adapter.
- ReviewKD has fusion/alignment modules, but mdistiller uses fixed spatial targets. Make the size references depend on the current forward-pass tensors while retaining the official fusion and loss operations.

Feature shapes above are illustrative, not measured values for a particular pair. Log actual shapes for each supported pair and resolution. Some pairs already align; do not add resizing where none is needed.

## 4. Preserve TGSC behavior

Use the selected existing TGSC experiment config as the authoritative source of its parameters. Persist the effective settings with each result.

The inspected TGSC JSONs use batch 128. ResNet32x4 and VGG use LR milestones `[48,144,192]`, while WRN40-2 -> WRN40-1 uses `[48,144,192,210]`. The configured stage sequence is `(16,48), (20,48), (24,48), (28,48), (32,48)` where each pair is `(crop_size, duration_epochs)`. Preserve the relevant config rather than replacing it with mdistiller's batch-64 and `[150,180,210]` profile. Verify each pair's actual config before generating its run configuration.

The earlier 20x20 ReviewKD floor was a proposed change. The latest instruction to preserve TGSC settings takes precedence: validate ReviewKD on the current schedule first. If a size is unsupported, identify the exact failing operation and resolve compatibility; do not silently raise the floor or alter stage durations.

### Score definitions

Retain the current definitions and temperature:

- r1: selected target-class teacher probability on crop divided by its full-image probability, with the existing epsilon.
- r2: `1 - min(KL(p_full || p_crop)/log(100), 1)`.
- r3: crop top-one/top-two probability gap divided by full-image top-one/top-two gap, with the existing epsilon.
- Candidate q: arithmetic mean of the three raw components.
- Existing dataset storage clamps the composite score to `[0,1]` after candidate ranking. Preserve that order.
- Oracle probabilities currently use softmax temperature 1; this is distinct from a method's distillation temperature.
- The target-class selection follows existing `use_gt_target` behavior.

Do not clamp r1/r3 individually, change KL direction, replace the top-two margin, or alter ranking. Save raw component values alongside the existing stored q. For DKD decoupled gates, apply the existing gate to the stated component combinations; do not reinterpret those components.

A large margin on a wrong prediction remains eligible. It can carry teacher confidence and class relationships useful to distillation; a large margin alone does not prove that a crop contains useful dark knowledge or fully describes the decision boundary. This qualification does not change the requested scoring policy.

### Gates and reduction

Keep `w(q)=min(1,q/0.75)` for current nonnegative scores. Detach oracle metadata from gradients.

Compute each sample's distillation loss first, multiply it by its own gate, then average: `mean_i(w_i * loss_i)`. Do not multiply two batch means or divide by the sum of gates. Preserve each method's native channel/spatial reduction and coefficients.

CE remains ungated when it belongs to the official method's objective. This does not introduce CE into methods whose official recipe omits it. Keep existing crop multiplicity and source exposure; do not silently sample fewer crops.

## 5. Preprocess once and reuse

### What can be shared

CIFAR-100 decoding/basic preprocessing can be shared when preprocessing settings match. Teacher-guided crop locations and quality scores depend on the teacher. The safe unit of expensive TGSC preprocessing is:

**dataset + teacher checkpoint + TGSC preprocessing/scoring settings + required crop sizes**.

For one teacher checkpoint, prepare the full crop bank once and reuse it for DKD, SimKD, FitNet, ReviewKD, future methods, and different students. Different teachers generally require different crop banks and scores, even though CIFAR-100 is the same. Do not reuse resnet32x4-selected crops as if they were VGG-selected crops.

### Preparation workflow

1. Load CIFAR-100 in a stable source order using TGSC's existing normalization.
2. Load the chosen teacher using the appropriate TGSC variable-size adaptations.
3. Build full-image teacher reference outputs and CAM/saliency union maps using the current algorithm. Persist these if useful for later resumable preprocessing.
4. For each existing curriculum resolution, generate/select candidates exactly as TGSC currently does.
5. Save selected crop tensors, labels, source indices, crop identifiers/boxes, selection flags, raw r1/r2/r3, and the current stored q.
6. Write a versioned manifest and verify that all requested resolutions are present. A later run can build only missing entries.
7. All compatible training runs load these base-crop files and apply the existing TGSC default augmentation during training. Do not cache random augmented outputs in place of the reusable base crop bank.

Keep tensors numerically consistent with current TGSC; do not save lossy JPEGs or switch cache precision silently. Start with the existing per-resolution tensor files; consider sharding only if storage/RAM measurements justify it.

### Cache identity and validation

The existing key records teacher class and preprocessing parameters but not a teacher checkpoint content hash. Strengthen it with:

- Teacher architecture and checkpoint hash.
- Dataset split/order identity and normalization.
- TGSC proposal/selection/scoring settings and code/schema version.
- Crop resolution and stable source/crop identity.

The method name and student architecture should not invalidate a genuinely method-independent oracle cache. Check record counts, tensor shapes, finite metrics, labels, source alignment, and manifest compatibility. Keep raw components unbounded where current TGSC produces them; range checks must respect existing semantics.

Old q-only caches must be regenerated or rescored for decoupled DKD. Do not invent r1/r2/r3 from q. Use atomic writes and avoid multiple trainers overwriting a shared cache. Offline preparation belongs in a separate process/command, not inside dataloader workers.

Caching crops and oracle scores does not remove all teacher computation during training. Feature methods still need teacher features on the crop, and logit methods need teacher targets. Initially use online frozen teacher forwards for these targets; caching dense features is a separate storage/design decision. Persisted oracle outputs must not be mistaken for complete feature-distillation targets.

## 6. Official method integration

### Reference audit

Compare local references before implementing TGSC changes: native objectives, coefficients, warmups, feature selection, adapters, teacher behavior, inference, and checkpoint contents. Use fixed-tensor forward/gradient checks where useful. This does not require new full baseline training.

An official method's own adapters and constituent losses are part of that method. Preserve them. Additional independently proposed KL, CE, or feature losses are excluded.

### DKD

mdistiller is the official reference. Preserve TCKD/NCKD, masks, temperature scaling, pair coefficients, and its distillation-loss warmup.

- Unified TGSC gate: CE plus `warmup * mean(w(q)*(alpha*TCKD + beta*NCKD))`.
- Requested decoupled TGSC gate: CE plus `warmup * mean(w((r1+r3)/2)*alpha*TCKD + w(r2)*beta*NCKD)`.

These are TGSC weighting policies on existing DKD components, not extra loss terms. Keep separate config identities. Their scores require the extended cache.

Use effective official pair configs: for example the inspected WRN -> ShuffleV1 config inherits beta 8 rather than the earlier attached blanket WRN beta 6. Do not assign coefficients solely by teacher family.

### SimKD

Port the official SimKD class from `../SimKD/models/util.py`, retaining its bottleneck, BN/ReLU, pooling alignment, MSE, factor, and teacher-classifier reuse. The official example uses feature loss weight 1 and CE/KL weights zero. Exclude earlier proposed CE options and weight-10 variants from this scope.

Apply TGSC to per-sample official feature loss. Inference must be student backbone -> projector -> pooled vector -> frozen teacher classifier. Save this complete graph; do not use mdistiller's generic original-student-head inference or student-only export. No teacher-backbone forward should be needed during inference.

### ReviewKD

Use `../ReviewKD/CIFAR-100/` to cross-check mdistiller's ABF, residual feedback, HCL, features, coefficients, and loss warmup. Preserve the official objective and only apply TGSC weighting to its distillation term.

Replace static spatial sizes with actual matched feature sizes where needed for the existing curriculum. Validate stage counts/channels explicitly, including pooled features. Do not silently discard features through a truncated `zip`. Native deep 1x1/pooled features are not automatically errors.

The inspected mdistiller resnet32x4 -> resnet8x4 YAML sets review weight 5, while its global default is 1. Use verified pair settings; no extra classical KD or newly invented per-stage weighting.

### FitNet

**Deferred.** The following notes are historical planning material. No FitNet implementation, notebook, dependency, or epoch-budget decision is required for the current DKD/SimKD/ReviewKD deliverable.

The supplied paper describes hint pretraining followed by whole-network training with label and softened-teacher supervision. mdistiller's FitNet is a single-stage CE-plus-hint adaptation. These are distinct recipes, not evidence that the paper omits teacher-logit supervision.

The user has selected the original paper's two-stage procedure: first train the student's early layers and regressor with hint supervision, then initialize from those learned student parameters and train the whole student using labels and softened teacher predictions. Do not retain hint loss in stage two merely because mdistiller does so in its separate adaptation. Its stage durations and application of the unchanged TGSC curriculum across those two steps still need a concrete specification. Do not silently implement the earlier simultaneous CE+KL+hint hybrid or label mdistiller's adaptation an exact paper reproduction.

The inspected mdistiller implementation and README do not explain why its FitNet adaptation omits the original two-stage procedure. A simpler common training loop is a possible explanation, not a verified author rationale. Existing mdistiller FitNet baseline scores describe that adaptation and should not be presented as an exact matched baseline for the two-stage recipe.

Use the paper and original code link it supplies (`https://github.com/adri-romsor/FitNets`) to settle the regressor, hint/guided layer correspondence, and training schedule. Preserve the official selected recipe's components and apply TGSC weighting to its teacher-supervision terms. No new adapter is introduced merely because a 1x1 convolution seems convenient.

#### Current decision: match FitNet training settings; verify phase lengths

The user now prefers the original paper's training settings for FitNet. The earlier 240 + 240 proposal is superseded. This is an explicit FitNet-specific exception to keeping TGSC's standalone optimizer/temperature settings; TGSC crop selection, scores, caching, and default augmentation remain unchanged.

| Phase | Trainable student parts | Objective | Duration |
| --- | --- | --- | --- |
| 1: hint pretraining | Layers through the guided layer plus official regressor | TGSC-gated official hint loss; no classification loss | Exact original selected count not yet established |
| 2: whole-network distillation | Entire student, initialized with phase-one learned student weights | Ground-truth CE plus TGSC-gated official softened-teacher loss; no hint loss | Official CIFAR-10 all-data config specifies 288; CIFAR-100 config not yet retrieved |

Paper settings (Appendix A.1.2): RMSProp, initial LR 0.005, batch 128, softened-target temperature 3, teacher-loss coefficient 4 decreasing linearly to 1 over 500 epochs. CIFAR-10 uses 40k/10k training/validation, stopping after 100 epochs without improvement with a cap of 500 per phase; selected durations are then used for CIFAR-100. The paper passage does not state those selected durations. Up to 1000 epochs total is an upper-bound compute estimate, not a reported actual run length.

Code evidence: the official `yaml/cifar10_fitnet19_all.yaml` sets `max_epochs: 288`; `scripts/fitnets_training.py` accepts a separate `--hints_epochs` argument for the first phase. Therefore 288 is not evidence of a 288 + 288 historical run. The inspected CIFAR-10 config also specifies teacher decay start 5 / saturation 400, differing from the paper's stated 500-epoch decay. Record this paper/code discrepancy and do not silently mix recipes. No original per-phase run logs or CIFAR-100 epoch totals have been confirmed in this review.

Sources: https://github.com/adri-romsor/FitNets/blob/master/yaml/cifar10_fitnet19_all.yaml and https://github.com/adri-romsor/FitNets/blob/master/scripts/fitnets_training.py . The CIFAR-100 YAML is listed in the official repository, but its contents could not be retrieved during this check.

Recommended transition: retain all student weights, remove the auxiliary regressor from the active graph/optimizer, enable the entire student for training, and initialize a fresh optimizer using the selected FitNet reference settings. Do not attach TGSC's standalone SGD milestones automatically to RMSProp. Checkpoint phase index, phase-local epoch, global epoch, optimizer/scheduler state, RNG, and cache identity. In phase one, avoid running/updating unused student layers and their BN buffers; the teacher remains frozen.

Reuse exactly the same per-teacher base crop bank and oracle metadata for both phases. Apply default TGSC augmentation independently on each draw, and teacher/student receive the same augmented tensor. Additional epochs do not require generating new crops.

Repeat the crop progression for both phases in principle, but do not finalize its epoch mapping until phase lengths are chosen. Matching a non-240 duration and keeping exactly five 48-epoch crop stages cannot both hold for every phase. Any extension at 32x32 or proportional change in crop-stage durations must be an explicit integration decision. If original epoch counts remain unavailable, use the documented stopping procedure or a clearly labeled estimate; do not invent historical counts.

## 7. Shared integration structure

TGSC remains separate from method-specific mathematics:

- Dataset/oracle service: existing crop generation, persistent crop bank, metrics, metadata.
- Curriculum service: current stage settings, loader transitions, resume state.
- Shared gate helpers: detached per-sample weights and correct reduction.
- Each distiller: official architecture/losses, per-sample decomposition, inference requirements.
- Trainer: metadata transfer, unchanged TGSC protocol, method warmup, logging, checkpoint/export.

Future methods reuse these services. Batch-dependent relational/contrastive losses require an explicit gating definition when added; a gate on a batch-average scalar is not a universal solution.

Keep ordinary mdistiller paths compatible. Do not introduce extra losses through diagnostic dictionary entries: the trainer sums losses. Instantiate adapters/projectors before optimizer creation and retain them across crop changes. Initial support targets the base trainer; DOT requires a separate gradient-flow compatibility check.

Log the full effective TGSC/method config, reference revisions, cache manifest, current resolution, raw/gated losses, and gate statistics. Resume must restore optimizer, LR, epoch/curriculum, RNG/sampler, and cache identity. Teacher CAM creation can need gradients; ordinary distillation teacher forwards remain frozen and under no-grad.

## 8. Work order and acceptance checks

1. Finish official reference comparisons for DKD, SimKD, and ReviewKD. Defer FitNet.
2. Document the exact existing TGSC config for each teacher/student run, retaining its default augmentation.
3. Port and verify the existing TGSC variable-size model support; inspect each method's intermediate feature alignment.
4. Extend persistent preprocessing/cache schema, add teacher checkpoint identity, and save all metric components without changing scoring.
5. Add the prepare-once/load-many workflow for all current resolutions per teacher.
6. Integrate per-crop metadata, gates, curriculum switching, logging, and resume.
7. Integrate DKD, SimKD, and ReviewKD with their official loss behavior and inference exports.
8. Run focused checks and TGSC training smoke runs; then launch the requested TGSC-only experiment matrix.

Required checks:

- Supported crop sizes produce valid logits/features and finite losses.
- Checkpoints load; intended pooling changes behave as verified at 32x32.
- Unity gates preserve reference loss/gradient behavior on fixed tensors; zero/mixed gates affect the intended samples/terms only. These are correctness checks, not ungated baseline training runs.
- r1/r2/r3 and stored q match existing TGSC scoring on replayed base crops; only TGSC's default post-score transforms run, without duplicate mdistiller augmentation.
- Different students/methods load the same compatible teacher crop bank without reprocessing; incompatible checkpoints invalidate it.
- SimKD deployment predictions match distiller evaluation without teacher feature extraction.
- Crop-stage transitions preserve learned modules/optimizer and work after resume.
- Cache preparation is not accidentally repeated by each training process or worker.

See the progress report for completed implementation checks and synthetic CPU train/resume tests; real-data GPU/Kaggle execution is pending.

## 9. Evaluation scope

Train the requested official methods with TGSC. Use available published/local baseline scores rather than launching fresh runs without TGSC. Record each baseline's provenance, architecture/checkpoint, preprocessing, optimizer/schedule, seed/reporting convention where available, and label protocol differences. Published values can be useful references even when they do not isolate TGSC's causal contribution under identical settings.

Retain current crop multiplicity and training budget. Report source draws, actual crop count, optimizer steps, resolution schedule, seed results, accuracy, training time, preprocessing time, cache size, and deployed model costs. Test on full 32x32 images with the actual method inference graph. No test data enters teacher-guided crop construction.

FitNet's two-stage curriculum specification is deferred. Current outstanding work is reference verification, engineering compatibility, and the runnable Kaggle/local workflow specified below.

## 10. Reference links

- Official mdistiller/DKD: https://github.com/megvii-research/mdistiller
- Official SimKD: https://github.com/DefangChen/SimKD
- Official ReviewKD: https://github.com/JIA-Lab-research/ReviewKD
- FitNet paper: `../fitnet.pdf`; original code linked by the paper: https://github.com/adri-romsor/FitNets

Local source paths above are authoritative for this review; external links identify provenance and were not newly audited in this revision.

## 11. Kaggle and local GPU delivery plan

Status: first runnable workflow implemented; offline validation passed, actual GPU/Kaggle acceptance pending. This section defines the delivery target for DKD, SimKD, and ReviewKD only. It does not authorize pushing, publishing notebooks, or running long training jobs from this session.

### 11.1 One clone must contain the runtime

The workspace root is not a Git repository. `mdistiller/` is a separate Git checkout; TGSC, SimKD, and ReviewKD currently live beside it. Its configured origin currently points to upstream Megvii.

Delivery assumption: the user will push the modified mdistiller checkout to their own repository and clone that repository on Kaggle. Make that checkout self-contained:

- Move/adapt the needed TGSC runtime into the mdistiller package, preserving its current behavior and recording provenance.
- Include the official SimKD projector and required ReviewKD adaptations with attribution/license compliance.
- Carry over required TGSC model changes into mdistiller's model package.
- Do not depend on `../TGSC`, `../SimKD`, `../ReviewKD`, absolute Windows paths, symlinks to sibling folders, or editable installs outside the clone.
- Do not commit datasets, teacher checkpoints, crop caches, trained weights, tokens, or notebook execution outputs. Add appropriate ignore patterns for new artifact paths.
- Do not change remotes or push automatically. Notebook clone configuration must let the user supply their repository URL and revision.

### 11.2 Shared runner, thin notebooks

Proposed deliverables:

| File | Purpose |
| --- | --- |
| `tools/run_tgsc.py` | Shared Python entry point for environment checks, cache preparation, training, and evaluation |
| `notebooks/tgsc_prepare_cifar100.ipynb` | Prepare a reusable crop bank once per teacher checkpoint |
| `notebooks/tgsc_dkd_cifar100.ipynb` | Run official DKD + TGSC; configurable unified/decoupled gates |
| `notebooks/tgsc_simkd_cifar100.ipynb` | Run official SimKD + TGSC |
| `notebooks/tgsc_reviewkd_cifar100.ipynb` | Run official ReviewKD + TGSC |
| `configs/tgsc/cifar100/` | Explicit validated pair/method configs preserving the selected TGSC settings |
| `requirements-tgsc.txt` and setup guide | Auxiliary dependencies and supported environment instructions |
| `docs/TGSC_RUNNING.md` | Kaggle/local instructions, cache reuse, GPU selection, resume, expected artifacts |

The notebooks, runner, running guide and requirements now exist. Run profiles use JSON files named `<method>_<teacher>_to_<student>.json` under `configs/tgsc/cifar100/`; YAML paths below are historical proposed examples, superseded by the running guide. Keep existing `tools/train.py` behavior compatible. All new notebooks invoke the shared runner rather than duplicate training, gate, or preprocessing code.

Notebook launch should use the active kernel's Python (`sys.executable`) in a subprocess with an argument list. This isolates GPU setup from CUDA already initialized in a notebook cell and avoids shell quoting issues. Stream training output to the notebook and save it to the run directory; propagate subprocess failures.

### 11.3 Notebook layout

Each training notebook will contain:

1. Short setup instructions: enable a GPU; enable Internet when cloning/installing/downloading; identify offline input alternatives.
2. Clone/check out the user's selected repository revision into a writable directory and install the package plus lightweight dependencies.
3. One clearly labeled settings cell: config/pair, GPU, seed, dataset root, teacher checkpoint, cache location, output directory, and optional resume checkpoint. DKD also exposes its gate mode.
4. Preflight: validate CUDA, actual GPU selection, data/checkpoint/cache paths, effective config, and method feature shapes before expensive work.
5. Cache validation: load the prepared cache. If missing/incompatible, stop with the exact preparation command. Optional `--prepare-if-missing` is explicit; repeated preprocessing is not the training default.
6. Train using the common runner, with regular recoverable checkpoints.
7. Evaluate/export using the actual method inference graph; show accuracy, saved model paths, and run summary.

A notebook runs one selected teacher/student pair at a time by default. Changing that settings cell selects another validated pair without editing library code. Future methods receive another thin notebook and config, reusing the same infrastructure.

### 11.4 Preparation and Kaggle persistence

Recommended Kaggle workflow:

1. Attach CIFAR-100 and the chosen teacher checkpoint, or explicitly download them when Internet is available.
2. Run the preparation notebook for the teacher's required resolutions. Reuse completed compatible entries; build missing ones. Write a manifest and completion markers.
3. Save the prepared artifacts as Kaggle notebook output or a user-created Kaggle Dataset.
4. Attach that artifact to each method notebook and point it at the same crop bank.
5. Write training checkpoints/results to the current session's writable output directory. Preserve the notebook output to continue in another session.

Default example locations:

| Artifact | Kaggle | Local |
| --- | --- | --- |
| CIFAR / teacher inputs | User-selected paths under `/kaggle/input/` | User-selected data/checkpoint directories |
| Prepared cache input | Attached artifact under `/kaggle/input/` | Shared cache directory |
| New cache files | `/kaggle/working/tgsc_cache/` | Configurable cache output directory |
| Training output | `/kaggle/working/tgsc_runs/<run-id>/` | Configurable runs directory |

Treat attached inputs as read-only. Support separate cache-read and cache-write roots; a completed attached cache must not need copying or mutation. Missing entries can be written to a writable overlay after explicit preparation. Avoid downloading CIFAR to a read-only path; validate the expected torchvision CIFAR-100 layout and clearly handle supplied archives.

Kaggle sessions do not provide a shared writable filesystem across separate method notebooks. Disk caching within one session alone does not satisfy cross-session reuse; saved/attached artifacts are required. Provide instructions, not automatic uploads or publication. Offline use requires code/dependencies/data/checkpoints to be supplied in advance; a Git clone itself needs network access.

Load only the active resolution's crop tensors into RAM, releasing the preceding resolution where appropriate. Print estimated cache sizes before preparation. Example capacity estimate: exactly three retained crops for each of 50,000 images at all five sizes in FP32 would use about 5.1 GiB for crop pixels alone per teacher; actual counts vary, particularly at full size. Do not change precision, crop count, or score semantics to reduce storage without a separate decision.

### 11.5 Python runner interface

Proposed subcommands and arguments:

- `check`: environment, config, GPU, checkpoint, dataset/cache validation; report problems before a long run.
- `prepare`: build/validate the teacher-specific crop bank for every configured resolution; resumable at completed-file boundaries.
- `train`: load the cache, train one method/pair, checkpoint, and optionally resume.
- `evaluate`: load the method-aware exported model and evaluate full 32x32 CIFAR-100 images.

All relevant commands accept `--config`, explicit data/checkpoint/cache paths, and GPU selection. Training also accepts seed, output directory, and `--resume <checkpoint>`. Save the fully resolved config and command arguments. Reject conflicting method/pair/cache combinations with actionable messages.

Illustrative local training command (planned interface, not executable yet):

`python tools/run_tgsc.py train --config configs/tgsc/cifar100/dkd/res32x4_res8x4.yaml --gpu 1 --teacher-checkpoint /path/teacher.pth --data-root /path/data --cache-root /path/cache --output-dir /path/runs/dkd-seed27`

The same runner command will be constructed by the corresponding notebook. Method-specific defaults live in configs; notebooks should not override scientific parameters invisibly. Required dataset/checkpoint paths can be configured without editing the model registry. Disable W&B by default so no account/login is required; retain local CSV/JSON logs and optional TensorBoard.

### 11.6 Multiple GPUs: first-version scope

Support selecting one GPU per process. On a host with several GPUs, the user can run DKD on GPU 0 and ReviewKD on GPU 1 as independent jobs, using different output directories and the same read-only compatible cache.

- Set GPU visibility before importing torch or initializing CUDA in the child runner.
- State whether `--gpu` indexes devices in the inherited visible-device list. Recommended definition: it selects one index from that list, then the child sees the selected device as `cuda:0`. If no visibility restriction is inherited, it indexes the host's CUDA-visible devices.
- Print the inherited selection, resolved visibility, GPU name, and logical device used. Reject invalid indices; never silently use all GPUs or silently fall back to CPU for training.
- Avoid unconditional `.cuda()`/`.module` assumptions in the TGSC runner path. One-GPU use should not require DataParallel merely for accessing the underlying distiller.
- Cache preparation also selects a GPU; use a completed cache for concurrent trainers and prevent simultaneous writes.
- Support Windows local execution through a proper main guard and worker-safe code.

Training one model across several GPUs with DDP is a separate extension, not implied by selecting a GPU on a multi-GPU host. It changes sampler/reduction/resume requirements and needs dedicated verification. Do not silently change global batch size or scale LR.

### 11.7 Checkpoints and interrupted sessions

Save at least each completed epoch, with atomic replacement of `latest`, plus best and configurable periodic checkpoints. Resume from an explicit file, including a file in an attached read-only Kaggle artifact, while writing new checkpoints to the current writable output directory.

Checkpoint contents: method/pair, student and method modules, optimizer, scheduler/LR state, completed epoch, next curriculum position, RNG/loader generator state, best metric, effective config, cache manifest identity, code revision, and relevant environment versions. No claims of exact mid-batch recovery: first version resumes at the latest completed epoch. Save the state at a point that faithfully continues the next epoch's sampling.

A configurable stop-after-epoch or runtime budget can end gracefully at an epoch boundary after saving. It does not guarantee recovery from an abrupt session termination; regular saved outputs are still necessary.

Export inference artifacts separately from resumable training checkpoints. SimKD exports backbone + projector + frozen teacher head. DKD/ReviewKD exports the inference student, while their resumable checkpoints keep required training adapters. Evaluation must reconstruct each export without sibling repos or a teacher-backbone checkpoint.

### 11.8 Dependencies and environment checks

The current mdistiller requirements pin torch 1.9/torchvision 0.10. Do not install that legacy file blindly into Kaggle's current GPU environment. Retain the platform's compatible torch/torchvision pair when valid; install needed auxiliary packages separately. Define a tested supported environment after smoke checks, and record Python, torch, torchvision, CUDA runtime, GPU, and dependency versions in every run.

Preflight checks should catch incompatible imports/API usage, missing teacher weights, unsupported feature shapes, incomplete caches, accidental read-only output paths, unavailable GPU, and insufficient storage estimates. Do not assume installed CUDA packages prove that a usable GPU exists. FP32 is the initial numerical reference; AMP or performance changes are opt-in and require their own validation.

Kaggle references checked for planning: https://www.kaggle.com/docs/notebooks , https://github.com/Kaggle/docker-python , and https://www.kaggle.com/docs/datasets . Do not hard-code current session-hour or output-size quotas.

### 11.9 Implementation and delivery order

1. Finish the local official-method audits and preserve selected TGSC pair settings.
2. Make the mdistiller runtime self-contained and resolve modern environment compatibility.
3. Implement/verify pooling, alignment, cache schema and prepare-once workflow.
4. Implement the shared runner with explicit GPU selection, resume, exports, and local logs.
5. Integrate DKD, then SimKD and ReviewKD with focused reference checks.
6. Add the preparation notebook and three thin method notebooks; document attached artifact reuse.
7. Verify a clean-clone installation/import and notebook structure; run a small real GPU prepare/train/resume/evaluate cycle for each method when GPU access is available.
8. Validate paths against a Kaggle-style read-only input/writable output layout and actual Kaggle execution when available. Report unavailable GPU/Kaggle checks explicitly; do not mark the notebooks GPU-verified based only on parsing.

Completion means the user can clone one pushed repository, select paths/pair/GPU in a method notebook or Python command, reuse a prepared cache, train, resume, and evaluate the saved model. Implementation is now authorized; use the task/progress documents for current state.
