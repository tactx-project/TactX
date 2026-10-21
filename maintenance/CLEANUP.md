# Release cleanup and validation

Working branch: `refactor/tactx-release-cleanup`.
Base: `f5d66f82af79def636b4ce3d4e5b6bc78da8f311`.

## Scope

The retained workflows are paired-data preprocessing, TactX alignment training, reconstruction/contrastive/L2 objective baselines, static and optional contact/dynamic evaluation, tactile-to-latent HDF5 conversion, and offline ACT policy training.

The [path manifest](cleanup_manifest.json) accounts for **274 removed paths** and **39 renamed paths** from the base tree. Old path names appear in that historical manifest only; they are not import aliases or supported entry points. Original files remain recoverable from the base commit. This branch does not rewrite Git history.

- Training entry points are consolidated into `scripts/train_alignment.py`; `--alignment_loss l2` selects the L2 baseline.
- The paired loader is consolidated into `joint_embedding_learning/data/datasets.py`. The documented ten-object configuration is retained in `config/datasets_config.yaml` with the same statistics, episode IDs, and orientation corrections.
- The canonical scripts are `scripts/preprocess.py`, `scripts/evaluate.py`, and `scripts/convert_tactile_to_latent.py`.
- The downstream package is `policy_learning/`, distributed as `tactx-policy` and imported as `tactile_policy`. Its Wang Lab metadata and DETR license remain intact.
- Removed code includes obsolete two-sensor/Bubbles/GelSlim/STL10 paths, LARS, experimental dynamic/MMD/CWCL/neighbor/contact-aware training variants, duplicate converters/evaluators, one-off plots, machine-specific launchers, PI0/OpenPI copies, unrelated hardware drivers and policy configurations, and the redundant paper PDF. These are outside the retained README workflows; the deletion does not claim they are useless for other experiments.
- Current residual and legacy small-CNN alignment architectures are preserved for checkpoint compatibility. ACT source dependencies and all 12 included TactX policy task/sensor split configurations remain available.
- Local ignored notes and paper visualization files remain excluded. Paper authors and citations are preserved; personal experiment names are removed from active code/configuration paths.

## Behavioral fixes

These changes affect execution and, where noted, new training trajectories. Old model state dictionaries remain structurally compatible.

1. Adam now applies the configured `weight_decay`. Previously it silently ignored that setting. Set it to `0.0` if reproducing the old optimizer behavior.
2. A zero-length InfoNCE warm-up now enables alignment at the configured start epoch. Previously the VAE trainer unintentionally disabled it at epoch zero. Setting `infonce_warmup_epochs: 1` reproduces that previous schedule.
3. Shorter paired loaders restart their iterators instead of retaining all tensor batches in `itertools.cycle`. Training rejects empty/undersized pairs before model construction; validation keeps remainder batches and resizes NT-Xent's batch mask accordingly.
4. Paired filenames are sorted deterministically by object, episode, and frame. Saved data configuration now includes selected objects and frame strides. Evaluation uses that saved configuration and defaults to `model.pt`; relocations and explicit overrides are supported.
5. Small validation sets no longer fail reconstruction plots. eFlesh bar plots select the correct sample. t-SNE selects the installed scikit-learn API and clamps perplexity to the available sample count. Dynamic evaluation includes all requested episodes.
6. Latent conversion handles scalar datasets, groups, attributes, and compression, batches sensor preprocessing, updates the latent width metadata, and atomically replaces output files. Existing outputs need `--overwrite`; input/output aliasing and missing streams fail explicitly. eFlesh geometric transforms never reorder frames.
7. ACT supports CPU device selection, config/data paths independent of the working directory, configurable worker counts, strict weight loading, and periodic optimizer/scheduler checkpoints. Unsupported S3/EMA/JIT options with missing or broken implementations are removed. Training-step naming is explicit; `--num_epochs` remains an alias.
8. ACT now implements PyTorch's `forward` interface instead of overriding module `__call__`. Device placement belongs to Accelerate. The package includes subpackages, YAML/JSON configs, and its existing third-party license in wheels.
9. The normalization helper finds per-object directories. The unused `pretrain` configuration key was removed; representation encoders retain their previous effective random initialization. The contrastive configuration retains its smaller FlexiTac encoder and the README identifies that difference. SimCLR is pinned to the tested revision. OpenCV is constrained to releases compatible with the pinned NumPy 1.26 environment.

New training after these fixes is not bit-for-bit equivalent to a historical run. In particular, weight decay, the first epoch's alignment weight, deterministic sample order, and fresh shorter-loader shuffles can change optimization. This branch does not assert that a smoke-trained model reproduces published results.

## Validation

Tests live in `tests/`. They create raw pickle recordings and HDF5 demonstrations in temporary directories, invoke the released CLI commands, and inspect the resulting tensors, model weights, metrics, and files.

The compatibility test loads the base commit's model definitions using Git history and verifies exact evaluation outputs after a strict state-dictionary transfer for residual VAE, small-CNN VAE, and small-CNN contrastive models. A normal full clone contains that base commit.

CPU tests cover:

- Raw preprocessing, frame pairing, held-out episode separation, normalization, orientation, and DataLoader workers.
- The four documented alignment objectives, finite saved weights, strict checkpoint reloads, and actual parameter updates.
- Static VAE and contrastive evaluation, small validation remainders, contact labels/probes, dynamic evaluation, plots, and metrics.
- D/E/F HDF5 conversion, preserved metadata and non-tactile data, latent values, vector transform ordering, and failure paths that preserve existing files.
- ACT training, validation, saved normalization statistics, action inference, and optimizer/scheduler checkpoint resume using the full included ACT layer/hidden-size configuration with a small batch and action chunk.
- Undefined-name/syntax checks, dependency consistency, README links, and a built wheel's included source/config/license files.

Environment-specific outcomes are recorded in [validation.json](validation.json). Run the complete suite in the Python 3.11 policy environment with the commands in the root README. The representation tests also run in Python 3.10.

No project dataset or published checkpoint is included locally. CUDA, mixed precision, distributed GPU execution, full 300-epoch/50,000-step training, hardware interfaces, and real-robot success rates are **not validated** by these CPU smoke tests. Code provenance findings and their limits are recorded in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
