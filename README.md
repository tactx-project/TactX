<div align="center">

# TactX (CoRL 2026)

### TactX: Learning Shared Tactile Representations Across Diverse Sensors

Junsung Park<sup>\*</sup> · Sachin Bhadang<sup>\*</sup> · Carmelo Sferrazza · Sha Yi · Xiaolong Wang

UC San Diego · Seoul National University · Amazon FAR

[![Project Page](https://img.shields.io/badge/Project%20Page-4285F4?style=for-the-badge)](https://tactx-project.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.31236-b31b1b?style=for-the-badge)](https://arxiv.org/abs/2606.31236)

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](conda_environment.yaml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](requirements.txt)

</div>

TactX learns a **shared 16-dimensional tactile representation** across Daimon (vision-based), eFlesh (magnetic), and FlexiTac (resistive) sensors. Sensor-specific encoders learn from paired contacts using contrastive alignment, self- and cross-reconstruction, and KL regularization. The shared representation lets tactile-conditioned ACT policies transfer across sensors without policy retraining.

The [paper](https://arxiv.org/abs/2606.31236) evaluates pick-and-place, plug insertion, board wiping, and object reorientation, including an additional out-of-distribution pick-and-place setting. Average zero-shot cross-sensor policy success improves from **27.5% to 45.9%** over vision-only transfer.

This repository includes representation learning, alignment and reconstruction evaluation, objective baselines, tactile-to-latent conversion, and downstream ACT policy training code.

## Installation

Clone the repository and create the representation-learning environment:

```bash
git clone https://github.com/tactx-project/TactX.git
cd TactX

conda create -n tactx python=3.10 -y
conda activate tactx
python -m pip install -r requirements.txt
```

The core dependencies include `torch==2.5.1`, `torchvision==0.20.1`, and `numpy==1.26.4`; see [requirements.txt](requirements.txt) for the full list. The commands below use a CUDA-capable GPU and run from the repository root. Training logs to Weights & Biases; `WANDB_MODE=offline` keeps logs local.

The downstream policy package specifies **Python 3.11** and additional dependencies. Follow the separate [ACT policy workflow](policy_learning/README.md) for that environment.

## Data

Datasets and pretrained checkpoints are **not bundled with this repository**. The examples below require local paired-contact data, and policy training additionally requires robot demonstrations. The [project page](https://tactx-project.github.io/) is the central resource for the paper and project materials.

Representation-learning examples use the `dataset_new` entry in [config/datasets_config.yaml](config/datasets_config.yaml). Its default data root is:

```text
policy_learning/data/latent_data_new_processed/
├── de_pair/
│   ├── daimon/<object>/ep0000_fr0000.pt
│   └── eflesh/<object>/ep0000_fr0000.pt
├── ef_pair/
│   ├── eflesh/<object>/ep0000_fr0000.pt
│   └── flexi/<object>/ep0000_fr0000.pt
└── fd_pair/
    ├── flexi/<object>/ep0000_fr0000.pt
    └── daimon/<object>/ep0000_fr0000.pt
```

Each pair matches frames by filename within an object directory. The preprocessing script [scripts/preprocess.py](scripts/preprocess.py) produces the following tensors:

| Sensor | Sensing modality | Stored tensor per frame |
| :--- | :--- | :--- |
| Daimon | Vision-based | `(3, 240, 320)`: depth, shear-x, shear-y |
| eFlesh | Magnetic | `(15,)`: five 3-axis readings |
| FlexiTac | Resistive | `(12, 16)`: pressure grid |

The loader resizes Daimon inputs to `224 × 224` and applies the configured normalization and pair-specific orientation corrections. `dataset_new` defines held-out episode IDs separately for each object. If using a different dataset, update its paths, normalization statistics, and splits accordingly.

To process a new collection with the raw directory layout described in the script:

```bash
python -m scripts.preprocess \
    --input_root /path/to/raw_recordings \
    --output_root policy_learning/data/latent_data_new_processed
```

For a new dataset, compute normalization statistics on its training recordings with `python -m scripts.compute_tri_stats --data_root /path/to/training_recordings`, then put those statistics and the held-out episode IDs into a dataset YAML. Existing published-data settings are kept in `dataset_new`.

Define the ten-object list once in the shell used for training and evaluation:

```bash
OBJECTS="circle cylinder cylinder_vertical ellipse large_ball plane plane_horizontal small_ball square triangle"
```

## Train

### TactX representation

Train the `cross_modal_vae3` model with paired batches from all three sensor pairs:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python -m scripts.train_alignment \
    --dataset_name dataset_new \
    --model_name cross_modal_vae3 \
    --device cuda:0 \
    --run_name tactx \
    --objects $OBJECTS \
    --epochs 300 \
    --num_workers 4
```

The model and loss settings are in [config/models_config.yaml](config/models_config.yaml). The current `cross_modal_vae3` configuration uses a 16-D latent, a ResNet-18 Daimon encoder, an eFlesh MLP, and a residual CNN for FlexiTac. It uses Adam with learning rate `1e-4` and weight decay `1e-4`, batch size `64` per sensor pair, contrastive temperature `0.01`, reconstruction weight `1.0`, and KL weight warmed up to `0.1` over 30 epochs. The command above overrides the configuration's 100-epoch default to 300 epochs. Representation encoders start from random initialization.

For another data location, add `--data_root /path/to/paired_data`. A custom YAML can be selected with `--dataset_config`; `--batch_size` and `--checkpoint_root` are also configurable. Use `--device cpu` for small smoke runs. Training requires at least one full batch per pair; validation includes the final incomplete batch. `--resume_from` warm-starts model weights only.

### Objective baselines

The repository also implements reconstruction-only, contrastive-only, and L2-alignment variants. The included contrastive configuration uses the legacy small-CNN FlexiTac encoder; the VAE configuration uses the residual encoder. Keep this architecture difference in mind when comparing results.

<details>
<summary>Baseline training commands</summary>

**Reconstruction + KL, without contrastive alignment:**

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python -m scripts.train_alignment \
    --dataset_name dataset_new \
    --device cuda:0 \
    --run_name recon_only \
    --objects $OBJECTS \
    --epochs 300 \
    --infonce_warmup_start 9999
```

**Contrastive-only (`cross_modal_clr3`, without decoders or KL):**

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python -m scripts.train_alignment \
    --dataset_name dataset_new \
    --model_name cross_modal_clr3 \
    --device cuda:0 \
    --run_name contrastive_only \
    --objects $OBJECTS \
    --epochs 100
```

**L2 alignment + reconstruction + KL:**

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python -m scripts.train_alignment --alignment_loss l2 \
    --dataset_name dataset_new \
    --device cuda:0 \
    --run_name l2_alignment \
    --objects $OBJECTS \
    --epochs 300
```

</details>

## Checkpoints

Training writes a timestamped run directory with weights and the model/data configuration:

```text
checkpoints/cross_modal_tri/cross_modal_vae3/dataset_new_run_tactx_<timestamp>/
├── model.pt
├── model_details.yaml
└── dataset_details.yaml
```

`model.pt` is updated during training and saved again at completion. Longer runs also retain selected `model_ep*.pt` snapshots. Contrastive-only runs are saved under `checkpoints/cross_modal_tri/cross_modal_clr3/`.

Keep the configuration files with the weights. In particular, the current residual FlexiTac model and legacy `small_cnn` checkpoints use different architectures; loading must use the checkpoint's own `model_details.yaml`.

## Evaluation

Set `CHECKPOINT_DIR` to a trained run and evaluate the same dataset and object split used during training:

```bash
CHECKPOINT_DIR="/path/to/trained/run"

CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --checkpoint_file model.pt \
    --dataset_name dataset_new \
    --objects $OBJECTS \
    --out_dir eval_results/tactx \
    --device cuda:0
```

The evaluator writes `metrics.txt` and figures for paired/transitive cosine similarity, latent L2 distance, cross-modal retrieval, sensor-identity probes, self/cross reconstruction, and t-SNE. Use `--max_tsne_points 2000` to limit the number of plotted points. For a `cross_modal_clr3` checkpoint, add `--contrastive_only`; reconstruction evaluation is skipped.

The evaluator defaults to `model.pt`. Use `--checkpoint_file model_ep100.pt` to select a saved snapshot. It reads the saved dataset configuration, object list, and frame strides so normalization and episode splits match training. Use `--data_root /new/path` to relocate data, or `--dataset_config /path/to/datasets.yaml` to explicitly override the data configuration.

<details>
<summary>Optional contact probes and dynamic-data evaluation</summary>

Build threshold-derived contact labels from the paired tactile recordings:

```bash
python -m scripts.build_contact_labels \
    --data_root policy_learning/data/latent_data_new_processed \
    --out_root contact_labels/dataset_new \
    --objects $OBJECTS
```

Then add `--contact_label_root contact_labels/dataset_new` to the evaluation command to enable contact probes.

For additional paired dynamic recordings, pass `--dynamic_data_root` with one or more directories containing `de_pair/`, `ef_pair/`, and `fd_pair/`, and `--dynamic_objects` with the corresponding object/task directory names in the same order.

</details>

## Downstream ACT policies

The [policy workflow](policy_learning/README.md) covers converting demonstration tactile streams into posterior means, preparing dataset splits, and training an ACT policy with 16-D tactile inputs. Code and model/sensor configurations are in [policy_learning/](policy_learning/).

Representation evaluation above operates on recorded tactile data. Reproducing robot policy success rates additionally requires demonstrations, policy checkpoints, and the corresponding robot/sensor deployment setup.

## Repository structure

```text
config/                    Dataset splits, normalization, and model settings
joint_embedding_learning/  Encoders, decoders, losses, data loaders, and trainers
scripts/                   Preprocessing, training, evaluation, latent conversion
policy_learning/           ACT training and task configurations
tests/                     Synthetic workflow and regression tests
maintenance/               Cleanup manifest and validation notes
```

## Validation and maintenance

The cleanup branch keeps the documented offline workflows. See [the change and validation record](maintenance/CLEANUP.md) and [the complete path manifest](maintenance/cleanup_manifest.json) for removed and renamed files, behavioral fixes, and test limitations. Historical experiment flags and hardware-specific launchers are no longer part of this release.

Run the synthetic tests in the Python 3.11 policy environment:

```bash
python -m pip install -r requirements-dev.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest tests -q
python -m ruff check --select F821,F822,F823,E9 joint_embedding_learning scripts policy_learning tests
```

Tests exercise real preprocessing, training, evaluation, and HDF5 conversion code with tiny generated datasets. They do not reproduce the paper's robot success rates.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{park2026tactx,
  title         = {TactX: Learning Shared Tactile Representations Across Diverse Sensors},
  author        = {Park, Junsung and Bhadang, Sachin and Sferrazza, Carmelo and Yi, Sha and Wang, Xiaolong},
  year          = {2026},
  eprint        = {2606.31236},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2606.31236}
}
```

## Acknowledgements

The representation-learning code uses [SimCLR](https://github.com/Spijkervet/SimCLR) components. The policy implementation builds on ACT and includes adapted DETR components under [policy_learning/tactile_policy/third_party/detr/](policy_learning/tactile_policy/third_party/detr/), with their [upstream license](policy_learning/tactile_policy/third_party/detr/LICENSE). The policy package retains its Wang Lab attribution. See [third-party notices](THIRD_PARTY_NOTICES.md) for provenance and scope.
