# PPD: Offline Model-Based Optimization via Tabular Foundation Model Guided Preference Distillation

Code for the paper "PPD: Offline Model-Based Optimization via Tabular Foundation Model-Guided Preference Distillation".

## Overview

PPD addresses offline model-based optimization (MBO) via teacher-student knowledge distillation:

1. **TabPFN V2.6** serves as teacher, providing context-based predictions with conservative synthetic augmentation.
2. A **two-stage distillation set** (Sobol global exploration + density-aware local augmentation) covers the design space broadly while exploiting known high-value regions.
3. A lightweight **MLP** is trained with **uncertainty-weighted ListNet** loss on both offline data and the teacher-labeled distillation set.
4. **Gradient-based search** on the student produces optimized design candidates.

## Installation

TabPFN and design_bench have conflicting dependency requirements (TabPFN requires
`torch>=2.1`, while design_bench is built against `torch==1.13.1` and `numpy<1.22`).
Two separate conda environments are required.

### Environment 1: PPD Pipeline (training + candidate generation)

```bash
conda create -n ppd python=3.10 -y
conda activate ppd
pip install tabpfn>=7.1 scikit-learn scipy torch
```

### Environment 2: Oracle Evaluation (design-bench)

design-bench depends on MuJoCo for the Ant/DKitty continuous tasks and RDKit for
ChEMBL. For TFBind8/10 and Superconductor tasks only, MuJoCo can be omitted.

```bash
# Full installation (all tasks including Ant/DKitty)
conda create -n design_bench python=3.7 -y
conda activate design_bench
conda install -c conda-forge mujoco-py rdkit -y
pip install design-bench[all]==2.0.20
pip install morphing-agents==1.5.1

# Minimal installation (TFBind + Superconductor only, no MuJoCo)
conda create -n design_bench python=3.7 -y
conda activate design_bench
pip install design-bench==2.0.20
```

**Note:** `design-bench[all]` installs `gym[mujoco]<0.26.0`, which requires a MuJoCo
license key. Set `MUJOCO_PY_MJKEY_PATH` to your license file path. For offline MBO
evaluation, MuJoCo is only needed for the continuous morphology tasks (Ant/DKitty).

### TabPFN Model Checkpoint

TabPFN V2.6 model checkpoints are downloaded automatically on first use to
`~/.cache/tabpfn/`. An internet connection is required for the first run. The model
version is selected explicitly in the code:

```python
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

regressor = TabPFNRegressor.create_default_for_version(ModelVersion.V2_6)
```

This is already handled in `pipeline/teacher.py`.

### Data

Design-Bench datasets can be downloaded following the instructions at:

[https://huggingface.co/datasets/beckhamc/design_bench_data](https://huggingface.co/datasets/beckhamc/design_bench_data)

Place the downloaded `.npy` files under `data/`.

## Usage

```bash
# Single run (default: trains + oracle eval, seeds 0-7)
./scripts/run_ablation.sh AntMorphology-Exact-v0

# Single seed quick test
SEED_START=0 SEED_END=0 ./scripts/run_ablation.sh AntMorphology-Exact-v0
```

## Code Structure

```
PPD/
├── run.py                     # Main entry point
├── pipeline/
│   ├── teacher.py             # TabPFN V2.6 teacher
│   ├── distill_set.py         # Two-stage distillation set construction
│   ├── student.py             # MLP + uncertainty-weighted ListNet training
│   ├── search.py              # Gradient-based optimization
│   └── utils.py               # Standardization, helpers
├── eval/
│   └── eval_design_bench.py   # Oracle evaluation on Design-Bench
├── scripts/
│   └── run_ablation.sh        # Multi-seed experiment runner
└── data/                      # Design-Bench task data
```

## Code References

We sincerely appreciate the open-source contributions of:

- **TabPFN**: [https://github.com/PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN)
- **Design-Bench**: [https://github.com/rail-berkeley/design-bench](https://github.com/rail-berkeley/design-bench)
