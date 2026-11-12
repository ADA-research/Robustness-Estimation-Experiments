# Robustness-Estimation-Experiments

This repository contains the code, neural networks, results and analysis for the paper:

**Adversarial Upperbounds to Robustness Distributions**

*Aaron Berger\*, Annelot W. Bosman\*, Nils Eberhardt, Henning Duwe, Holger H. Hoos and Jan N. van Rijn* (\* equal contribution)\
Annals of Mathematics and Artificial Intelligence (LION track), extended version of [1].

<!-- TODO: add journal citation / DOI once available -->

[1] A. Berger, N. Eberhardt, A. W. Bosman, H. Duwe, H. H. Hoos, J. N. van Rijn: *Empirical Analysis of Upper Bounds for
Robustness Distributions using Adversarial Attacks*. LION 19, LNCS (2025).

[2] *Robustness Distributions in Neural Network Verification*. Journal of Artificial Intelligence Research (JAIR).
Code and data: [ADA-research/NNV_JAIR_robustness_distributions](https://github.com/ADA-research/NNV_JAIR_robustness_distributions).
<!-- TODO: add authors, volume and DOI of [2] -->

## Overview

| Folder | Content |
|---|---|
| [`networks/`](networks) | MNIST and CIFAR-10 networks (ONNX). ImageNet models are loaded from [timm](https://huggingface.co/timm), see below. |
| [`experiments/mnist/`](experiments/mnist) | Experiment scripts for MNIST (alpha,beta-CROWN, AutoAttack, FAB, FGSM, PGD) |
| [`experiments/cifar10/`](experiments/cifar10) | Experiment scripts for CIFAR-10 (AutoAttack, FGSM, PGD; binary and iterative search) |
| [`experiments/imagenet/`](experiments/imagenet) | Experiment scripts for ImageNet (AutoAttack, FGSM, PGD; binary and iterative search) |
| [`results/`](results) | All results used in the paper (CSV) |
| [`analysis/`](analysis) | Jupyter notebooks that produce every figure and table of the paper from `results/` |

All robustness distributions are computed with [VERONA](https://github.com/ADA-research/VERONA).

## Networks

- **MNIST**: [mnist-net_256x2](networks/mnist_networks/mnist-net_256x2.onnx),
  [mnist_relu_3_50](networks/mnist_networks/mnist_relu_3_50.onnx),
  [mnist_relu_4_1024](networks/mnist_networks/mnist_relu_4_1024.onnx),
  [mnist_relu_9_100](networks/mnist_networks/mnist_relu_9_100.onnx)
- **CIFAR-10**: [cifar_7_1024](networks/cifar10_networks/cifar_7_1024_standard.onnx),
  [conv_big](networks/cifar10_networks/conv_big_standard.onnx),
  [resnet_4b](networks/cifar10_networks/resnet_4b_standard.onnx)
- **ImageNet** (loaded with `timm.create_model(name, pretrained=True)`):

| Name in paper | timm identifier |
|---|---|
| eva | `eva_giant_patch14_224.clip_ft_in1k` |
| beitv2 | `beitv2_large_patch16_224.in1k_ft_in22k_in1k` |
| openai_1 | `vit_large_patch14_clip_224.openai_ft_in12k_in1k` |
| openai_2 | `vit_large_patch14_clip_224.openai_ft_in1k` |
| laion_2b | `vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k` |

<!-- TODO: add timm version used on the cluster -->

## Experimental setup (summary)

| | MNIST | CIFAR-10 | ImageNet |
|---|---|---|---|
| Images | first 100 test images (alpha,beta-CROWN), complete test set (attacks) | class-balanced sample of 100 test images (alpha,beta-CROWN, from JAIR [2]), complete test set (attacks) | validation set (50 000 images, used as test set); AutoAttack on the first 5 000 |
| Search space | i/255 in [0, 0.4] | `np.arange(0, 0.4, 0.0039)` | `np.arange(0, 0.4, 0.0001)` |
| Success criterion | top-1 | top-1 | top-5 (the true label is not among the 5 highest scores) |
| Preprocessing | none ([0, 1]) | none ([0, 1]) | `Resize([224, 224])` + `ToTensor()`, no mean/std normalisation ([0, 1]) |
| Misclassified images | removed | removed | removed (top-5) |

- PGD: 40 iterations; for CIFAR-10/ImageNet with step size epsilon/40 (`PGDAttack(number_iterations=40)`); for MNIST
  with the fixed step size (0.2 - 1/255)/k, i.e. about 0.0049 for k = 40. `experiments/mnist/run_pgd_mnist_experiment.py`
  uses the default step size epsilon/k. The MNIST results in `results/mnist/` were produced with the fixed step size, so
  running the script unchanged will not reproduce them exactly. To reproduce them, pass
  `step_size=(0.2 - 1/255) / num_iterations` to `PGDAttack`.
- AutoAttack: standard version, L-infinity ([fra31/auto-attack](https://github.com/fra31/auto-attack), installed from
  GitHub; latest commit `a392200`), unchanged: APGD-CE (cross-entropy, 100 iterations), APGD-T (targeted DLR loss,
  9 target classes, 100 iterations each), FAB-T (9 target classes, 100 iterations) and Square (margin loss, 5 000 queries).
- The attacks optimise their standard (top-1) objectives; only the success check uses top-k
  (`AttackEstimationModule(top_k=...)` in VERONA: a query is successful if the true label is not in `torch.topk(output, k)`).
- AutoAttack components stop for an input once its top-1 prediction changes, and inputs that are already misclassified
  under top-1 are not attacked (`run_standard_evaluation`). Under top-5, AutoAttack therefore finds no adversarial example
  for images that are misclassified under top-1 but correctly classified under top-5 (about 10 % on ImageNet).
- All perturbed inputs are clipped to [0, 1]; the L-infinity constraint is applied in this pixel space.

## Reproducibility details

- **Models:** MNIST and CIFAR-10 networks as ONNX files in `networks/`; ImageNet models by their timm identifiers
  (table above), loaded with `timm.create_model(name, pretrained=True)`.
- **Software:** VERONA (`ada_verona` for CIFAR-10/ImageNet, the earlier `robustness_experiment_box` for MNIST),
  AutoAttack from GitHub (latest commit `a392200`), alpha,beta-CROWN. Exact versions: see the environment file below
  (TODO).
- **Seeds:** `torch.manual_seed(0)` in all experiment scripts; CIFAR-10 sample: `StratifiedShuffleSplit(..., random_state=42)`;
  FAB: `seed=42`. AutoAttack is called without a seed by VERONA's wrapper, so its components are seeded with the current
  time (`AutoAttack.get_seed`).
- **Attack budgets and restarts:**
  - FGSM: one step. PGD: 40 steps with the step size per dataset given above, no restarts (PGD random: random start).
  - AutoAttack, standard version: APGD-CE (1 restart, 100 iterations), APGD-T (9 target classes, 1 restart and
    100 iterations each), FAB-T (9 target classes, 1 restart, 100 iterations), Square (1 restart, 5 000 queries).
  - FAB (MNIST only): untargeted and targeted, 5 restarts, 100 iterations (`experiments/mnist/run_fab_attack_experiment.py`).
  - Search: binary search over the radii of the setup table; the monotonicity experiments evaluate every radius.
- **alpha,beta-CROWN:** MNIST: 600 s per query (`experiments/mnist/run_abcrown_mnist_experiment.py`), configuration
  `experiments/mnist/abcrown_verification/ffn_config.yaml`. CIFAR-10: results of [2] (3 600 s per query), see
  "Data taken from the JAIR repository".
- **Input domain and norm constraints:** enforced by construction. FGSM and PGD clip every perturbed input to the
  epsilon-ball around the original image and to [0, 1] after every step (VERONA `fgsm_attack.py`, `pgd_attack.py`);
  AutoAttack projects onto the epsilon-ball and clips to [0, 1] internally. The success of a query is evaluated on the
  returned (constrained) input.

## Results

All files are CSVs with one row per (network, image, method).

| File | Content |
|---|---|
| `results/mnist/results.csv` | alpha,beta-CROWN (first 100 images) and all attacks (complete test set), incl. PGD with 10/40/100 iterations |
| `results/mnist/monotonicity.csv` | Iterative search over all epsilons, first 100 images (highest epsilon without and lowest epsilon with an adversarial example per image; `undefined` if none was found) |
| `results/cifar10/attacks.csv` | AutoAttack, FGSM, PGD on the complete test set |
| `results/cifar10/monotonicity.csv` | Iterative search over all epsilons, 20 images per network |
| `results/cifar10/abcrown_3600s.csv` | alpha,beta-CROWN (3 600 s timeout per query) on a class-balanced sample of 100 test images, taken from the JAIR repository (see below). In the JAIR file, `image_id` is the position in the sample (`StratifiedShuffleSplit(test_size=100, random_state=42)` on the test labels); here it is converted to the CIFAR-10 test index (original value in `jair_sample_position`) by `results/cifar10/convert_jair_ids.py` |
| `results/cifar10/balanced_sample_test_indices.csv` | The 100 test indices of this sample (10 per class), i.e. the CIFAR-10 "100 images" in the analysis |
| `results/imagenet/attacks_top5.csv` | AutoAttack (first 5 000 images), FGSM, PGD (validation set), top-5 |
| `results/imagenet/attacks_top1.csv` | top-1 runs (Figure 9) |
| `results/imagenet/monotonicity.csv` | Iterative search over all epsilons |

### Data taken from the JAIR repository

The CIFAR-10 alpha,beta-CROWN results are not recomputed here. They come from the experiments of [2], published in
[ADA-research/NNV_JAIR_robustness_distributions](https://github.com/ADA-research/NNV_JAIR_robustness_distributions):

- Source file: [`data/CIFAR-10/results_CIFAR_JAIR.csv`](https://github.com/ADA-research/NNV_JAIR_robustness_distributions/blob/0179f7830df441e3da31031475ba5a178402acda/data/CIFAR-10/results_CIFAR_JAIR.csv)
  (commit `0179f78`, 2025-02-20; git blob `4efacdb`), test-split rows only.
- Sampling code: `experiments/CIFAR-10/main.py` in that repository (class-balanced sample of 100 test images).
- Derived files here: `results/cifar10/abcrown_3600s.csv` and `results/cifar10/balanced_sample_test_indices.csv`,
  both reproduced by `results/cifar10/convert_jair_ids.py` from the source file and the CIFAR-10 test batch.

If you use these results, please also cite [2].

Columns:

| Column | Meaning |
|---|---|
| `network`, `image_id` | network and index of the image in the test/validation set |
| `original_label` | true label |
| `verifier` | method; for CIFAR-10/ImageNet the full VERONA description including the attack configuration and top-k |
| `epsilon_value` | lower bound: highest epsilon for which no adversarial example was found (verified for alpha,beta-CROWN) |
| `smallest_sat_value` | upper bound: smallest epsilon for which an adversarial example was found. If none was found, the maximum of the search space (0.4 MNIST, 0.3978 CIFAR-10, 0.3999 ImageNet; `undefined` in the monotonicity files) |
| `total_time` | wall-clock time in seconds for the complete search for this image |
| `adversarial` | (MNIST, FAB only) whether FAB found an adversarial example |

## Analysis

```bash
pip install -r analysis/requirements.txt
cd analysis
jupyter notebook
```

| Notebook | Content |
|---|---|
| `paper_figures_and_tables.ipynb` | Every data-based figure and table of the paper, in the order in which they appear (Table 2, Fig. 2–10, Tables 3, 4, C2–C7, D8–D17, Fig. D1, D2) |
| `attack_failures_sample_sizes_runtimes.ipynb` | Attack failure rates, sample sizes, running-time quantiles |

Figures are written to `analysis/plots/` as `.eps` with the file names used in the paper's LaTeX source; tables to `analysis/tables/` as LaTeX table bodies in the paper's format (one file per table, the first line names its `\label`). Shared code is in `analysis/common.py`.
`analysis/tables/table_experiments.tex` and `analysis/tables/table_sample_sizes.tex` are complete `table` environments (new appendix tables: the experiments per dataset with initial images, methods and radii; the images per network and the denominator of every reported quantity).

**Failed attacks.** If a method finds no adversarial example within the search space, the stored value is the
maximum of the search space. These instances are treated as failures: they are excluded from averages, standard
deviations and statistical tests, and the tables report the failure rate per method. Rank-based metrics (RMC, VBA)
treat a failure as worse than any successful result. `EXCLUDE_FAILURES = False` in `analysis/common.py`
reproduces the tables of the originally submitted manuscript, which included these values.

## Reproducing the experiments

- MNIST: `experiments/mnist/` (uses the earlier VERONA package `robustness_experiment_box`).
- CIFAR-10: `experiments/cifar10/` (uses `ada_verona`), the versions run on the cluster:
  - `main_cifar.py`: binary search on the complete test set; AutoAttack in batches of 500 images (SLURM job `run.sh`),
    FGSM and PGD on the complete test set in one job.
  - `main_monotonicity_cifar.py`: iterative search over all epsilons for 20 images per network. The images are taken
    from a class-balanced sample of 100 **training** images (`StratifiedShuffleSplit`, seed 42).
  - `config/`: alpha,beta-CROWN configurations.
- ImageNet: `experiments/imagenet/` (uses `ada_verona`), the versions run on the cluster:
  - `main_vit.py` / `main.py`: top-5, binary search, in batches of 5 000 validation images (`arguments.txt`).
    `main_vit.py` skips images that are already done; `main.py` resumes from `--start`
    (`arguments_eva.txt`, `arguments_fitlarge.txt`).
  - `main_top1_image.py`: top-1 runs (`arguments_top1.txt`).
  - `main_monotonicity_image.py`: iterative search over all epsilons for 20 images per network
    (`arguments_monotonicity.txt`).
  - `run_batches.sh`: SLURM array job.
- Paths to data and output folders (`/path/to/...` in the Python scripts, `SCRATCH_DIR` in the `.sh` files) need to be adapted.

## Open TODOs before publishing this branch

- [ ] Add the exact environment (`pip freeze` of the cluster environment: ada_verona, torch, timm, AutoAttack commit).
- [ ] Add the journal citation.
- [ ] Complete reference [2] (JAIR authors, volume, DOI).

## External packages

- [VERONA](https://github.com/ADA-research/VERONA): open-source package for creating robustness distributions.
- [alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
- [AutoAttack](https://github.com/fra31/auto-attack), [FAB attack](https://github.com/fra31/fab-attack)
- FGSM [Goodfellow et al., 2015](https://arxiv.org/abs/1412.6572), PGD [Madry et al., 2018](https://arxiv.org/abs/1706.06083)
