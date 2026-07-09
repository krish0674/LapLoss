# LapLoss: Laplacian Pyramid-based Multiscale Loss for Image Translation

Official implementation of **"LapLoss: Laplacian Pyramid-based Multiscale Loss for Image Translation"** by Krish Didwania, Ishaan Gakhar, Prakhar Arya, and Sanskriti Labroo (Manipal Institute of Technology), accepted at the **DeLTa Workshop, ICLR 2025 (ICLRW)**. Paper: [arXiv:2503.05974](https://arxiv.org/abs/2503.05974). An extended journal version is in preparation; this repository accompanies both.

This repository contains the reference implementation of a contrast-enhancement framework that applies **adversarial supervision independently at each level of a Laplacian pyramid**. Instead of a single discriminator operating on the full-resolution output, each pyramid band is matched against its ground-truth counterpart by a dedicated discriminator, so the generator receives localized, frequency-aware gradients: fine edge detail is supervised at high-frequency bands while global brightness and structure are supervised at the low-frequency residual. The `main` branch instantiates this idea on the **Laplacian Pyramid Transformer Network (LPTN)** backbone and is trained and evaluated on the SICE dataset.

> **Note on branches.** `main` is the LPTN variant described here. Other backbones and task variants (e.g. the LapGSR variant referenced in the paper) live on separate branches (`3d`, `minor_proj`, `rain`, `wll`, `temp`). This README documents `main` only; the run instructions are not guaranteed to transfer to the other branches unchanged.

## Abstract

Contrast enhancement is a crucial component of image-to-image translation (I2IT) that improves visual quality by adjusting the intensity differences between pixels. Many existing methods struggle to preserve fine-grained details, often losing low-level features. This work integrates adversarial training into a Laplacian pyramid by introducing a multi-level discriminator architecture, applying adversarial supervision independently at each pyramid level. Aligning discriminators with individual pyramid levels lets the network learn scale-specific representations — capturing subtle edge detail at higher frequencies while maintaining consistent brightness and structure at lower frequencies. This decomposition-driven supervision provides more localized and frequency-aware gradients during training, yielding stable convergence and improved generalization across lighting conditions. The framework is validated on two pyramidal architectures — LPTN and the Laplacian Pyramid for Guided Thermal Super-Resolution (LapGSR) — demonstrating flexibility across multi-scale networks, and achieves state-of-the-art results across lighting conditions on the SICE dataset.

## Method at a glance

The generator (`LPTNPaper`) decomposes the input into a Laplacian pyramid with `num_high = 2`, i.e. **three levels**: two high-frequency detail bands (indices 0, 1) and one low-frequency Gaussian residual (index 2).

- **`Trans_low`** translates the coarsest residual (`nrb_low` residual blocks, `tanh` output).
- **`Trans_high`** predicts a mask for the mid band from the concatenation of the mid band, the upsampled input residual, and the upsampled translated residual (`nrb_high` blocks).
- **`Trans_top`** refines the upsampled mask for the finest band (`nrb_top` blocks).
- The translated bands are reassembled with `pyramid_recons`.

Three discriminators — `Discriminator1/2/3` — supervise levels 0/1/2 respectively (PatchGAN-style with InstanceNorm; `D3` is one block shallower because it acts on the smallest map). The generator loss is a weighted sum over the pyramid levels of a pixel term (MSE, scaled by `loss_weight`) plus an adversarial term; discriminators are trained with a real/fake loss plus a gradient penalty (`gp_weight = 100`). The generator is optimized with **SOAP**; the discriminators with **Adam**.

## Repository structure

```
LapLoss/
├── README.md
├── tryinit.ipynb                     # exploratory Kaggle notebook (duplicated under src/)
└── src/
    ├── train.py                      # training entry point (argparse + Weights & Biases)
    ├── eval.py                       # evaluation entry point
    ├── flops.py                      # MACs / parameter count via ptflops
    └── utils/
        ├── dataloader.py             # SICE train/test datasets + augmentations
        ├── trainer.py                # training loop
        ├── evaluater.py              # evaluation loop
        ├── misc.py                   # small visualization/helper utilities
        └── models/
            ├── base_model.py         # metrics (PSNR/SSIM/LPIPS/MS-SSIM), save/load, schedulers
            ├── lptn_model.py         # generator + 3 discriminators, losses, optimize step
            ├── optimizer.py          # SOAP optimizer
            ├── lr_scheduler.py       # MultiStepRestartLR
            ├── archs/
            │   ├── LPTN_paper_arch.py    # Lap_Pyramid_Conv + generator (LPTNPaper)
            │   ├── lptn.py              # alternate LPTN arch
            │   └── discriminator_arch.py # per-level discriminators
            └── losses/
                ├── losses.py          # MSELoss, GANLoss, gradient penalty, r1, path reg
                ├── metrics.py         # MultiScaleSSIM + custom LPIPS
                └── loss_util.py       # weighted_loss decorator
```

All entry points use imports rooted at `utils` (e.g. `from utils.trainer import train_model`), so **commands must be run from inside `src/`**.

## Environment

The bytecode in the repo was produced by **CPython 3.11**. There is no lockfile in the repo; the dependency set below is reconstructed from the imports. Pin these to the exact versions you use for the camera-ready and commit the result as `requirements.txt`.

```
torch
torchvision
torchmetrics
lpips
albumentations
opencv-python
numpy
scipy
scikit-learn
pandas
matplotlib
Pillow
tqdm
ptflops
wandb
```

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt   # after you create it from the list above
```

A CUDA-capable GPU is assumed (`device='cuda'` by default). Install the `torch` / `torchvision` build that matches your CUDA toolkit.

## Datasets

The framework is trained and tested on **SICE** and on its mixed-exposure variants **SICE_Grad** and **SICE_Mix**.

| Dataset | Role | Source | Original paper |
|---|---|---|---|
| SICE (v1 / v2) | Train / val / test | [github.com/csjcai/SICE](https://github.com/csjcai/SICE) (images hosted on the Google Drive / BaiduYun links in that repo) | Cai et al., *IEEE TIP* 2018 |
| SICE_Grad, SICE_Mix | Test only (mixed exposure) | Official: [github.com/ShenZheng2000/LLIE_Survey](https://github.com/ShenZheng2000/LLIE_Survey) · Mirror used here: [Google Drive](https://drive.google.com/file/d/1gii4AEyyPp_kagfa7TyugnNPvUhkX84x/view) | Zheng et al., arXiv:2212.10772, 2022 |

SICE contains 589 multi-exposure scene sequences (7 or 9 images each, from under- to over-exposed) with a single well-exposed reference per scene. SICE_Grad and SICE_Mix are derived from SICE by cutting each reference into panels and re-tiling them: SICE_Grad arranges panels from low to high exposure (with some normally-exposed panels randomly placed), while SICE_Mix permutes panels at random. Both are reshaped to roughly 600×900 and are meant purely as **test** sets for uneven-illumination robustness.

The links are provided for reference only; all rights to the data belong to the original authors, and any use must comply with their terms.

### Expected directory layout

The dataloaders expect the following on-disk layout. `Dataset_Part2` is used for **training/validation**; `Dataset_Part1` for the **per-folder exposure test**; the `SICE_Grad` / `SICE_Mix` folders (with a shared `SICE_Reshape` label folder) for the mixed-exposure tests.

```
<root_dir>/
├── Dataset_Part1/Dataset_Part1/
│   ├── <NNN>/            # numbered scene folders, 7 or 9 exposures each
│   └── Label/<NNN>.JPG   # reference (well-exposed) image per scene (.JPG/.PNG/.JPEG)
├── Dataset_Part2/Dataset_Part2/
│   ├── <NNN>/
│   └── Label/<NNN>.JPG
├── SICE_Grad/            # gradually-degraded inputs   (SICEGradTest)
├── SICE_Mix/             # mixed-degradation inputs    (SICEMixTest)
└── SICE_Reshape/         # shared labels for Grad/Mix
```

### How the data is partitioned

You do **not** need to pre-split anything manually — the partitioning is done in code from the folder structure above:

- **Train / validation** come from `Dataset_Part2`. `SICETrainDataset` lists the numbered scene folders, shuffles them with a fixed `seed=42`, and takes the first 80% as training and the remaining 20% as validation (`split_ratio=0.8`). The split is at the *scene* level, so no scene appears in both train and val.
- **Test (standard SICE)** comes from `Dataset_Part1` via `SICEAllImagesTestDataset` (all exposures of a single scene selected with `--tf`) or `SICETestDataset` (a fixed index list, one exposure per scene).
- **Test (mixed exposure)** comes from the `SICE_Grad/` and `SICE_Mix/` input folders, each paired against the shared `SICE_Reshape/` references, via `SICEGradTest` / `SICEMixTest`.

To reproduce the paper's setup exactly: download SICE and place its `Dataset_Part1` and `Dataset_Part2` (each with its `Label/` subfolder) under `<root_dir>`; download the SICE_Grad/SICE_Mix archive and place `SICE_Grad/`, `SICE_Mix/`, and `SICE_Reshape/` under the same `<root_dir>`. Train on `Dataset_Part2` and report on all four test sets. If you change `--split_ratio` or the `seed`, record it — the numbers depend on the split.

Loader behaviour to be aware of when reproducing numbers:

- Images are resized to **608 × 896**; portrait images are rotated to landscape; pixels are scaled to `[0, 1]`. Training augmentation adds vertical/horizontal flips and a mild shift-scale-rotate.
- **Train/val split** is the 80/20 scene-level split described above, shuffled with a fixed `seed=42`.
- **Exposure selection** (`--exposure`): `under` keeps the first half of each scene's exposures, `over` keeps the second half, `both` keeps all. The reported configuration uses `over`.
- One caveat: the validation split is constructed without passing an exposure type, so validation defaults to `both` even when training on `over`. Keep this in mind if you tune on validation metrics.

## Training

Run from `src/`:

```bash
cd src
python train.py \
  --root_dir /path/to/SICE_root \
  --exposure over \
  --epochs 300 \
  --batch_size 8 \
  --loss_weight 3000 \
  --gan_type vanilla \
  --nrb_low 3 --nrb_high 3 --nrb_top 3 \
  --levels 0 1 2 \
  --weights 0.5 0.3 0.2
```

| Argument | Default | Meaning |
|---|---|---|
| `--root_dir` | — | Dataset root laid out as above (required in practice) |
| `--dset` | `sice` | Dataset selector (only `sice` is wired up for training) |
| `--exposure` | `over` | Exposure subset used for training (`under` / `over` / `both`) |
| `--epochs` | `300` | Number of epochs |
| `--batch_size` | `8` | Batch size |
| `--loss_weight` | `3000` | Weight on the per-level pixel (MSE) term |
| `--gan_type` | `vanilla` | GAN objective (`vanilla`, `standard`, `lsgan`, `wgan`, `wgan_softplus`, `hinge`) |
| `--nrb_low` / `--nrb_high` / `--nrb_top` | `3` / `3` / `3` | Residual blocks in `Trans_low` / `Trans_high` / `Trans_top` |
| `--levels` | `0 1 2` | Pyramid levels supervised by the generator adversarial loss |
| `--weights` | `0.5 0.3 0.2` | Per-level loss weights (finest → coarsest) |
| `--device` | `cuda` | Compute device |
| `--lr` | `1e-4` | **Currently unused** — see Reproducibility notes |
| `--key` | *(hardcoded)* | Weights & Biases API key — **remove before publishing** |
| `--sf_path` | `./best_model_g.pth` | Path used by a (currently disabled) warm-start hook |

**Logging and checkpoints.** Training logs to a Weights & Biases project named `LapLoss`. The best generator (by validation SSIM) is written to `./best_model_g.pth` (discriminators to `./best_model_d.pth`) in the working directory, and is reloaded at the end of training.

To run without a W&B account, either export `WANDB_MODE=offline` or replace `wandb.init(...)` with a no-op; do **not** rely on the committed key.

## Evaluation

> **Fix required before this runs outside the original Kaggle environment.** In `utils/evaluater.py`, `eval()` hardcodes `root_dir="/kaggle/input/sicedataset"` internally (the passed `--root_dir` is ignored), and `--model_path` defaults to a `/kaggle/working/...` path. Point both at your local paths before running. As written, only the `SICEAllImagesTestDataset` block (a single scene folder, id `--tf`) is active; the other test sets (`SICETestDataset`, `SICEMixTest`, `SICEGradTest`) are present but commented out.

```bash
cd src
python eval.py \
  --root_dir /path/to/SICE_root \
  --model_path /path/to/best_model_g.pth \
  --exposure over \
  --tf 10 \
  --nrb_low 3 --nrb_high 3 --nrb_top 3
```

| Argument | Default | Meaning |
|---|---|---|
| `--root_dir` | — | Dataset root (**note:** overridden inside `eval()` — patch this) |
| `--model_path` | `/kaggle/working/...` | Generator checkpoint to load (patch to a local path) |
| `--exposure` | `over` | Exposure subset for the test set |
| `--tf` | `10` | Scene-folder id used as the fixed test folder |
| `--nrb_low` / `--nrb_high` / `--nrb_top` | `3` | Must match the trained model |
| `--gan_type` | `vanilla` | Must match training for the model to instantiate consistently |
| `--device` | `cuda` | Compute device |
| `--key` | *(hardcoded)* | W&B key — **remove before publishing** |

Evaluation reports **PSNR**, **SSIM**, **LPIPS** (VGG backbone), and **MS-SSIM**, and writes qualitative triplets (input / output / reference) via `visualise()`.

## Model complexity

```bash
cd src
python flops.py
```

This reports MACs and parameter count for `LPTNPaper` at 224 × 224 using `ptflops`. Note `flops.py` instantiates the network with `nrb_low=4, nrb_high=4, num_high=2`; set these to match the configuration you report in the paper so the complexity figures correspond to your trained model.

## Reproducibility notes

Please read these before quoting numbers — several defaults do not behave as their names suggest.

- **`--lr` has no effect.** `setup_optimizers` hardcodes the generator optimizer to `SOAP(lr=1e-3, betas=(0.95, 0.95), weight_decay=0.01, precondition_frequency=10)` and each discriminator to `Adam(lr=1e-4)`. The `--lr` flag is stored but never used. Change the code (not just the flag) to sweep learning rates, and report the SOAP settings explicitly.
- **Pyramid depth is fixed.** `num_high = 2` is hardcoded in `LPTNModel`, giving three pyramid levels, and there are exactly three discriminators. `--levels` therefore only meaningfully ranges over `{0, 1, 2}`; supervising more levels requires adding discriminators.
- **LR schedule.** `MultiStepRestartLR` with milestones `[50000, 100000, 200000, 300000]` and `gamma=0.5` (iteration-based). Confirm this matches your epoch/iteration budget.
- **Determinism.** Only the dataset folder shuffle is seeded (`seed=42`). `torch`, `numpy`, and CUDA RNGs are not globally seeded, so runs are not bit-reproducible. For the camera-ready, add a seeding utility at startup, e.g.:

```python
  import torch, numpy as np, random
  def set_seed(s=42):
      random.seed(s); np.random.seed(s)
      torch.manual_seed(s); torch.cuda.manual_seed_all(s)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
```

- **Metric convention.** PSNR/SSIM are computed after clamping to `[0, 1]`, scaling to `[0, 255]`, and casting to `int` then back to `float` (i.e. on 8-bit-quantized images). LPIPS is computed on `[-1, 1]`-normalized inputs with the VGG backbone. Match this convention when comparing against other methods.
- **Import side effect.** `losses/metrics.py` runs an LPIPS demo and prints a loss value at import time (bottom of the file). It is harmless but noisy and downloads VGG weights; consider guarding it under `if __name__ == "__main__":`.
- **`--exposure` mismatch on validation.** As noted above, validation always uses `both`. Decide whether that is intended for your reported validation numbers.

## Results

Fill in with your measured numbers before submission. The metrics below are exactly the ones the evaluation code emits; do not carry over placeholder values.

| Test set | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MS-SSIM ↑ |
|---|---|---|---|---|
| SICE (over-exposed) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| SICE (under-exposed) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| SICE_Mix | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| SICE_Grad | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

## Before making the repository public

- [ ] **Revoke and remove** the Weights & Biases API keys in `train.py` and `eval.py`. Treat both as compromised.
- [ ] Add a pinned `requirements.txt` (or `environment.yml`).
- [ ] Replace the hardcoded `/kaggle/...` paths in `eval.py` / `evaluater.py` with `--root_dir` / `--model_path`.
- [ ] Add global seeding and, ideally, release the exact checkpoint used for the reported numbers.
- [ ] Remove one of the duplicated `tryinit.ipynb` files (or fold the useful parts into documented scripts).

## Citation

If you use this code, the method, or the checkpoints, please cite LapLoss (accepted at the **DeLTa Workshop, ICLR 2025**):

```bibtex
@article{didwania2025laploss,
  title   = {LapLoss: Laplacian Pyramid-based Multiscale Loss for Image Translation},
  author  = {Didwania, Krish and Gakhar, Ishaan and Arya, Prakhar and Labroo, Sanskriti},
  journal = {arXiv preprint arXiv:2503.05974},
  note    = {Accepted at the DeLTa Workshop, ICLR 2025},
  year    = {2025}
}
```

This work builds directly on the following; please also cite them where relevant:

```bibtex
@inproceedings{liang2021high,
  title     = {High-Resolution Photorealistic Image Translation in Real-Time: A Laplacian Pyramid Translation Network},
  author    = {Liang, Jie and Zeng, Hui and Zhang, Lei},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2021}
}

@article{kasliwal2024lapgsr,
  title   = {LapGSR: Laplacian Reconstructive Network for Guided Thermal Super-Resolution},
  author  = {Kasliwal, Aditya and Gakhar, Ishaan and Kamani, Aryan and Seth, Pratinav and Verma, Ujjwal},
  journal = {arXiv preprint arXiv:2411.07750},
  year    = {2024}
}

@article{vyas2024soap,
  title   = {SOAP: Improving and Stabilizing Shampoo using Adam},
  author  = {Vyas, Nikhil and Morwani, Depen and Zhao, Rosie and Kwun, Mujin and Shapira, Itai and Brandfonbrener, David and Janson, Lucas and Kakade, Sham},
  journal = {arXiv preprint arXiv:2409.11321},
  year    = {2024}
}

@inproceedings{zhang2018unreasonable,
  title     = {The Unreasonable Effectiveness of Deep Features as a Perceptual Metric},
  author    = {Zhang, Richard and Isola, Phillip and Efros, Alexei A. and Shechtman, Eli and Wang, Oliver},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2018}
}

@article{cai2018learning,
  title     = {Learning a Deep Single Image Contrast Enhancer from Multi-Exposure Images},
  author    = {Cai, Jianrui and Gu, Shuhang and Zhang, Lei},
  journal   = {IEEE Transactions on Image Processing},
  volume    = {27},
  number    = {4},
  pages     = {2049--2062},
  year      = {2018},
  publisher = {IEEE}
}

@article{zheng2022low,
  title   = {Low-Light Image and Video Enhancement: A Comprehensive Survey and Beyond},
  author  = {Zheng, Shen and Ma, Yiling and Pan, Jinqian and Lu, Changjie and Gupta, Gaurav},
  journal = {arXiv preprint arXiv:2212.10772},
  year    = {2022}
}
```

## Acknowledgements

This repository accompanies **"LapLoss: Laplacian Pyramid-based Multiscale Loss for Image Translation"** by Krish Didwania, Ishaan Gakhar, Prakhar Arya, and Sanskriti Labroo (Manipal Institute of Technology, Manipal Academy of Higher Education), accepted at the **DeLTa Workshop, ICLR 2025**. All four authors contributed equally.

The method builds on several prior works, whose authors we gratefully acknowledge: the **Laplacian Pyramid Translation Network (LPTN)** of Liang, Zeng, and Zhang (CVPR 2021) and the **LapGSR** guided-super-resolution network of Kasliwal, Gakhar, Kamani, Seth, and Verma (arXiv:2411.07750, 2024), which supply the pyramidal backbones; the **SOAP** optimizer of Vyas et al. (arXiv:2409.11321, 2024); the **LPIPS** perceptual metric of Zhang et al. (CVPR 2018); the **SICE** dataset of Cai, Gu, and Zhang (IEEE TIP 2018); and the **SICE_Grad / SICE_Mix** mixed-exposure benchmarks of Zheng, Ma, Pan, Lu, and Gupta (arXiv:2212.10772, 2022). All rights to third-party code and data remain with their original authors, and their use here is subject to the respective original licenses.
