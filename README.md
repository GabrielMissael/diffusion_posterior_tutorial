# Diffusion Posterior Sampling Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GabrielMissael/diffusion_posterior_tutorial/blob/master/notebooks/diffusion_posterior_sampling.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-View%20Notebook-black?logo=github)](https://github.com/GabrielMissael/diffusion_posterior_tutorial/blob/master/notebooks/diffusion_posterior_sampling.ipynb)

This repository contains a Jupyter tutorial on diffusion posterior sampling for astronomical inverse problems. It starts with PSF deconvolution and super-resolution, then extends the same Bayesian logic to strong-lensing source inference with a fixed lens model and a diffusion prior.

## Structure

- `notebooks/diffusion_posterior_sampling.ipynb`: main class notebook.
- `src/diffusion_posterior_tutorial/`: lightweight teaching code used by the notebook.
- `data/`: small local assets.
- `tests/`: smoke tests for operators, lensing, and sampler-history recording.

## Run Locally

```bash
cd diffusion_posterior_tutorial
python -m pip install -e .[test]
python -m pip install git+https://github.com/AlexandreAdam/score_models.git@dev
jupyter lab
```

Open `notebooks/diffusion_posterior_sampling.ipynb` and run it top to bottom.

The notebook setup cell also prepends `src/` to `sys.path`, so local imports work even in notebook kernels where `pip install -e .` is not immediately visible without a restart.

## Run In Colab

1. Click the Colab badge in the notebook or README.
2. Open `notebooks/diffusion_posterior_sampling.ipynb`.
3. Run the setup cell. It:
   - clones `https://github.com/GabrielMissael/diffusion_posterior_tutorial.git` into `/content/diffusion_posterior_tutorial` when running in Colab,
   - installs the local tutorial package from that clone,
   - `score_models` from GitHub,
   - `caustics==1.2.0`.
4. The notebook then downloads the pretrained galaxy diffusion prior automatically from Hugging Face.

## Pretrained Model

The notebook loads the pretrained diffusion prior from:

- Hugging Face repo: `GMissaelBarco/galaxy-prior`

The download is handled with `huggingface_hub.snapshot_download`, and the model is loaded through `ScoreModel` from the `score_models` package.

## Credits

- Workshop data tensors are reused from `super-resolution-workshop`.
- The strong-lensing section uses `caustics` for differentiable lensing simulation.
- The score prior is loaded through `score-models`.
