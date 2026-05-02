# Variational AutoEncoder

Minimal PyTorch reproduction of **Auto-Encoding Variational Bayes** on MNIST.

Paper: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

## What This Reproduces

This project implements the core VAE pipeline:

```math
x \rightarrow q_{\phi}(z \mid x) \rightarrow z \rightarrow p_{\theta}(x \mid z)
```

The current baseline is an MLP VAE with:

- MNIST input flattened to 784 dimensions
- encoder outputting `mu` and `logvar`
- Gaussian reparameterization
- Bernoulli decoder over pixels
- negative ELBO loss: reconstruction BCE + KL
- latent dimension comparison: 2, 20, 50
- 2D latent scatter and manifold visualization

The next comparison model is a CNN VAE with:

- MNIST kept as `[batch, channel, height, width]` image tensors in the encoder
- strided convolution downsampling from `28x28` to `14x14` to `7x7`
- `mu` and `logvar` predicted from the flattened convolutional feature map
- transposed convolution upsampling from `7x7` to `14x14` to `28x28`
- the same negative ELBO objective as the MLP baseline

## Project Layout

```text
configs/
  mnist_mlp.yaml
  mnist_cnn.yaml
src/
  models/vae_mlp.py
  models/vae_cnn.py
  data.py
  losses.py
  train.py
  generate.py
  visualize_latent.py
  utils.py
notes/
  paper_mapping.md
  implementation_notes.md
experiments/
  mnist_mlp/
    README.md
    runs/
      latent2_e20/
      latent20_e20/
      latent50_e20/
```

README guide:

- This file is the project entry point.
- `experiments/mnist_mlp/README.md` records experiment settings, metrics, and visual observations.
- `notes/paper_mapping.md` maps paper notation to the implementation.
- `notes/implementation_notes.md` keeps short implementation reminders.

## Setup

```powershell
pip install -r requirements.txt
```

## Train

Edit `configs/mnist_mlp.yaml` to choose `latent_dim` and `run_name`, then run the MLP baseline:

```powershell
python -m src.train --config configs/mnist_mlp.yaml
```

Run the CNN VAE comparison with:

```powershell
python -m src.train --config configs/mnist_cnn.yaml
```

Outputs are written to:

```text
experiments/<experiment_name>/runs/<run_name>/
  checkpoints/
  logs/
  samples/
```

## Generate Samples

```powershell
python -m src.generate --checkpoint experiments\mnist_mlp\runs\latent20_e20\checkpoints\latest.pt --output experiments\mnist_mlp\runs\latent20_e20\samples\generated_from_checkpoint.png --device cpu
```

## Visualize 2D Latent Space

```powershell
python -m src.visualize_latent --checkpoint experiments\mnist_mlp\runs\latent2_e20\checkpoints\latest.pt --output-dir experiments\mnist_mlp\runs\latent2_e20\visualizations --device cpu
```

## Results

See [experiments/mnist_mlp/README.md](experiments/mnist_mlp/README.md).

Summary:

- The MLP VAE trains stably on CPU.
- Reconstructions are clear after 20 epochs.
- Prior samples are recognizable but blurry, which is expected for this baseline.
- Increasing latent dimension improves reconstruction metrics, but does not automatically improve prior sample coherence.
- `latent_dim=2` learns a compact and visually interpretable latent space.

## Notes

- [Paper-to-code mapping](notes/paper_mapping.md)
- [Implementation notes](notes/implementation_notes.md)
