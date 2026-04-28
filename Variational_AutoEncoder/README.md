# Variational AutoEncoder

Reproduction of **Auto-Encoding Variational Bayes** on MNIST.

Paper: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

## Goal

This project focuses on the core VAE mechanism:

$$x \rightarrow q_{\phi}(z \mid x) \rightarrow z \rightarrow p_{\theta}(x \mid z)$$

The first implementation target is a minimal MLP VAE on MNIST, with:

- encoder outputting $\mu_{\phi}(x)$ and $\log\sigma_{\phi}^{2}(x)$
- reparameterization trick
- ELBO training objective
- sample generation from $z \sim \mathcal{N}(0,I)$
- optional 2D latent-space visualization

## Planned Structure

```text
configs/
  mnist_mlp.yaml
src/
  models/
    vae_mlp.py
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
    checkpoints/
    logs/
    samples/
```

## First Milestone

Train a small MLP VAE on MNIST and generate samples from the learned decoder.

