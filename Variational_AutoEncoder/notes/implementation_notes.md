# Implementation Notes

## Current Plan

Start with the smallest useful VAE:

- dataset: MNIST
- architecture: MLP encoder and decoder
- latent dimension: 20
- likelihood: Bernoulli over pixels
- optimizer: Adam

## Conceptual Checklist

- Encoder outputs distribution parameters, not a deterministic latent code.
- Reparameterization keeps randomness but makes the path differentiable.
- Decoder maps latent samples to parameters of $p_{\theta}(x \mid z)$.
- The loss is negative ELBO.

## Questions to Track

- How does latent dimension affect reconstruction and generation?
- What changes when using latent dimension 2 for visualization?
- How do generated samples improve over training?

