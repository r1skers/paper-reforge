# MNIST MLP VAE

Experiment log for the first minimal VAE reproduction.

## Target

- Train an MLP VAE on MNIST.
- Save reconstruction samples during training.
- Generate new samples from $z \sim \mathcal{N}(0,I)$.

## Run 001

Config:

- latent_dim: 20
- hidden_dim: 400
- batch_size: 128
- epochs: 20
- learning_rate: 1e-3
- device: cpu

Final metrics:

- train loss: 104.1901
- train recon: 78.5278
- train kl: 25.6623
- test loss: 104.1232
- test recon: 78.8769
- test kl: 25.2463

Observations:

- Training was stable.
- Reconstruction improved quickly in the first few epochs.
- KL stabilized around 25.
- Test loss stayed close to train loss, so there is no obvious overfitting in this baseline.
- Generated samples are less sharp than reconstructions, which is expected for a first MLP VAE baseline.

## Latent Dimension Comparison

All runs use the same MLP architecture family and 20 training epochs.

| run | latent_dim | train loss | train recon | train kl | test loss | test recon | test kl | observation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| latent2_e20 | 2 | 150.0160 | 143.9665 | 6.0495 | 151.1283 | 144.9927 | 6.1356 | Reconstructions are recognizable but softer. Prior samples are surprisingly coherent and digit-like. |
| latent20_e20 | 20 | 104.1901 | 78.5278 | 25.6623 | 104.1232 | 78.8769 | 25.2463 | Good reconstruction. Prior samples show recognizable but blurry digit-like shapes. |
| latent50_e20 | 50 | 103.9198 | 77.7475 | 26.1723 | 103.7298 | 76.7843 | 26.9455 | Best quantitative reconstruction/ELBO among these runs, but prior samples are less visually coherent than latent_dim=2. |

Notes:

- Increasing latent dimension improves reconstruction and test negative ELBO from latent_dim=2 to latent_dim=20/50.
- Better reconstruction does not automatically imply better prior samples.
- The latent_dim=2 run has a much smaller KL term and a stronger bottleneck, which appears to produce a more compact and visually coherent prior-sampling space.
- The latent_dim=50 run has more capacity and better reconstruction, but sampling from the full high-dimensional prior can land in regions where the decoder is less visually stable.

### latent_dim=2 Visualization

Generated files:

- `runs/latent2_e20/visualizations/latent_scatter.png`
- `runs/latent2_e20/visualizations/latent_manifold.png`

![latent_dim=2 scatter](runs/latent2_e20/visualizations/latent_scatter.png)

![latent_dim=2 manifold](runs/latent2_e20/visualizations/latent_manifold.png)

Observations:

- The 2D latent scatter plot shows partial clustering by digit label, but with visible overlap between classes. This is expected because latent_dim=2 is a strong bottleneck.
- The latent manifold shows a smooth transition across digit-like regions. The left side is dominated by `1`/`7`-like shapes, the center transitions through `3`/`2`/`9`/`6`-like shapes, and the right side is dominated by `0`-like shapes.
- This supports the qualitative observation that the latent_dim=2 model learns a compact and continuous latent space, which helps prior sampling produce more coherent digit-like samples.
- The tradeoff is clear: latent_dim=2 has worse quantitative reconstruction metrics than latent_dim=20/50, but its 2D prior-sampling space is easier to visualize and appears more globally organized.
