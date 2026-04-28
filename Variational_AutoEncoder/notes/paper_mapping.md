# Paper-to-Code Mapping

This note maps the notation in **Auto-Encoding Variational Bayes** to this implementation.

## Generative Model

Paper:

```math
p_{\theta}(z)p_{\theta}(x \mid z)
```

Code:

- prior: standard normal `p(z) = N(0, I)`
- decoder: `Decoder(theta)`
- decoder output parameterizes `p_theta(x | z)`

For MNIST, the first version uses a Bernoulli likelihood:

```math
p_{\theta}(x \mid z)=\prod_{j=1}^{784}\operatorname{Bernoulli}(x_j;\pi_{\theta,j}(z))
```

In code, `decoder(z)` returns pixel probabilities:

```math
\pi_{\theta}(z) \in [0,1]^{784}
```

## Recognition Model / Encoder

Paper:

```math
q_{\phi}(z \mid x)
```

Code:

- encoder: `Encoder(phi)`
- `encoder(x)` returns `mu` and `logvar`

Gaussian approximate posterior:

```math
q_{\phi}(z \mid x)=\mathcal{N}\left(\mu_{\phi}(x),\operatorname{diag}(\sigma_{\phi}^{2}(x))\right)
```

## Reparameterization

Paper:

```math
z=g_{\phi}(\epsilon,x), \qquad \epsilon \sim p(\epsilon)
```

Gaussian case:

```math
z=\mu_{\phi}(x)+\sigma_{\phi}(x)\odot\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I)
```

Code:

```python
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + std * eps
```

## ELBO and Loss

Paper:

```math
\mathcal{L}(\theta,\phi;x)
=
\mathbb{E}_{q_{\phi}(z \mid x)}[\log p_{\theta}(x \mid z)]
-
D_{\mathrm{KL}}(q_{\phi}(z \mid x)\|p(z))
```

Training minimizes the negative ELBO:

```math
\text{loss}=\text{reconstruction loss}+\text{KL loss}
```

For the Bernoulli decoder, the reconstruction loss is binary cross entropy:

```math
-\log p_{\theta}(x \mid z)
=
-\sum_{j=1}^{784}
\left[
x_j\log \pi_{\theta,j}(z)
+
(1-x_j)\log(1-\pi_{\theta,j}(z))
\right]
```

For diagonal Gaussian posterior and standard normal prior:

```math
D_{\mathrm{KL}}(q_{\phi}(z \mid x)\|p(z))
=
-\frac{1}{2}\sum_j
\left(
1+\log\sigma_j^2-\mu_j^2-\sigma_j^2
\right)
```

Code:

- reconstruction loss: `torch.nn.functional.binary_cross_entropy`
- KL loss: closed-form diagonal Gaussian KL

