# Paper-to-Code Mapping

This note maps the VAE paper notation to the implementation.

## Generative Model

Paper:

$$p_{\theta}(z)p_{\theta}(x \mid z)$$

Code:

- prior: standard normal $p(z)=\mathcal{N}(0,I)$
- decoder: `Decoder(theta)`
- decoder output parameterizes $p_{\theta}(x \mid z)$

For MNIST, the first version will use a Bernoulli likelihood:

$$p_{\theta}(x \mid z)=\operatorname{Bernoulli}(x;\pi_{\theta}(z))$$

In code, `decoder(z)` returns pixel probabilities.

## Recognition Model / Encoder

Paper:

$$q_{\phi}(z \mid x)$$

Code:

- encoder: `Encoder(phi)`
- `encoder(x)` returns `mu` and `logvar`

Gaussian approximate posterior:

$$q_{\phi}(z \mid x)=\mathcal{N}\left(\mu_{\phi}(x),\operatorname{diag}(\sigma_{\phi}^{2}(x))\right)$$

## Reparameterization

Paper:

$$z=g_{\phi}(\epsilon,x), \qquad \epsilon \sim p(\epsilon)$$

Gaussian case:

$$z=\mu_{\phi}(x)+\sigma_{\phi}(x)\odot\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I)$$

Code:

```python
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + std * eps
```

## ELBO and Loss

Paper:

$$\mathcal{L}(\theta,\phi;x)=\mathbb{E}_{q_{\phi}(z \mid x)}[\log p_{\theta}(x \mid z)]-D_{\mathrm{KL}}(q_{\phi}(z \mid x)\|p(z))$$

Training usually minimizes negative ELBO:

$$\text{loss}=\text{reconstruction loss}+\text{KL loss}$$

For diagonal Gaussian posterior and standard normal prior:

$$D_{\mathrm{KL}}(q_{\phi}(z \mid x)\|p(z))=-\frac{1}{2}\sum_j\left(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2\right)$$

Code:

- reconstruction loss: binary cross entropy
- KL loss: closed-form diagonal Gaussian KL

