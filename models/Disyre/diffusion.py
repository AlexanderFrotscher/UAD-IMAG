__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import torch
from tqdm import tqdm


class VPSDE:
    def __init__(
        self,
        beta_min=0.1,
        beta_max=20,
        N=1000,
    ):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__()
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)


def get_step_fn(sde, alpha):
    """Create a one-step training/evaluation function.
    Returns:
      A one-step function for training.
    """
    loss_fn = get_ddpm_loss_fn(sde, alpha)
    return loss_fn


def get_ddpm_loss_fn(vpsde, alpha):
    """DDPM training that estimates x_0 for Disyre."""

    assert isinstance(vpsde, VPSDE)

    def loss_fn(model, image, anom_image):
        # labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
        labels = (
            (vpsde.sqrt_alphas_cumprod[None] - alpha[:, None])
            .abs()
            .argmin(dim=1)
            .to(image.device)
        )
        # sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(image.device)
        # sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(image.device)
        # noise = torch.randn_like(batch)
        # perturbed_data = (
        #    sqrt_alphas_cumprod[labels, None, None, None] * batch
        #    + sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
        # )
        rec = model(anom_image, labels)
        losses = torch.square(rec - image)
        losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
        loss = torch.mean(losses)
        return loss

    return loss_fn


def sampler(sde, model, batch, n_steps, device):
    """The sampler/converter function.

    Args:
        model: A disyre model.
    Returns:
        Samples
    """
    with torch.no_grad():
        predictor = DisyrePipeline(sde)
        timesteps = torch.arange(sde.N, 0, -(sde.N // n_steps), device=device)
        timesteps -= 1
        timesteps = torch.cat([timesteps, torch.zeros(1, device=device)])
        x = batch
        for i in tqdm(range(len(timesteps) - 1)):
            t = timesteps[i]
            s = timesteps[i + 1]
            vec_t = torch.ones(batch.shape[0], device=t.device) * t
            vec_s = torch.ones(batch.shape[0], device=t.device) * s
            x, x_0 = predictor.update_fn(x, vec_t, vec_s, model)
    return x, x_0


class DisyrePipeline:
    """The Disyre converter/sampler."""

    def __init__(
        self,
        sde,
    ):
        super().__init__()
        self.sde = sde

        if not isinstance(sde, VPSDE):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

    def vpsde_update_fn(self, x, t, s, model):
        sde = self.sde
        alpha = sde.sqrt_alphas_cumprod.to(t.device).flip(0)[t.long()]
        alpha_m1 = sde.sqrt_alphas_cumprod.to(s.device).flip(0)[s.long()]
        x_0 = model(x, t)
        xT_bar = (x - x_0 * (1 - alpha[:, None, None, None])) / alpha[
            :, None, None, None
        ]
        xt_bar = (
            x_0 * (1 - alpha[:, None, None, None]) + alpha[:, None, None, None] * xT_bar
        )
        xt_sub1_bar = (
            x_0 * (1 - alpha_m1[:, None, None, None])
            + alpha_m1[:, None, None, None] * xT_bar
        )
        pred_prev_sample = x - xt_bar + xt_sub1_bar
        pred_prev_sample.clamp_(-1.0, 1.0)
        return pred_prev_sample, x_0

    def update_fn(self, x, t, s, model):
        return self.vpsde_update_fn(x, t, s, model)
