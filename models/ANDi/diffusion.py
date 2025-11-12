__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import torch
from tqdm import tqdm

from utils.helpers import pyramid_noise_like


class classic_schedule:
    def __init__(self):
        super().__init__()

    # original DDPM scaling which was called linear, but this schedule is not linear when considering the SNR
    def logsnr_schedule_classic(self, t):
        gamma = -torch.log(
            torch.expm1(1e-4 + 10 * t**2)
        )  # already the truncated version
        return gamma

    def get_logsnr(self, t):
        return self.logsnr_schedule_classic(t)


class cosine_schedule:
    def __init__(
        self, logsnr_min: int, logsnr_max: int, image_d: int, noise_d=56, shift=True
    ) -> None:
        super().__init__()
        self.logsnr_min = logsnr_min
        self.logsnr_max = logsnr_max
        self.shift = shift
        self.image_d = image_d
        self.noise_d = noise_d  # reference image dimension, for us 56 instead of 64

    def logsnr_schedule_cosine(self, t):
        logsnr_min = torch.ones(1).to(t.device) * self.logsnr_min
        logsnr_max = torch.ones(1).to(t.device) * self.logsnr_max
        t_min = torch.atan(torch.exp(-0.5 * logsnr_max))
        t_max = torch.atan(torch.exp(-0.5 * logsnr_min))
        gamma = -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))
        return gamma

    def logsnr_schedule_cosine_shifted(self, t):
        image_d = torch.ones(1).to(t.device) * self.image_d
        noise_d = torch.ones(1).to(t.device) * self.noise_d
        return self.logsnr_schedule_cosine(t) + 2 * torch.log(noise_d / image_d)

    def get_logsnr(self, t):
        if self.shift:
            return self.logsnr_schedule_cosine_shifted(t)
        else:
            return self.logsnr_schedule_cosine(t)


class my_VPSDE:
    diffusion_settings = {"classic": classic_schedule, "cosine": cosine_schedule}

    def __init__(
        self,
        schedule: str,
        N=1000,
        pyramid=False,
        discount=0.8,
        logsnr_min: int = None,
        logsnr_max: int = None,
        image_d: int = None,
        noise_d: int = 56,
        shift: bool = True,
    ):
        super().__init__()
        self.N = N
        self.pyramid = pyramid
        self.discount = discount
        if schedule not in self.diffusion_settings:
            raise KeyError(
                f"invalid schedule {schedule} for specifying the diffusion. "
                "You can use the string classic or cosine."
            )

        schedule_fn = self.diffusion_settings[schedule]

        if schedule == "cosine":
            assert logsnr_min is not None, "cosine schedule needs logsnr_min"
            assert logsnr_max is not None, "cosine schedule needs logsnr_max"
            assert image_d is not None, "cosine schedule needs image_d"

            self.schedule = schedule_fn(
                logsnr_min=logsnr_min,
                logsnr_max=logsnr_max,
                image_d=image_d,
                noise_d=noise_d,
                shift=shift,
            )
        else:
            self.schedule = schedule_fn()

    def __call__(self, t):
        return self.schedule.get_logsnr(t)

    def prior_sampling(self, shape):
        if self.pyramid:
            return pyramid_noise_like(
                shape[0], shape[1], shape[2], discount=self.discount, device="cpu"
            )
        else:
            return torch.randn(*shape)


def get_step_fn(sde, vpred: bool = True):
    """Create a one-step training/evaluation function.

    Returns:
      A one-step function for training.
    """

    if vpred:
        loss_fn = vpred_loss(sde)
    else:
        loss_fn = noise_loss(sde)

    return loss_fn


def vpred_loss(sde):
    def loss_fn(model, batch):
        t = torch.rand(batch.shape[0], device=batch.device)
        if sde.pyramid:
            noise = pyramid_noise_like(
                batch.shape[0],
                batch.shape[1],
                batch.shape[2],
                discount=sde.discount,
                device=batch.device,
            )
        else:
            noise = torch.randn_like(batch)

        alpha_t = torch.sqrt(torch.sigmoid(sde(t)))
        sigma_t = torch.sqrt(torch.sigmoid(-sde(t)))

        z_t = (
            alpha_t[:, None, None, None] * batch + sigma_t[:, None, None, None] * noise
        )  # z_t is the noisy latent

        v_pred = model(z_t, t)
        eps_pred = (
            sigma_t[:, None, None, None] * z_t + alpha_t[:, None, None, None] * v_pred
        )
        losses = torch.square(eps_pred - noise)
        # therefore this loss uses a weighting equal to the eps weighting
        losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
        return losses

    return loss_fn


def noise_loss(sde):
    def loss_fn(model, batch):
        t = torch.rand(
            batch.shape[0], device=batch.device
        )  # this is already continous training
        if sde.pyramid:
            noise = pyramid_noise_like(
                batch.shape[0],
                batch.shape[1],
                batch.shape[2],
                discount=sde.discount,
                device=batch.device,
            )
        else:
            noise = torch.randn_like(batch)

        alpha_t = torch.sqrt(torch.sigmoid(sde(t)))
        sigma_t = torch.sqrt(torch.sigmoid(-sde(t)))

        z_t = (
            alpha_t[:, None, None, None] * batch + sigma_t[:, None, None, None] * noise
        )  # z_t is the noisy latent

        noise_pred = model(z_t, t)
        losses = torch.square(noise_pred - noise)
        # original weighting
        losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
        return losses

    return loss_fn


def get_sampling_fn(
    sde, shape: tuple, device: str, steps: int, sampler: str, vpred: bool = True
):
    """Create a sampling function."""

    def sampling_fn(model):
        """The sampler funciton.

        Args:
          model: A DDPM model.
        Returns:
          Samples
        """
        assert sampler in ("Ancestral", "DDIM"), (
            "The sampling strategy needs to be Ancestral or DDIM"
        )
        if sampler == "Ancestral":
            sampling = AncestralSampling(sde, model, vpred)
        elif sampler == "DDIM":
            sampling = DDIMSampling(sde, model, vpred)
        with torch.no_grad():
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            timesteps = torch.linspace(1, 1 / sde.N, (sde.N // steps), device=device)
            timesteps = torch.cat([timesteps, torch.zeros(1, device=device)])

            for i in tqdm(range(len(timesteps) - 1), position=0):
                t = timesteps[i]
                s = timesteps[i + 1]
                vec_t = torch.ones(shape[0], device=t.device) * t
                vec_s = torch.ones(shape[0], device=s.device) * s
                x, x_mean = sampling.update_fn(x, vec_t, vec_s)

            return x_mean

    return sampling_fn


class AncestralSampling:
    """Ancestral sampling."""

    def __init__(self, sde, model, vpred):
        super().__init__()
        self.sde = sde
        self.model = model
        self.vpred = vpred

    def update_fn(self, x, t, s):
        pred = self.model(x, t)
        log_snr_t = self.sde(t)
        log_snr_s = self.sde(s)
        alpha_t = torch.sqrt(torch.sigmoid(log_snr_t))
        sigma_t = torch.sqrt(torch.sigmoid(-log_snr_t))
        alpha_s = torch.sqrt(torch.sigmoid(log_snr_s))
        alpha_st = torch.sqrt((1 + torch.exp(-log_snr_t)) / (1 + torch.exp(-log_snr_s)))
        r = torch.exp(log_snr_t - log_snr_s)
        one_minus_r = -torch.expm1(log_snr_t - log_snr_s)

        if self.vpred:
            x_pred = (
                alpha_t[:, None, None, None] * x - sigma_t[:, None, None, None] * pred
            )
        else:
            x_pred = (x - sigma_t[:, None, None, None] * pred) / alpha_t[
                :, None, None, None
            ]
        # Clip here now in x space
        x_pred = torch.clamp(x_pred, -1, 1)

        x_mean = (
            r[:, None, None, None] * alpha_st[:, None, None, None] * x
            + one_minus_r[:, None, None, None] * alpha_s[:, None, None, None] * x_pred
        )
        if not self.sde.pyramid:
            noise = torch.randn_like(x)
        else:
            noise = pyramid_noise_like(
                x.shape[0],
                x.shape[1],
                x.shape[2],
                discount=self.sde.discount,
                device=x.device,
            )
        var = one_minus_r * torch.sigmoid(-log_snr_t)
        x = x_mean + torch.sqrt(var[:, None, None, None]) * noise
        return x, x_mean


class DDIMSampling:
    def __init__(self, sde, model, vpred):
        super().__init__()
        self.sde = sde
        self.model = model
        self.vpred = vpred

    def update_fn(self, x, t, s):
        pred = self.model(x, t)
        log_snr_t = self.sde(t)
        log_snr_s = self.sde(s)
        alpha_t = torch.sqrt(torch.sigmoid(log_snr_t))
        sigma_t = torch.sqrt(torch.sigmoid(-log_snr_t))
        alpha_s = torch.sqrt(torch.sigmoid(log_snr_s))
        sigma_s = torch.sqrt(torch.sigmoid(-log_snr_s))

        if self.vpred:
            x_pred = (
                alpha_t[:, None, None, None] * x - sigma_t[:, None, None, None] * pred
            )
            eps_pred = (
                sigma_t[:, None, None, None] * x + alpha_t[:, None, None, None] * pred
            )
        else:
            x_pred = (x - sigma_t[:, None, None, None] * pred) / alpha_t[
                :, None, None, None
            ]
            eps_pred = pred
        # Clip here now in x space
        x_pred = torch.clamp(x_pred, -1, 1)
        x = (
            alpha_s[:, None, None, None] * x_pred
            + sigma_s[:, None, None, None] * eps_pred
        )
        return x, x_pred
