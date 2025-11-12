__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import torch
from tqdm import tqdm

from utils.helpers import pyramid_noise_like


def ANDi(
    sde, model, batch, aggregation: str, vpred: bool, start: int = 125, stop: int = 25
):
    assert aggregation in ("gmean", "prob"), (
        "aggregation type needs to be gmean or prob"
    )
    my_range = start - stop
    with torch.no_grad():
        results = torch.zeros(
            batch.shape[0],
            my_range,
            batch.shape[1],
            batch.shape[2],
            batch.shape[3],
        ).to(batch.device)

        timesteps = torch.linspace(1, 1 / sde.N, (sde.N // 1), device=batch.device)
        timesteps = torch.cat([timesteps, torch.zeros(1, device=batch.device)])

        for i in tqdm(range(my_range), position=0):
            t = timesteps[(sde.N - start) + i]
            s = timesteps[(sde.N - start) + i + 1]
            vec_t = torch.ones(batch.shape[0], device=t.device) * t
            vec_s = torch.ones(batch.shape[0], device=s.device) * s

            log_snr_t = sde(vec_t)
            log_snr_s = sde(vec_s)
            alpha_t = torch.sqrt(torch.sigmoid(log_snr_t))
            sigma_t = torch.sqrt(torch.sigmoid(-log_snr_t))
            alpha_s = torch.sqrt(torch.sigmoid(log_snr_s))
            alpha_st = torch.sqrt(
                (1 + torch.exp(-log_snr_t)) / (1 + torch.exp(-log_snr_s))
            )
            r = torch.exp(log_snr_t - log_snr_s)
            one_minus_r = -torch.expm1(log_snr_t - log_snr_s)

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

            z_t = (
                alpha_t[:, None, None, None] * batch
                + sigma_t[:, None, None, None] * noise
            )
            pred = model(z_t, vec_t)

            if vpred:
                x_pred = (
                    alpha_t[:, None, None, None] * z_t
                    - sigma_t[:, None, None, None] * pred
                )
            else:
                x_pred = (z_t - sigma_t[:, None, None, None] * pred) / alpha_t[
                    :, None, None, None
                ]

            x_pred = torch.clamp(x_pred, -1, 1)

            x_q = (
                r[:, None, None, None] * alpha_st[:, None, None, None] * z_t
                + one_minus_r[:, None, None, None]
                * alpha_s[:, None, None, None]
                * batch
            )

            x_mean = (
                r[:, None, None, None] * alpha_st[:, None, None, None] * z_t
                + one_minus_r[:, None, None, None]
                * alpha_s[:, None, None, None]
                * x_pred
            )

            if aggregation == "prob":
                var = one_minus_r * torch.sigmoid(-log_snr_t)
                results[:, i] = torch.exp(
                    (-(torch.square(x_mean - x_q)) / (2 * var[:, None, None, None]))
                )
            elif aggregation == "gmean":
                results[:, i] = torch.square(x_mean - x_q)

    return results
