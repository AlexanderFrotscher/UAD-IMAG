__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import functools
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from diffusion import VPSDE, get_step_fn, sampler
from preprocessing import create_dataloader
from torch import optim
from tqdm import tqdm

import wandb
from conf import Disyre_config as config
from models.ANDi.DDPMpp import DDPMpp
from utils.ema import ExponentialMovingAverage
from utils.helpers import make_dicts, upload_images

logging.basicConfig(format="%(message)s", level=logging.INFO)


def train(conf):
    project_path = Path(os.path.dirname(__file__))
    project_path = os.fspath(project_path.parent.parent)
    make_dicts(project_path, conf.training.run_name)
    accelerator = Accelerator()
    device = accelerator.device

    dataloader = create_dataloader(conf)

    sde = VPSDE(
        conf.diffusion.beta_min, conf.diffusion.beta_max, conf.diffusion.num_latents
    )

    model = DDPMpp(conf).to(device)
    model = accelerator.prepare(model)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=conf.optim.lr)
    optimizer, dataloader = accelerator.prepare(optimizer, dataloader)

    ema = ExponentialMovingAverage(model.parameters(), decay=conf.model.ema_rate)
    accelerator.register_for_checkpointing(ema)

    start_step = 1

    if conf.accelerator.train_continue:
        accelerator.load_state(
            f"{project_path}/checkpoints/{conf.training.run_name}/checkpoint_{conf.accelerator.checkpoint}"
        )
        if type(model) in (
            nn.parallel.DataParallel,
            nn.parallel.DistributedDataParallel,
        ):
            start_step = model.module.step.int().item() + 1
        else:
            start_step = model.step.int().item() + 1

    end_step = start_step + conf.training.n_iters

    train_step = functools.partial(get_step_fn, sde)

    alphas_keys = []
    if conf.data.anom_type in [
        "dag",
        "fpi",
        "dag_no_quant",
    ]:
        alphas_keys.append("alpha_texture")
    if conf.data.anom_type in ["dag", "dag_no_quant", "bias_only"]:
        alphas_keys.append("alpha_bias")

    for step in tqdm(range(start_step, end_step), position=0):
        data_dict = next(dataloader)

        anom_image = data_dict["data_c"].to(accelerator.device)
        image = data_dict["data"].to(accelerator.device)

        for alpha_n, alpha_k in enumerate(alphas_keys):
            if alpha_n == 0:
                alpha_dag = data_dict[alpha_k].to(torch.float32)
            else:
                alpha_dag = torch.stack(
                    [alpha_dag, data_dict[alpha_k].to(torch.float32)]
                )
                alpha_dag = torch.amax(alpha_dag, dim=0)

        image = (image * 2) - 1
        anom_image = (anom_image * 2) - 1
        step_fn = train_step(alpha_dag)
        loss = step_fn(model, image, anom_image)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        ema.update(model.parameters())

        if step % conf.training.log_freq == 0 and accelerator.is_main_process:
            logging.info(" step: %d, training_loss: %.4f" % (step, loss.item()))
            wandb.log({"L2": loss.item()}, step = step)

        if (
            step != start_step
            and accelerator.is_main_process
            and step % conf.training.snapshot_freq == 0
            or step == conf.training.n_iters
        ):
            save_step = step // conf.training.snapshot_freq
            if type(model) in (
                nn.parallel.DataParallel,
                nn.parallel.DistributedDataParallel,
            ):
                model.module.update_step(step, device)
            else:
                model.update_step(step, device)
            accelerator.save_state(
                f"{project_path}/checkpoints/{conf.training.run_name}/checkpoint_{save_step}"
            )

            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            model.eval()
            prediction, x_0 = sampler(
                sde,
                model,
                anom_image[0 : conf.sampling.num_up].detach(),
                conf.sampling.n_steps_each,
                device,
            )
            model.train()
            ema.restore(model.parameters())

            up_images = wandb.Image(upload_images(image[0:conf.sampling.num_up],mode='L'))
            wandb.log({"Images": up_images})

            up_recon = wandb.Image(upload_images(x_0,mode='L'))
            wandb.log({"Reconstructions": up_recon})


def main():
    torch.backends.cudnn.benchmark = (
        True  # additional speed up if input size does not change
    )
    conf = config.get_config()
    wandb.init(entity="", project=conf.training.run_name)
    train(conf)


if __name__ == "__main__":
    main()
