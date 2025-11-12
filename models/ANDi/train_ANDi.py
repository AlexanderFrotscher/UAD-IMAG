__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"


import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from DDPMpp import DDPMpp
from diffusion import get_sampling_fn, get_step_fn, my_VPSDE
from torch import optim
from torchvision.transforms import v2
from tqdm import tqdm

import wandb
from conf import ANDi_config as config
from utils.dataloaders import MRI_Slices, MRI_Slices_val
from utils.ema import ExponentialMovingAverage
from utils.helpers import EarlyStopper, make_dicts, upload_images

logging.basicConfig(format="%(message)s", level=logging.INFO)


def train(conf):
    project_path = Path(os.path.dirname(__file__))
    project_path = os.fspath(project_path.parent.parent)
    torch.manual_seed(conf.seed)
    make_dicts(project_path, conf.training.run_name)
    accelerator = Accelerator()
    device = accelerator.device
    transform = v2.Compose([v2.CenterCrop(conf.data.image_size)])
    dataloader = MRI_Slices(conf, transform)

    sde = my_VPSDE(
        conf.diffusion.schedule,
        conf.diffusion.num_latents,
        conf.model.pyramid,
        conf.model.discount,
        conf.diffusion.logsnr_min,
        conf.diffusion.logsnr_max,
        conf.data.image_size,
        shift=conf.diffusion.shift,
    )

    model = DDPMpp(conf)
    model = accelerator.prepare(model)  # needs to be called individually for FSDP
    # model = torch.compile(model, mode="reduce-overhead")
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=conf.optim.lr)
    optimizer, dataloader = accelerator.prepare(optimizer, dataloader)

    stopper = EarlyStopper(conf.training.patience, conf.training.min_delta)

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

    train_iter = iter(dataloader)
    train_step = get_step_fn(sde, vpred=conf.model.vpred)
    sampling_shape = (
        conf.sampling.num_images,
        conf.data.num_channels,
        conf.data.image_size,
        conf.data.image_size,
    )
    sampling_fn = get_sampling_fn(
        sde,
        sampling_shape,
        device,
        conf.sampling.steps,
        conf.sampling.sampler,
        conf.model.vpred,
    )

    end_step = start_step + conf.training.n_iters
    num_epochs = 1

    for step in tqdm(range(start_step, end_step), position=0):
        try:
            images = next(train_iter)
        except StopIteration:
            if (
                conf.training.early_stop and accelerator.is_main_process
            ):  # check if early stopping
                model.eval()
                with torch.no_grad():
                    dataloader_val_h, dataloader_val_l = MRI_Slices_val(conf, transform)
                    loss_h, loss_l = [], []
                    for samples_h, samples_l in zip(dataloader_val_h, dataloader_val_l):
                        samples_h = samples_h.to(device)
                        samples_h = (samples_h * 2) - 1
                        samples_l = samples_l.to(device)
                        samples_l = (samples_l * 2) - 1

                        # healthy samples
                        small_loss = train_step(model, samples_h)
                        loss_h.append(small_loss)

                        # lesion samples
                        small_loss = train_step(model, samples_l)
                        loss_l.append(small_loss)

                    loss_h = torch.cat(loss_h)
                    loss_l = torch.cat(loss_l)
                    loss_h = torch.mean(loss_h)
                    loss_l = torch.mean(loss_l)

                model.train()
                logging.info(
                    "\n\tstep: %d, val-loss healthy: %.4f, val-loss lesion: %.4f"
                    % (step, loss_h.item(), loss_l.item())
                )
                wandb.log(
                    {"L2-Val-H": loss_h.item(), "L2-Val-L": loss_l.item()},
                    step=step,
                )

                if stopper.early_stop(loss_l.item(), loss_h.item()):
                    if type(model) in (
                        nn.parallel.DataParallel,
                        nn.parallel.DistributedDataParallel,
                    ):
                        model.module.update_step(step, device)
                    else:
                        model.update_step(step, device)
                    accelerator.save_state(
                        f"{project_path}/checkpoints/{conf.training.run_name}/checkpoint_{conf.training.early_stop_model}"
                    )
                    logging.info("Early Stopper has ended training.")
                    sys.exit()
            train_iter = iter(dataloader)
            images = next(train_iter)
            num_epochs += 1
            logging.info(f"Starting epoch {num_epochs}")

        images = (images * 2) - 1
        # Execute one training step
        losses = train_step(model, images)
        loss = torch.mean(losses)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        ema.update(model.parameters())
        if step % conf.training.log_freq == 0 and accelerator.is_main_process:
            logging.info(" step: %d, training_loss: %.4f" % (step, loss.item()))
            wandb.log({"Loss": loss.item()}, step=step)

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
            sample = sampling_fn(model)
            sample = (sample + 1) / 2
            model.train()
            ema.restore(model.parameters())
            sampled_images = wandb.Image(upload_images(sample, mode="L"))
            wandb.log({"Sampled-Images": sampled_images})


def main():
    torch.backends.cudnn.benchmark = (
        True  # additional speed up if input size does not change
    )
    conf = config.get_config()
    wandb.init(entity="", project=conf.training.run_name)

    train(conf)


if __name__ == "__main__":
    main()
