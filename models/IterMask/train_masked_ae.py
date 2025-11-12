__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from masking import create_input
from torch import optim
from torchvision.transforms import v2
from tqdm import tqdm
from UNet import UNet

import wandb
from conf import IterMaskAE_config as config
from utils.dataloaders import MRI_Slices, MRI_Slices_val
from utils.ema import ExponentialMovingAverage
from utils.helpers import EarlyStopper, make_dicts, upload_images

logging.basicConfig(format="%(message)s", level=logging.INFO)


def train(conf):
    project_path = Path(os.path.dirname(__file__))
    project_path = os.fspath(project_path.parent.parent)
    make_dicts(project_path, conf.training.run_name)
    accelerator = Accelerator()
    device = accelerator.device
    transform = v2.Compose([v2.CenterCrop(conf.data.image_size)])
    dataloader = MRI_Slices(conf)
    model = UNet(conf).to(device)
    model = accelerator.prepare(model)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=conf.optim.lr)
    optimizer, dataloader = accelerator.prepare(optimizer, dataloader)

    stopper = EarlyStopper(conf.training.patience, conf.training.min_delta)

    ema = ExponentialMovingAverage(model.parameters(), decay=conf.model.ema_rate)
    accelerator.register_for_checkpointing(ema)
    start_step = 0

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

    loss_fn = nn.MSELoss()

    train_iter = iter(dataloader)
    end_step = start_step + conf.training.n_iters
    num_epochs = 0

    for step in tqdm(range(start_step, end_step), position=0):
        try:
            images = next(train_iter)
        except StopIteration:
            if (
                conf.training.early_stop and accelerator.is_main_process
            ):  # check if early stopping
                model.eval()
                with torch.no_grad():
                    dataloader_val_h, dataloader_val_l = MRI_Slices_val(conf)
                    loss_h, loss_l = [], []
                    for healthy_tensor, lesion_tensor in zip(
                        dataloader_val_h, dataloader_val_l
                    ):
                        healthy_tensor = healthy_tensor.to(device)
                        lesion_tensor = lesion_tensor.to(device)

                        # healthy samples
                        samples_h, input_h, _ = create_input(healthy_tensor, transform)
                        rec_h = model(input_h)
                        small_loss = torch.square(rec_h - samples_h)
                        small_loss = torch.mean(
                            small_loss.reshape(small_loss.shape[0], -1), dim=-1
                        )
                        loss_h.append(small_loss)

                        # lesion samples
                        samples_l, input_l, _ = create_input(lesion_tensor, transform)
                        rec_l = model(input_l)
                        small_loss = torch.square(rec_l - samples_l)
                        small_loss = torch.mean(
                            small_loss.reshape(small_loss.shape[0], -1), dim=-1
                        )
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

        images, input, _ = create_input(images, transform)
        rec = model(input.float())
        loss = loss_fn(images, rec)

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        ema.update(model.parameters())

        if step % conf.training.log_freq == 0 and accelerator.is_main_process:
            logging.info(" step: %d, training_loss: %.4f" % (step, loss.item()))
            wandb.log({"L2": loss.item()}, step=step)

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

            up_images = wandb.Image(
                upload_images(images[0 : conf.training.num_up], mode="L")
            )
            wandb.log({"Images": up_images})

            up_recon = wandb.Image(
                upload_images(rec[0 : conf.training.num_up].detach(), mode="L")
            )
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
