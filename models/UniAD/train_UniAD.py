__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch import optim
from torchvision.transforms import v2
from tqdm import tqdm
from UniAD import UniAD

import wandb
from conf import UniAD_config as config
from utils.dataloaders import MRI_Slices, MRI_Slices_val
from utils.helpers import EarlyStopper, make_dicts

logging.basicConfig(format="%(message)s", level=logging.INFO)


def train(conf):
    project_path = Path(os.path.dirname(__file__))
    project_path = os.fspath(project_path.parent.parent)
    make_dicts(project_path, conf.training.run_name)
    accelerator = Accelerator()
    device = accelerator.device
    transform = v2.Compose([v2.CenterCrop(conf.data.image_size)])
    dataloader = MRI_Slices(conf, transform)
    model = UniAD(conf).to(device)
    model = accelerator.prepare(model)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=conf.optim.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, conf.optim.scheduler_step)
    optimizer, scheduler, dataloader = accelerator.prepare(
        optimizer, scheduler, dataloader
    )

    stopper = EarlyStopper(conf.training.patience, conf.training.min_delta)
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
    end_step = start_step + conf.training.n_iters
    num_epochs = 1

    loss_fn = nn.MSELoss()

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
                        samples_l = samples_l.to(device)

                        # healthy samples
                        feat_h, rec_h = model(samples_h)
                        small_loss = torch.square(rec_h - feat_h)
                        small_loss = torch.mean(
                            small_loss.reshape(small_loss.shape[0], -1), dim=-1
                        )
                        loss_h.append(small_loss)

                        # lesion samples
                        feat_l, rec_l = model(samples_l)
                        small_loss = torch.square(rec_l - feat_l)
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
            scheduler.step(num_epochs)

        features, rec = model(images)
        loss = loss_fn(features, rec)

        optimizer.zero_grad()
        accelerator.backward(loss)
        if conf.accelerator.clip_grad_norm:
            accelerator.clip_grad_norm_(
                model.parameters(), conf.accelerator.clip_grad_norm
            )
        optimizer.step()

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


def main():
    torch.backends.cudnn.benchmark = (
        True  # additional speed up if input size does not change
    )
    conf = config.get_config()
    wandb.init(entity="", project=conf.training.run_name)
    train(conf)


if __name__ == "__main__":
    main()
