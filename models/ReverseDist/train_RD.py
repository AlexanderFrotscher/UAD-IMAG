__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from RD_model import ReverseDist
from torch import optim
from torchvision.transforms import v2
from tqdm import tqdm

import wandb
from conf import RD_config as config
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
    model = ReverseDist(conf).to(device)
    model = accelerator.prepare(model)
    model.train()

    optimizer = optim.AdamW(
        model.parameters(), lr=conf.optim.lr, betas=conf.optim.betas
    )
    optimizer, dataloader = accelerator.prepare(optimizer, dataloader)

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
                        small_loss = loss_fucntion_val(feat_h, rec_h)
                        loss_h.append(small_loss)

                        # lesion samples
                        feat_l, rec_l = model(samples_l)
                        small_loss = loss_fucntion_val(feat_l, rec_l)
                        loss_l.append(small_loss)

                    loss_h = torch.mean(torch.cat(loss_h))
                    loss_l = torch.mean(torch.cat(loss_l))
                model.train()
                logging.info(
                    "\n\tstep: %d, val-loss healthy: %.4f, val-loss lesion: %.4f"
                    % (step, loss_h.item(), loss_l.item())
                )
                wandb.log(
                   {"Cosine-Val-H": loss_h.item(), "Cosine-Val-L": loss_l.item()},
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

        features, rec = model(images)
        loss = loss_fucntion(features, rec)

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        if step % conf.training.log_freq == 0 and accelerator.is_main_process:
            logging.info(" step: %d, training_loss: %.4f" % (step, loss.item()))
            wandb.log({"Cosine": loss.item()}, step=step)

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


def loss_fucntion(a, b):
    # mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        # print(a[item].shape)
        # print(b[item].shape)
        # loss += 0.1*mse_loss(a[item], b[item])
        loss += torch.mean(
            1
            - cos_loss(
                a[item].view(a[item].shape[0], -1), b[item].view(b[item].shape[0], -1)
            )
        )
    return loss


def loss_fucntion_val(a, b):
    # mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = torch.zeros(a[0].shape[0]).to(
        a[0].device
    )  # vector with size equals batch_size
    for item in range(len(a)):
        # print(a[item].shape)
        # print(b[item].shape)
        # loss += 0.1*mse_loss(a[item], b[item])
        loss += 1 - cos_loss(
            a[item].view(a[item].shape[0], -1), b[item].view(b[item].shape[0], -1)
        )
    return loss


if __name__ == "__main__":
    main()
