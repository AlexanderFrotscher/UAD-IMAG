__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import logging
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from gms_loss import MSGMS_Loss
from masking import gen_mask
from torch import optim
from torchvision.transforms import v2
from tqdm import tqdm

import wandb
from conf import RIAD_config as config
from models.ANDi.DDPMpp import DDPMpp
from models.FAE.pytorch_ssim import my_SSIMLoss
from utils.dataloaders import MRI_Slices, MRI_Slices_val
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

    model = DDPMpp(conf)
    model = accelerator.prepare(model)  # needs to be called individually for FSDP
    # model = torch.compile(model, mode="reduce-overhead")
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=conf.optim.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 7)
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

    ssim = my_SSIMLoss(window_size=11)
    mse = nn.MSELoss(reduction="mean")
    msgms = MSGMS_Loss()

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
                        k_value = random.sample(conf.training.k, 1)
                        Ms_generator = gen_mask(
                            k_value, conf.training.n, conf.data.image_size
                        )
                        Ms = next(Ms_generator)
                        inputs = [
                            samples_h
                            * (torch.tensor(mask, requires_grad=False).to(device))
                            for mask in Ms
                        ]
                        outputs = [model(x) for x in inputs]
                        output = sum(
                            map(
                                lambda x, y: x
                                * (torch.tensor(1 - y, requires_grad=False).to(device)),
                                outputs,
                                Ms,
                            )
                        )
                        l2_loss = mse(samples_h, output)
                        gms_loss = msgms(samples_h, output)
                        ssim_loss = ssim(samples_h, output)
                        small_loss = (
                            conf.training.gamma * l2_loss
                            + conf.training.alpha * gms_loss
                            + conf.training.beta * ssim_loss
                        )
                        loss_h.append(small_loss)

                        # lesion samples
                        k_value = random.sample(conf.training.k, 1)
                        Ms_generator = gen_mask(
                            k_value, conf.training.n, conf.data.image_size
                        )
                        Ms = next(Ms_generator)
                        inputs = [
                            samples_l
                            * (torch.tensor(mask, requires_grad=False).to(device))
                            for mask in Ms
                        ]
                        outputs = [model(x) for x in inputs]
                        output = sum(
                            map(
                                lambda x, y: x
                                * (torch.tensor(1 - y, requires_grad=False).to(device)),
                                outputs,
                                Ms,
                            )
                        )
                        l2_loss = mse(samples_l, output)
                        gms_loss = msgms(samples_l, output)
                        ssim_loss = ssim(samples_l, output)
                        small_loss = (
                            conf.training.gamma * l2_loss
                            + conf.training.alpha * gms_loss
                            + conf.training.beta * ssim_loss
                        )
                        loss_l.append(small_loss)

                    loss_h = torch.stack(loss_h)
                    loss_l = torch.stack(loss_l)
                    loss_h = torch.mean(loss_h)
                    loss_l = torch.mean(loss_l)

                model.train()
                logging.info(
                    "\n\tstep: %d, val-loss healthy: %.4f, val-loss lesion: %.4f"
                    % (step, loss_h.item(), loss_l.item())
                )
                wandb.log(
                    {"Loss-Val-H": loss_h.item(), "Loss-Val-L": loss_l.item()},
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

        k_value = random.sample(conf.training.k, 1)
        Ms_generator = gen_mask(k_value, conf.training.n, conf.data.image_size)
        Ms = next(Ms_generator)

        inputs = [
            images * (torch.tensor(mask, requires_grad=False).to(device)) for mask in Ms
        ]
        outputs = [model(x) for x in inputs]
        output = sum(
            map(
                lambda x, y: x * (torch.tensor(1 - y, requires_grad=False).to(device)),
                outputs,
                Ms,
            )
        )

        l2_loss = mse(images, output)
        gms_loss = msgms(images, output)
        ssim_loss = ssim(images, output)

        loss = (
            conf.training.gamma * l2_loss
            + conf.training.alpha * gms_loss
            + conf.training.beta * ssim_loss
        )

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

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

            up_images = wandb.Image(
                upload_images(images[0 : conf.training.num_up], mode="L")
            )
            wandb.log({"Images": up_images})

            up_recon = wandb.Image(
                upload_images(output[0 : conf.training.num_up].detach(), mode="L")
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
