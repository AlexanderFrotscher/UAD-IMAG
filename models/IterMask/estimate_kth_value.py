__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"


import os
from pathlib import Path

import torch
from accelerate import Accelerator
from masking import create_input
from torchvision.transforms import v2
from tqdm import tqdm
from UNet import UNet

from conf import IterMask_eval as config
from utils.dataloaders import MRI_Slices
from utils.ema import ExponentialMovingAverage


def main():
    project_path = Path(os.path.dirname(__file__))
    project_path = os.fspath(project_path.parent.parent)
    conf = config.get_config()
    torch.manual_seed(conf.seed)
    accelerator = Accelerator()
    transform = v2.Compose([v2.CenterCrop(conf.data.image_size)])
    model = UNet(conf)
    model = accelerator.prepare(model)  # needs to be called individually for FSDP
    # model = torch.compile(model, mode="reduce-overhead")

    ema = ExponentialMovingAverage(model.parameters(), decay=conf.model.ema_rate)
    accelerator.register_for_checkpointing(ema)
    accelerator.load_state(
        f"{project_path}/checkpoints/{conf.eval.run_name}/checkpoint_{conf.accelerator.checkpoint_1}"
    )
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    model.eval()

    conf.data.batch_size = 155
    dataloader = MRI_Slices(conf)
    dataloader = accelerator.prepare(dataloader)
    pbar = tqdm(dataloader)

    with torch.no_grad():
        my_volume = []
        my_masks = []
        for i, (images) in enumerate(pbar):
            images, input, input_mask = create_input(images, transform)
            rec = model(input.float())
            error_map = (images - rec) ** 2
            error_map, input_mask = accelerator.gather_for_metrics(
                (error_map, input_mask)
            )
            my_volume.append(error_map.to("cpu"))
            my_masks.append(input_mask.to("cpu"))

        if accelerator.is_main_process:
            my_volume = torch.cat(my_volume, dim=0)
            my_masks = torch.cat(my_masks, dim=0)
            loss_masked = my_volume * my_masks[:, None]
            kthnum = (
                my_masks.shape[0] * my_masks.shape[1] * my_masks.shape[2]
                - my_masks.sum() * 0.20
            )
            thres_validation = torch.kthvalue(
                loss_masked.flatten(), kthnum.int()
            ).values
            print(thres_validation)


if __name__ == "__main__":
    main()
