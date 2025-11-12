__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"


import os
from pathlib import Path

import torch
from accelerate import Accelerator
from anomaly_detection import ANDi
from DDPMpp import DDPMpp
from diffusion import my_VPSDE
from torchvision.transforms import v2
from tqdm import tqdm

from conf import ANDi_eval as config
from utils.dataloaders import MRI_Volume
from utils.ema import ExponentialMovingAverage
from utils.evaluation import evaluate
from utils.helpers import gmean


def main():
    project_path = Path(os.path.dirname(__file__))
    project_path = os.fspath(project_path.parent.parent)
    conf = config.get_config()
    torch.manual_seed(conf.seed)
    accelerator = Accelerator()
    transform = v2.Compose([v2.CenterCrop(conf.data.image_size)])
    dataloader = MRI_Volume(conf, transform)

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

    ema = ExponentialMovingAverage(model.parameters(), decay=conf.model.ema_rate)
    dataloader = accelerator.prepare(dataloader)
    accelerator.register_for_checkpointing(ema)
    accelerator.load_state(
        f"{project_path}/checkpoints/{conf.eval.run_name}/checkpoint_{conf.accelerator.checkpoint}"
    )
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    model.eval()
    pbar = tqdm(dataloader)

    with torch.no_grad():
        my_volume = []
        my_labels = []
        for image, label in pbar:
            num_volumes = image.shape[0]
            num_slices = image.shape[1]
            if conf.eval.data_is_healthy:
                label[image[:, :, 0] == 0] = True
            if conf.eval.slice:
                index = torch.argmax(torch.sum(label, (2, 3)), dim=1)
                image = image[torch.arange(num_volumes), index]
                label = label[torch.arange(num_volumes), index]
                num_slices = image.shape[0] // num_volumes
                image = image[None]
                label = label[:, None]
            image = image.flatten(0, 1)
            image = (image * 2) - 1
            split = torch.split(image, conf.eval.split_size)
            preds = []
            for my_tensor in split:
                pred = ANDi(
                    sde,
                    model,
                    my_tensor,
                    conf.sampling.aggregation,
                    conf.model.vpred,
                    conf.sampling.start,
                    conf.sampling.stop,
                )
                preds.append(pred)

            preds = torch.cat(preds, dim=0)

            if conf.sampling.aggregation == "gmean":
                aggregation = gmean(preds, dim=1)
            elif conf.sampling.aggregation == "prob":
                aggregation = torch.prod(preds, dim=1)
                aggregation = 1 - aggregation

            aggregation[image == -1] = 0

            aggregation = aggregation.view(
                num_volumes,
                num_slices,
                aggregation.shape[1],
                aggregation.shape[2],
                aggregation.shape[3],
            )

            aggregation = torch.permute(aggregation, (0, 2, 3, 4, 1))
            label = torch.permute(label, (0, 2, 3, 1))
            aggregation, label = accelerator.gather_for_metrics((aggregation, label))
            my_labels.append(label.to("cpu"))
            my_volume.append(aggregation.to("cpu"))

        if accelerator.is_main_process:
            evaluate(conf, my_volume, my_labels, project_path)


if __name__ == "__main__":
    main()
