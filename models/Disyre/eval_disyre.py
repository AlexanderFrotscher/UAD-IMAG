__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"


import os
from pathlib import Path

import torch
from accelerate import Accelerator
from diffusion import VPSDE
from sliding_window_inference import sliding_window_inference
from torchvision.transforms import v2
from tqdm import tqdm

from conf import Disyre_eval as config
from models.ANDi.DDPMpp import DDPMpp
from utils.dataloaders import MRI_Volume
from utils.ema import ExponentialMovingAverage
from utils.evaluation import evaluate


def main():
    project_path = Path(os.path.dirname(__file__))
    project_path = os.fspath(project_path.parent.parent)
    conf = config.get_config()
    torch.manual_seed(conf.seed)
    accelerator = Accelerator()
    device = accelerator.device
    transform = v2.Compose([v2.CenterCrop(conf.data.load_size)])
    dataloader = MRI_Volume(conf, transform)

    sde = VPSDE(
        conf.diffusion.beta_min, conf.diffusion.beta_max, conf.diffusion.num_latents
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
            timesteps = torch.arange(
                sde.N, 0, -(sde.N // conf.sampling.n_steps), device=device
            )
            timesteps -= 1
            for my_tensor in split:
                for i in tqdm(range(len(timesteps))):
                    if conf.eval.sw_batch_size:
                        t = timesteps[i]
                        x_0 = sliding_window_inference(
                            image=my_tensor,
                            patch_size=[conf.data.image_size, conf.data.image_size],
                            predictor=model,
                            timestep=t,
                            sw_batch_size=conf.eval.sw_batch_size,
                        )
                        pred = (my_tensor - x_0).abs()
                        preds.append(pred[:, None])
                    else:
                        t = timesteps[i]
                        vec_t = (
                            torch.ones(my_tensor.shape[0], device=my_tensor.device) * t
                        )
                        x_0 = model(my_tensor, vec_t)
                        pred = (my_tensor - x_0).abs()
                        preds.append(pred[:, None])

            preds = torch.cat(preds, dim=1)
            aggregation = torch.mean(preds, dim=1)
            weight = (image > -1).flatten(1).float().mean(1)[:, None, None, None]
            aggregation = aggregation * weight
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
