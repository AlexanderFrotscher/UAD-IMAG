__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"


import os
from pathlib import Path

import torch
from accelerate import Accelerator
from patchcore import PatchCore
from torchvision.transforms import v2
from tqdm import tqdm

from conf import PatchCore_eval as config
from utils.dataloaders import MRI_Volume
from utils.evaluation import evaluate


def main():
    project_path = Path(os.path.dirname(__file__))
    project_path = os.fspath(project_path.parent.parent)
    conf = config.get_config()
    torch.manual_seed(conf.seed)
    accelerator = Accelerator()
    transform = v2.Compose([v2.CenterCrop(conf.data.image_size)])
    dataloader = MRI_Volume(conf, transform)
    model = PatchCore(conf)
    model.nn_method.load(conf.model.index_file)
    model = accelerator.prepare(model)  # needs to be called individually for FSDP
    # model = torch.compile(model, mode="reduce-overhead")

    dataloader = accelerator.prepare(dataloader)

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
            split = torch.split(image, conf.eval.split_size)
            anomaly_maps = []
            for my_tensor in split:
                scores, masks = model.predict(my_tensor)
                masks[my_tensor == 0] = 0
                anomaly_maps.append(masks)

            anomaly_maps = torch.cat(anomaly_maps, dim=0)

            anomaly_maps = anomaly_maps.view(
                num_volumes,
                num_slices,
                anomaly_maps.shape[1],
                anomaly_maps.shape[2],
                anomaly_maps.shape[3],
            )
            anomaly_maps = torch.permute(anomaly_maps, (0, 2, 3, 4, 1))
            label = torch.permute(label, (0, 2, 3, 1))
            anomaly_maps, label = accelerator.gather_for_metrics((anomaly_maps, label))
            my_labels.append(label.type(torch.bool).to("cpu"))
            my_volume.append(anomaly_maps.to("cpu"))

        if accelerator.is_main_process:
            evaluate(conf, my_volume, my_labels, project_path)


if __name__ == "__main__":
    main()
