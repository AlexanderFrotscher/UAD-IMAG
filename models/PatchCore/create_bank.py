__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import os
from pathlib import Path

import torch
from accelerate import Accelerator
from patchcore import PatchCore
from torchvision.transforms import v2

from conf import PatchCore_config as config
from utils.dataloaders import MRI_Slices
from utils.helpers import make_dicts


def main():
    project_path = Path(os.path.dirname(__file__))
    project_path = os.fspath(project_path.parent.parent)
    conf = config.get_config()
    torch.manual_seed(conf.seed)
    make_dicts(project_path, conf.training.run_name)
    accelerator = Accelerator()
    transform = v2.Compose([v2.CenterCrop(conf.data.image_size)])
    dataloader = MRI_Slices(conf, transform)
    model = PatchCore(conf)
    model, dataloader = accelerator.prepare(model, dataloader)
    model.eval()

    with torch.no_grad():
        model.fit(dataloader, conf.model.index_file)


if __name__ == "__main__":
    main()
