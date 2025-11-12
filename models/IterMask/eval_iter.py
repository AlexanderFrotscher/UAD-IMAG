__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"


import os
from pathlib import Path

import torch
from accelerate import Accelerator
from masking import create_condition
from torchvision.transforms import v2
from tqdm import tqdm
from UNet import UNet

from conf import IterMask_eval as config
from utils.dataloaders import MRI_Volume
from utils.ema import ExponentialMovingAverage
from utils.evaluation import evaluate


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

    conf.model.condition = False
    accelerator2 = Accelerator()
    model_first = UNet(conf)
    model_first = accelerator2.prepare(model_first)
    ema2 = ExponentialMovingAverage(model_first.parameters(), decay=conf.model.ema_rate)
    accelerator2.register_for_checkpointing(ema2)
    accelerator2.load_state(
        f"{project_path}/checkpoints/{conf.eval.first_model}/checkpoint_{conf.accelerator.checkpoint_2}"
    )
    ema2.store(model_first.parameters())
    ema2.copy_to(model_first.parameters())
    model_first.eval()

    dataloader = MRI_Volume(conf, transform)
    dataloader = accelerator.prepare(dataloader)
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
            image = (image * 6) - 3
            split = torch.split(image, conf.eval.split_size)
            anomaly_maps = []
            for my_tensor in split:
                brain_mask = torch.zeros_like(my_tensor)
                brain_mask[my_tensor > -3] = 1
                x_cond = create_condition(my_tensor)
                x0_pred_firststep = model_first(x_cond)
                error_map_first = ((my_tensor - x0_pred_firststep) ** 2) * brain_mask
                error_map = error_map_first

                flag = torch.ones([my_tensor.shape[0]]).cuda()
                final_reconstruction = torch.zeros_like(my_tensor)
                x0_pred = torch.zeros_like(my_tensor)
                final_mask = torch.zeros_like(my_tensor)
                mask_inpaint_input = brain_mask
                j = 0
                while flag.sum() != 0:
                    if j == 0:
                        thres = torch.zeros([my_tensor.shape[0]]).cuda()
                        for num in range(my_tensor.shape[0]):
                            kthnum = (
                                brain_mask.shape[2] * brain_mask.shape[3]
                                - brain_mask[num, :, :, :].sum() * 0.6
                            )  # 0.6 this is some kind of lession estimate
                            thres[num] = torch.kthvalue(
                                error_map[num, 0, :, :].flatten(), kthnum.int()
                            ).values
                        thres = (
                            thres.unsqueeze(1)
                            .unsqueeze(2)
                            .unsqueeze(3)
                            .repeat(1, 1, my_tensor.shape[2], my_tensor.shape[3])
                        )
                    else:
                        thres = conf.eval.validation_thr

                    mask_inpaint_input_new = (
                        torch.where(thres < error_map, 1.0, 0.0) * brain_mask
                    )

                    ### the next lines decide which sample needs to be longer in the iterative process
                    ratio = (
                        mask_inpaint_input.sum(dim=(1, 2, 3))
                        - mask_inpaint_input_new.sum(dim=(1, 2, 3))
                    ) / mask_inpaint_input.sum(dim=(1, 2, 3))
                    ratio = torch.where(torch.isnan(ratio), -1, ratio)

                    update_flag = (ratio < 0.1) * (
                        flag == 1
                    )  # ratio normally < 0.01, but this works better
                    flag = flag * (~update_flag).int()
                    if j > 0:
                        ### save the final predictions ###
                        index = torch.where((update_flag == 1).int())
                        final_reconstruction[index] = x0_pred[index]
                        final_mask[index] = mask_inpaint_input[index]
                    if flag.sum() == 0:
                        break

                    mask_inpaint_input = mask_inpaint_input_new

                    noise = torch.randn_like(my_tensor)
                    x_masked = (
                        1 - mask_inpaint_input
                    ) * my_tensor + mask_inpaint_input * noise

                    x_input = torch.cat((x_masked, x_cond), dim=1)
                    x0_pred = model(x_input.float())
                    x0_pred_combine = (
                        mask_inpaint_input * x0_pred
                        + (1 - mask_inpaint_input) * my_tensor
                    )

                    error_map = (my_tensor - x0_pred_combine) ** 2
                    j += 1
                error_map = ((my_tensor - final_reconstruction) ** 2) * brain_mask
                anomaly_maps.append(error_map)

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
            my_labels.append(label.to("cpu"))
            my_volume.append(anomaly_maps.to("cpu"))

        if accelerator.is_main_process:
            evaluate(conf, my_volume, my_labels, project_path)


if __name__ == "__main__":
    main()
