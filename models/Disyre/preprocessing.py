import os

import numpy as np
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    ClipValueRange,
    GammaTransform,
)
from batchgenerators.transforms.noise_transforms import (
    GaussianBlurTransform,
    GaussianNoiseTransform,
)
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators_custom import SpatialTransform_2
from bias import BiasCorruption
from fpi import FPI
from loading import (
    CropForeground,
    DataLoader2Dfrom3D,
    DataLoader3D,
    DataLoaderLMDB,
    LMDBDataset,
)
from mask_generation import CreateRandomShape, GetRandomLocation


def get_train_transform(config, ag_transforms=[]):
    patch_size = [config.data.image_size, config.data.image_size]
    anom_type = config.data.anom_type

    tr_transforms = [
        CropForeground(
            key_input="data",
            keys_to_apply=[
                "data",
            ],
        ),
        SpatialTransform_2(
            patch_size,
            [p // 2 for p in patch_size],
            do_elastic_deform=True,
            deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(-15 / 360.0 * 2 * np.pi, 15 / 360.0 * 2 * np.pi),
            angle_y=(-15 / 360.0 * 2 * np.pi, 15 / 360.0 * 2 * np.pi),
            angle_z=(-15 / 360.0 * 2 * np.pi, 15 / 360.0 * 2 * np.pi),
            do_scale=True,
            scale=(0.75, 1.25),
            border_mode_data="constant",
            border_cval_data=0,
            border_mode_seg="constant",
            border_cval_seg=0,
            order_seg=1,
            order_data=3,
            random_crop=True,
            p_el_per_sample=0.1,
            p_rot_per_sample=0.1,
            p_scale_per_sample=0.1,
        ),
        MirrorTransform(axes=(0, 1, 2)),
    ]
    tr_transforms.extend(
        [
            BrightnessMultiplicativeTransform(
                (0.9, 1.1), per_channel=True, p_per_sample=0.15
            ),
            GammaTransform(
                gamma_range=(0.5, 2),
                invert_image=False,
                per_channel=True,
                p_per_sample=0.15,
            ),
            GammaTransform(
                gamma_range=(0.5, 2),
                invert_image=True,
                per_channel=True,
                p_per_sample=0.15,
            ),
            GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15),
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.5),
                different_sigma_per_channel=True,
                p_per_channel=0.5,
                p_per_sample=0.15,
            ),
            ClipValueRange(
                min=0.0,
                max=1.0,
            ),
        ]
    )
    tr_transforms.extend(ag_transforms)

    keys_to_torch = [
        "data",
        "data_c",
    ]
    if anom_type in ["dag", "dag_no_quant", "fpi"]:
        keys_to_torch += ["alpha_texture"]
    if anom_type in ["dag", "dag_no_quant", "bias_only"]:
        keys_to_torch += ["alpha_bias"]

    tr_transforms.append(NumpyToTensor(keys=keys_to_torch, cast_to="float"))

    tr_transforms = Compose(tr_transforms)

    return tr_transforms


def get_ag_transforms(
    type="dag",
    anom_patch_size=[64, 64, 64],
    no_mask_in_background=False,
    shape_dataset=None,
    p_anomaly=1.0,
    quantized_bias_codebook=[0.00565079, 0.38283867, 0.61224216, 0.76920754, 0.9385473],
    randomshape_kwargs={},
    fpi_kwargs={},
    bias_kwargs={},
):
    assert type in ["dag", "fpi", "dag_no_quant", "bias_only"]

    if shape_dataset is not None and "randommask_kwargs" not in randomshape_kwargs:
        randomshape_kwargs["randommask_kwargs"] = {
            "dataset": shape_dataset,
            "spatial_prob": 1.0,
            "scale_masks": (0.5, 0.75),
        }

    ag_transforms = [
        GetRandomLocation(anom_patch_size=anom_patch_size),
        CreateRandomShape(
            anom_patch_size=anom_patch_size,
            smooth_prob=1.0,
            no_mask_in_background=no_mask_in_background,
            **randomshape_kwargs,
        ),
    ]

    # Override with None if not needed
    quantized_bias_codebook = (
        quantized_bias_codebook if type in ["dag", "bias_only"] else None
    )

    if type == "fpi":
        ag_transforms.append(
            FPI(
                image_key="data",
                anomaly_interpolation="linear",
                output_key="data_c",
                p_anomaly=p_anomaly,
                anom_patch_size=anom_patch_size,
                normalize_fp=False,
                **fpi_kwargs,
            )
        )

    elif type in ["dag", "dag_no_quant"]:
        ag_transforms.extend(
            [
                FPI(
                    image_key="data",
                    anomaly_interpolation="linear",
                    output_key="data_c",
                    p_anomaly=p_anomaly,
                    anom_patch_size=anom_patch_size,
                    normalize_fp="minmax",
                    **fpi_kwargs,
                ),
                BiasCorruption(
                    image_key="data_c",
                    shape_key="shape",
                    output_key="data_c",
                    p_anomaly=p_anomaly,
                    quantized_bias_codebook=quantized_bias_codebook,
                    quantized_mask_key="shape_bias",
                    **bias_kwargs,
                ),
            ]
        )

    elif type == "bias_only":
        ag_transforms.extend(
            [
                BiasCorruption(
                    image_key="data",
                    shape_key="shape",
                    output_key="data_c",
                    p_anomaly=p_anomaly,
                    quantized_bias_codebook=quantized_bias_codebook,
                    quantized_mask_key="shape_bias",
                    **bias_kwargs,
                )
            ]
        )

    return ag_transforms


def create_dataloader(config):
    ag_transforms = get_ag_transforms(
        config.data.anom_type,
        config.data.anom_patch_size,
        config.data.no_anom_in_background,
        config.data.shape_dataset,
        config.data.p_anomaly,
        quantized_bias_codebook=config.data.codebook,
    )

    tr_transforms = get_train_transform(config, ag_transforms)

    # Get the number of cpu available in host
    num_workers = os.cpu_count()
    num_workers = min(num_workers, config.data.workers)

    my_dataset = LMDBDataset(config.data.dataset, None)
    dataloader_train = DataLoaderLMDB(my_dataset, config.data.batch_size)

    # dataloader_train = DataLoader2Dfrom3D(config.data.dataset,config.data.batch_size)

    tr_gen = MultiThreadedAugmenter(
        dataloader_train,
        tr_transforms,
        num_processes=num_workers,
        num_cached_per_queue=3,
    )
    return tr_gen
