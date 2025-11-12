import random

import numpy as np
import torch
import torch.fft as fft
from scipy.stats import multivariate_normal


def gen_mask(brain_mask):
    gauss_mask = np.ones_like(brain_mask)
    num = random.randint(0, 2)
    covar11 = random.uniform(0.3, 10)
    covar22 = random.uniform(0.3, 10)
    covar12 = random.uniform(0, np.sqrt(covar11 * covar22)) * (
        -1 if random.randint(0, 1) == 0 else 1
    )
    target_area = np.where(brain_mask > 0)
    rand = np.random.randint(target_area[0].shape)
    if num == 0 or num == 1:
        x, y = np.mgrid[-12:12:0.1, -12:12:0.1]
        # x, y = np.mgrid[-11.2:11.2:.1, -11.2:11.2:.1]
        num1 = np.random.randint(
            np.sqrt(covar11 * covar22) * 100, np.sqrt(covar11 * covar22) * 200
        )
        num2 = np.random.randint(5, 10)
        num3 = np.random.randint(3, 10)
        mean1 = target_area[0][rand][0] / 240 * 24 - 12  # / 224 * 22.4 - 11.2
        mean2 = target_area[1][rand][0] / 240 * 24 - 12  # / 224 * 22.4 - 11.2
    elif num == 2:
        x, y = np.mgrid[-48:48:0.4, -48:48:0.4]
        # x, y = np.mgrid[-44.8:44.8:.4, -44.8:44.8:.4]
        num1 = np.random.randint(
            np.sqrt(covar11 * covar22) * 5, np.sqrt(covar11 * covar22) * 20
        )
        num2 = np.random.randint(5, 10)
        num3 = np.random.randint(3, 10)
        mean1 = target_area[0][rand][0] / 240 * 96 - 48  # / 224 * 89.6 - 44.8
        mean2 = target_area[1][rand][0] / 240 * 96 - 48  # / 224 * 89.6 - 44.8

    pos = np.dstack((x, y))

    rv = multivariate_normal([mean1, mean2], [[covar11, covar12], [covar12, covar22]])
    gau_pdf = rv.pdf(pos)
    gau_pdf = gau_pdf / gau_pdf.max()

    p = gau_pdf[target_area[0], target_area[1]]
    l1 = 4
    l2 = 8
    l3 = 16
    l4 = 32

    unique_number_dim1 = np.random.choice(
        target_area[0].shape[0], num1 + num2 + num3, p=p.reshape(-1) / p.sum()
    )
    unique_number_dim2 = np.random.choice(
        target_area[0].shape[0], num1 + num2 + num3, p=p.reshape(-1) / p.sum()
    )
    for i in range(unique_number_dim1.shape[0]):
        if i < num1:
            gauss_mask[
                target_area[0][unique_number_dim1[i]] : target_area[0][
                    unique_number_dim1[i]
                ]
                + l1,
                target_area[1][unique_number_dim2[i]] : target_area[1][
                    unique_number_dim2[i]
                ]
                + l1,
            ] = 0
        elif i < num1 + num2:
            gauss_mask[
                target_area[0][unique_number_dim1[i]] : target_area[0][
                    unique_number_dim1[i]
                ]
                + l2,
                target_area[1][unique_number_dim2[i]] : target_area[1][
                    unique_number_dim2[i]
                ]
                + l2,
            ] = 0
        elif i < num1 + num2 + num3:
            gauss_mask[
                target_area[0][unique_number_dim1[i]] : target_area[0][
                    unique_number_dim1[i]
                ]
                + l3,
                target_area[1][unique_number_dim2[i]] : target_area[1][
                    unique_number_dim2[i]
                ]
                + l3,
            ] = 0
        else:
            gauss_mask[
                target_area[0][unique_number_dim1[i]] : target_area[0][
                    unique_number_dim1[i]
                ]
                + l4,
                target_area[1][unique_number_dim2[i]] : target_area[1][
                    unique_number_dim2[i]
                ]
                + l4,
            ] = 0
    return torch.from_numpy(gauss_mask)


def create_condition(images):
    y_input = fft.fftshift(fft.fft2(images))
    center = (images.shape[2] // 2, images.shape[3] // 2)
    X, Y = np.ogrid[: images.shape[2], : images.shape[3]]
    radius = 15
    dist_from_center1 = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = torch.from_numpy((dist_from_center1 >= radius)).cuda()
    y_masked = mask * y_input
    abs_masked = torch.abs(y_masked)
    abs = torch.abs(y_input)
    angle = torch.angle(y_input)
    abs_ones = torch.ones(abs.shape).cuda()
    abs_mask_zerotot1 = abs_masked * mask + abs_ones * ~mask
    fft_ = abs_mask_zerotot1 * torch.exp((1j) * angle)
    img = fft.ifft2(fft.ifftshift(fft_))
    x_mask_real = torch.real(img)
    x_cond = x_mask_real

    return x_cond


def create_input(images: torch.Tensor, transform):
    brain_masks = torch.zeros_like(images[:, 0])
    brain_masks[images[:, 0] > 0] = 1
    input_masks = torch.zeros_like(brain_masks)
    for i, mask in enumerate(brain_masks):
        input_masks[i] = gen_mask(mask.cpu().numpy()).to(images.device)
    images = transform(images)
    input_masks = transform(input_masks)
    brain_masks = transform(brain_masks)
    images = (images * 6) - 3
    x_cond = create_condition(images)
    noise = torch.randn_like(images)
    input_masks = (1 - input_masks) * brain_masks
    input = (1 - input_masks[:, None]) * images + input_masks[:, None] * noise
    input = torch.cat((input, x_cond), 1)
    return images, input, input_masks
