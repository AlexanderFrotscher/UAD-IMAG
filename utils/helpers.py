__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import functools
import multiprocessing as mp
import os
import random

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import median_filter
from torchvision import transforms
from tqdm import tqdm

from utils.metrics import dice, fpr


def show_slices(slices):
    """Function to display row of image slices"""
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice, cmap="gray")  # viridis
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    plt.show()


def plot_images(images, mode="RGB"):
    if mode == "L":  # mode L is gray scale
        batch_size = images.shape[0]
        channels = images.shape[1]
        images = images.reshape(
            (batch_size * channels, 1, images.shape[2], images.shape[3])
        )
        plt.figure(figsize=(32, 32))
        plt.imshow(
            torch.cat(
                [
                    torch.cat([i for i in images.cpu()], dim=-1),
                ],
                dim=-2,
            )
            .permute(1, 2, 0)
            .cpu(),
            cmap="gray",
        )
        plt.show()
    else:
        plt.figure(figsize=(32, 32))
        plt.imshow(
            torch.cat(
                [
                    torch.cat([i for i in images.cpu()], dim=-1),
                ],
                dim=-2,
            )
            .permute(1, 2, 0)
            .cpu()
        )
        plt.show()


def save_images(images, path, mode="RGB", **kwargs):
    if mode == "L":  # mode L is gray scale
        batch_size = images.shape[0]
        channels = images.shape[1]
        images = images.reshape(
            (batch_size * channels, 1, images.shape[2], images.shape[3])
        )
        grid = torchvision.utils.make_grid(images, **kwargs)
        ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
        im = Image.fromarray(ndarr)
        im.save(path)
    else:
        grid = torchvision.utils.make_grid(images, **kwargs)
        ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
        im = Image.fromarray(ndarr)
        im.save(path)


def upload_images(images, mode="RGB", **kwargs):
    """Creates a numpy array to upload the images to wandb.

    Parameters
    ----------
    images : tensor
        The tensor containing the images
    mode : str, optional
        flag that decides if the images are meant to be RBG or gray scale, by default "RGB"

    Returns
    -------
    numpy.array
        The array that can be uploaded to wandb
    """
    if mode == "L":  # mode L is gray scale
        batch_size = images.shape[0]
        channels = images.shape[1]
        images = images.reshape(
            (batch_size * channels, 1, images.shape[2], images.shape[3])
        )
        grid = torchvision.utils.make_grid(images, **kwargs)
        ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    else:
        grid = torchvision.utils.make_grid(images, **kwargs)
        ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    return ndarr


class EarlyStopper:
    """
    This class is an experimental EarlyStopper for reconstruction-based UAD approaches. It checks if the model is able to reconstruct
    the lesion class better than before. Set min_delta for the delta that is allowed as an decline of difference.
    """

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.difference = float("-inf")

    def early_stop(self, validation_loss, training_loss):
        current_diff = validation_loss - training_loss
        if current_diff > self.difference:
            self.difference = current_diff
            self.counter = 0
        elif current_diff < self.difference - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def median_filter_3D(volume, kernelsize=5):
    pbar = tqdm(range(len(volume)), desc="Median filtering")
    for i in pbar:
        volume[i] = median_filter(volume[i], size=(kernelsize, kernelsize, kernelsize))
    return volume


def norm_ndarray(array):
    my_max = np.max(array)
    my_min = np.min(array)
    my_array = (array - my_min) / (my_max - my_min)
    return my_array


def gmean(input_x, dim, keepdim=False):
    log_x = torch.log(input_x)
    return torch.exp(torch.mean(log_x, dim=dim, keepdim=keepdim))


def calc_dice_scores(conf, my_volume, my_labels):
    thresholds = [
        round(x, 7)
        for x in np.arange(conf.eval.thr_start, conf.eval.thr_end, conf.eval.thr_step)
    ]
    dice_scores = {}
    subject_scores = {}
    my_call = functools.partial(calc_indiv_dice, my_volume, my_labels)
    with mp.Pool(processes=conf.data.workers) as p:
        for i, result in enumerate(p.map(my_call, thresholds)):
            dice_scores[f"{thresholds[i]}"] = result[0]
            subject_scores[f"{thresholds[i]}"] = result[1]
    return dice_scores, subject_scores


def calc_indiv_dice(volume, labels, thr):
    segmentation = np.where(volume > thr, 1.0, 0.0)
    dices = np.asarray([float(x) for x in dice(segmentation, labels)])
    dice_score = np.mean(dices)  # mean over all subjects
    return dice_score, dices


def calc_fpr(volume, labels, thr):
    segmentation = np.where(volume > thr, 1.0, 0.0)
    my_fpr = fpr(segmentation, labels)
    my_fpr = round(my_fpr, 5)
    return my_fpr


def calc_fpr_dist(volumes, labels, thr):
    fprs = []
    for volume, label in zip(volumes, labels):
        volume = np.where(volume > thr, 1.0, 0.0)
        fprs.append(round(fpr(volume, label), 5))
    return fprs


def pyramid_noise_like(n, channels, image_size, discount, device):
    u = transforms.Resize(image_size, antialias=True)
    noise = torch.randn((n, channels, image_size, image_size)).to(device)
    w = image_size
    h = image_size
    for i in range(10):
        r = random.random() * 2 + 2  # Rather than always going 2x,
        w, h = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
        noise += u(torch.randn(n, channels, w, h).to(device)) * discount**i
        if w == 1 or h == 1:
            break  # Lowest resolution is 1x1
    return noise / noise.std()  # Scaled back to roughly unit variance


def make_dicts(path, run_name):
    os.makedirs(f"{path}/checkpoints", exist_ok=True)
    os.makedirs(f"{path}/results", exist_ok=True)
    os.makedirs(os.path.join(f"{path}/checkpoints", run_name), exist_ok=True)
    os.makedirs(os.path.join(f"{path}/results", run_name), exist_ok=True)
