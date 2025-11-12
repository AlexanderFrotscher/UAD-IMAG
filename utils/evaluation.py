__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_curve

from utils.helpers import (
    calc_dice_scores,
    calc_fpr,
    calc_fpr_dist,
    calc_indiv_dice,
    median_filter_3D,
    norm_ndarray,
)


def evaluate(
    conf,
    my_volume: torch.Tensor,
    my_labels: torch.Tensor,
    project_path: str,
):
    """This function evaluates anomaly maps with the Dice Score and AUPRC. Median filtering is applied as post-processing.

    Parameters
    ----------
    conf : ml_collections.ConfigDict
        The config
    my_volume : torch.Tensor
        A tensor of shape (B, C, H, W, D) containing the anomaly maps
    my_labels : torch.Tensor
        A tensor of shape (B, H, W, D) containing the binary segmentations
    project_path : str
        The path to the root directory. Needed to save the output .csv's
    """
    my_volume = torch.cat(my_volume, dim=0)
    my_labels = torch.cat(my_labels, dim=0)

    my_volume = my_volume[:, 0]  # remove channel dimension
    my_volume = my_volume.numpy()
    my_labels = my_labels.numpy()

    if conf.eval.calc_thr:
        my_volume = my_volume[
            ~my_labels
        ]  # for FPR on healthy data we only want to evaluate brain voxels
        my_labels = my_labels[~my_labels]
        fpr, tpr, thresholds = roc_curve(my_labels.ravel(), my_volume.ravel())
        for max_fpr in conf.eval.fpr:
            th = thresholds[fpr < max_fpr][-1]
            print(f"Threshold for FPR equal to {max_fpr} is {th}")
        opt_fpr = calc_fpr(my_volume, my_labels, conf.eval.opt_thr)
        print(f"Threshold for FPR equal to {opt_fpr} is {conf.eval.opt_thr}")
    else:
        my_volume_mf = np.copy(my_volume)
        my_volume_mf = median_filter_3D(my_volume_mf, conf.postprocessing.kernel_size)

        if conf.eval.data_is_healthy:
            if conf.eval.distribution:
                fpr_volumes = []
                fpr_labels = []
                fpr_volumes_mf = []
                for volume, volume_mf, label in zip(my_volume, my_volume_mf, my_labels):
                    fpr_volumes.append(volume[~label])
                    fpr_labels.append(label[~label])
                    fpr_volumes_mf.append(volume_mf[~label])
                fpr_per_subject = {}
                fpr_per_subject_mf = {}
                for thr, fpr in zip(conf.eval.thr_fpr, conf.eval.fpr):
                    fpr_per_subject[f"FPR-{fpr}"] = calc_fpr_dist(
                        fpr_volumes, fpr_labels, thr
                    )
                    fpr_per_subject_mf[f"FPR-{fpr}"] = calc_fpr_dist(
                        fpr_volumes_mf, fpr_labels, thr
                    )
                filehandler = open("fpr-distribution.obj", "wb")
                pickle.dump([fpr_per_subject, fpr_per_subject_mf], filehandler)
                filehandler.close
            my_volume = my_volume[
                ~my_labels
            ]  # for FPR on healthy data we only want to evaluate brain voxels
            my_volume_mf = my_volume_mf[~my_labels]
            my_labels = my_labels[~my_labels]
            dice_scores = {}  # not dice score now
            dice_scores_mf = {}
            for thr, fpr in zip(conf.eval.thr_fpr, conf.eval.fpr):
                dice_scores[f"FPR-{fpr}"] = calc_fpr(my_volume, my_labels, thr)
            for thr, fpr in zip(conf.eval.thr_fpr, conf.eval.fpr):
                dice_scores_mf[f"FPR-{fpr}"] = calc_fpr(my_volume_mf, my_labels, thr)

        else:
            aupr = average_precision_score(
                my_labels.ravel(), norm_ndarray(my_volume).ravel()
            )
            dice_scores, subject_scores = calc_dice_scores(conf, my_volume, my_labels)
            val_thr_dist = {}
            for thr, fpr in zip(conf.eval.thr_fpr, conf.eval.fpr):
                dice_scores[f"FPR-{fpr}"], val_thr_dist[f"FPR-{fpr}"] = calc_indiv_dice(
                    my_volume, my_labels, thr
                )
            dice_scores["AUPRC"] = aupr

            aupr_mf = average_precision_score(
                my_labels.ravel(), norm_ndarray(my_volume_mf).ravel()
            )
            dice_scores_mf, subject_scores_mf = calc_dice_scores(
                conf, my_volume_mf, my_labels
            )
            val_thr_dist_mf = {}
            for thr, fpr in zip(conf.eval.thr_fpr, conf.eval.fpr):
                dice_scores_mf[f"FPR-{fpr}"], val_thr_dist_mf[f"FPR-{fpr}"] = (
                    calc_indiv_dice(my_volume_mf, my_labels, thr)
                )
            dice_scores_mf["AUPRC"] = aupr_mf

            if conf.eval.distribution:
                filehandler = open("dice-distribution.obj", "wb")
                pickle.dump(
                    [
                        subject_scores,
                        subject_scores_mf,
                        val_thr_dist,
                        val_thr_dist_mf,
                    ],
                    filehandler,
                )
                filehandler.close

        df = pd.DataFrame.from_dict(dice_scores, orient="index", columns=["value"])
        df.index.name = "thr"
        df.to_csv(f"{project_path}/{conf.eval.output}")

        df_mf = pd.DataFrame.from_dict(
            dice_scores_mf, orient="index", columns=["value"]
        )
        df_mf.index.name = "thr"
        df_mf.to_csv(f"{project_path}/{conf.eval.output_mf}")
