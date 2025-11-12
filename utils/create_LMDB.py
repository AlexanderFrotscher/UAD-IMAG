__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import argparse
import os
import pickle

import lmdb
import nibabel as nib
import numpy as np
import pandas as pd
from utils.normalize import normalize_image
from tqdm import tqdm


def main(args):
    split_and_save(args)


def split_and_save(args):
    os.makedirs(
        args.path,
        exist_ok=True,
    )
    df = pd.read_csv(args.data_set)
    pbar = tqdm(df["Path"])
    map_size = 70000000 * len(pbar)  #  71424196 * len(pbar)
    env = lmdb.open(
        str(
            f"{args.path}\\slices"
        ),
        map_size=map_size,
    )
    with env.begin(write=True) as txn:
        num = 0
        for subject in pbar:
            my_slices = []
            img = np.asarray(nib.load(subject).dataobj, dtype=float)
            img = normalize_image(img)
            for i in range(img.shape[2]):
                my_slice = img[:, :, i]
                if np.any(my_slice):
                    my_slices.append(my_slice)
            for data in my_slices:
                key = f"{num:08}"
                num += 1
                txn.put(key.encode("ascii"), pickle.dumps(data))
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split the data into slices.")
    parser.add_argument(
        "-d",
        "--data_set",
        type=str,
        required=True,
        metavar="",
        help="The .csv that contains the paths to the MRI-Volumes.",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=True,
        metavar="",
        help="The path indicating where to store the LMDB.",
    )
    args = parser.parse_args()
    main(args)
