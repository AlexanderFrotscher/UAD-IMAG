__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import pickle
from pathlib import Path
from typing import Callable, Optional, Union

import lmdb
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from utils.normalize import normalize_image


class LMDBDataset(Dataset):
    """To load slices saved in a Lightning Memory-Mapped Database (LMDB).
    The numpy files in the LMDB are created by the create_LMDB.py script.
    These are all preprocessed slices, i.e. they are normalized. Used for training.

    Parameters
    ----------
    Dataset : _type_
        PyTorch class
    """

    def __init__(
        self, directory: Union[str, Path], transform: Optional[Callable] = None
    ):
        self.directory = directory
        self.transform = transform
        env = lmdb.open(
            self.directory,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
        env.close()

    def open_lmdb(self):
        self.env = lmdb.open(
            self.directory,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.txn = self.env.begin(write=False)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not hasattr(self, "txn"):
            self.open_lmdb()

        byteflow = self.txn.get(f"{index:08}".encode("ascii"))
        unpacked = pickle.loads(byteflow)
        tensor = torch.from_numpy(unpacked).float()
        if tensor.dim() <= 2:
            tensor = tensor[None]
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor


class MRIDataVolume(Dataset):
    """The data set class to load and normalize complete volumes with and without a lesion mask. Used for evaluation.

    Parameters
    ----------
    Dataset : _type_
        PyTorch class
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_type: str,
        transform: Optional[Callable] = None,
    ):
        self.df = df
        self.data_type = data_type
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, self.df.columns[0]]
        img = np.asanyarray(nib.load(img_path).dataobj, dtype=np.float32)
        img = normalize_image(img)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)
        img = img[:,None].float()
        mask_path = Path(img_path).name
        mask_path = mask_path.split(f"_{self.data_type}")[0]
        mask_path = f"{mask_path}_seg-anomaly_mask.nii.gz"
        mask_path = f"{Path(img_path).parent}/{mask_path}"
        try:
            mask = np.asanyarray(nib.load(mask_path).dataobj, dtype=np.float32)
            mask[mask > 0.5] = 1
            mask[mask < 1] = 0
            mask = np.transpose(mask, (2, 0, 1))
            mask = torch.from_numpy(mask)
            mask = mask.type(torch.bool)
        except FileNotFoundError:
            mask = torch.zeros(img.shape[0], img.shape[2], img.shape[3])
            mask = mask.type(torch.bool)
        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)
        return img, mask


def MRI_Slices(conf, my_transforms: Optional[Callable] = None):
    dataset = LMDBDataset(conf.data.dataset, my_transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=conf.data.batch_size,
        num_workers=conf.data.workers,
        shuffle=True,
    )
    return dataloader


def MRI_Volume(conf, my_transforms: Optional[Callable] = None):
    df = pd.read_csv(conf.data.dataset)
    dataset = MRIDataVolume(
        df, conf.data.data_type, my_transforms
    )
    dataloader = DataLoader(
        dataset,
        batch_size=conf.data.batch_size,
        num_workers=conf.data.workers,
        shuffle=False,
    )
    return dataloader


def MRI_Slices_val(conf, my_transforms: Optional[Callable] = None):
    dataset_h = LMDBDataset(conf.data.valset_healthy, my_transforms)
    dataloader_h = DataLoader(
        dataset_h,
        batch_size=conf.data.batch_size,
        num_workers=conf.data.workers,
    )
    dataset_l = LMDBDataset(conf.data.valset_lesion, my_transforms)
    dataloader_l = DataLoader(
        dataset_l,
        batch_size=conf.data.batch_size,
        num_workers=conf.data.workers,
    )
    return dataloader_h, dataloader_l
