import pickle

import lmdb
import nibabel as nib
import numpy as np
import pandas as pd
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading.dataset import Dataset
from batchgenerators.transforms.abstract_transforms import AbstractTransform

from utils.normalize import normalize_image


class CropForeground(AbstractTransform):
    def __init__(self, key_input="data", keys_to_apply=None):
        """
        Crop the foreground of an image
        :param keys_inputs:
        :param keys_to_apply:
        """
        self.key_input = key_input
        self.keys_to_apply = (
            [key_input]
            if keys_to_apply is None
            else (
                keys_to_apply
                if isinstance(keys_to_apply, (list, tuple))
                else [keys_to_apply]
            )
        )

    def __call__(self, **data_dict):
        outputs = {k: [] for k in self.keys_to_apply}
        for i, (data) in enumerate(data_dict[self.key_input]):
            nonzero = np.stack(np.abs(data).sum(0).nonzero(), -1)  # get coords not zero

            if nonzero.shape[0] != 0:
                nonzero = np.stack([np.min(nonzero, 0), np.max(nonzero, 0)], -1)
                # nonzero now has shape 3, 2. It contains the (min, max) coordinate of nonzero voxels for each axis
                for key in self.keys_to_apply:
                    if key in data_dict:
                        seg = data_dict[key][i]
                        if seg is not None:
                            # now crop to nonzero
                            seg = seg[
                                :,
                                nonzero[0, 0] : nonzero[0, 1] + 1,
                                nonzero[1, 0] : nonzero[1, 1] + 1,
                            ]
                            if nonzero.shape[0] == 3:
                                seg = seg[:, :, :, nonzero[2, 0] : nonzero[2, 1] + 1]

                            outputs[key].append(seg)

                # metadata["nonzero_region"] = nonzero
            else:
                for key in self.keys_to_apply:
                    outputs[key].append(data_dict[key][i])

        # Note that the output of this is a list, instead of single numpy array because each can have different sizes
        # Hope for anything that comes after to iterate over the list and does not expect a np.array
        for key in self.keys_to_apply:
            data_dict[key] = outputs[key]

        return data_dict


class DataLoader3D(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size,
        num_threads_in_multithreaded=0,
        seed_for_shuffle=1234,
        return_incomplete=True,
        shuffle=True,
        infinite=True,
    ):
        self.datalist = pd.read_csv(dataset)["Path"]
        super().__init__(
            self.datalist,
            batch_size,
            num_threads_in_multithreaded,
            seed_for_shuffle,
            return_incomplete,
            shuffle,
            infinite,
        )
        self.indices = list(range(len(self.datalist)))

    @staticmethod
    def load_file(file):
        data = nib.load(file)
        data_array = np.asanyarray(data.dataobj, order="C").astype(np.float32)
        # data_array = data.get_fdata()
        return data_array, {"affine": data.affine, "filename": file}

    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]

        # initialize empty array for image and labels
        images = []

        # iterate over patients_for_batch and include them in the batch
        for i, subj in enumerate(patients_for_batch):
            patient_data = []
            img, img_mtd = self.load_file(subj)
            patient_data.append(img)
        images.append(np.stack(patient_data, axis=0))

        return {"data": images}


class DataLoader2Dfrom3D(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size,
        num_threads_in_multithreaded=0,
        seed_for_shuffle=1234,
        num_slices=None,
        return_incomplete=True,
        shuffle=True,
        infinite=True,
    ):
        self.datalist = pd.read_csv(dataset)["Path"]
        super().__init__(
            self.datalist,
            batch_size,
            num_threads_in_multithreaded,
            seed_for_shuffle,
            return_incomplete,
            shuffle,
            infinite,
        )

        self.indices = list(range(len(self.datalist)))
        self.num_slices = num_slices

    @staticmethod
    def load_file(file):
        data = nib.load(file)
        data_array = np.asanyarray(data.dataobj, order="C").astype(np.float32)
        # data_array = data.get_fdata()
        return data_array, {"affine": data.affine, "filename": file}

    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]

        # initialize empty array for image and labels
        images = []

        # iterate over patients_for_batch and include them in the batch
        for i, subj in enumerate(patients_for_batch):
            patient_data = []
            img, img_mtd = self.load_file(subj)
            img = normalize_image(img)
            patient_data.append(img)
            patient_data = np.stack(patient_data, axis=0)

            # Get non-empty axial slices
            list_ax_sl_ind = (
                np.abs(patient_data).reshape(-1, patient_data.shape[-1]).sum(0)
            )
            list_ax_sl_ind = list_ax_sl_ind.nonzero()[0]

            if self.num_slices is not None:
                # Randomly sample num_slices slices
                z_slices = np.random.randint(
                    list_ax_sl_ind.min(), list_ax_sl_ind.max(), size=(self.num_slices,)
                )
                images.extend(list(np.moveaxis(patient_data[:, :, :, z_slices], -1, 0)))
            else:
                # Get all non-empty slices
                images.extend(
                    list(np.moveaxis(patient_data[:, :, :, list_ax_sl_ind], -1, 0))
                )

        return {"data": images}


class LMDBDataset(Dataset):
    """To load slices saved in a Lightning Memory-Mapped Database (LMDB).
    The numpy files in the LMDB are created by the split_healthy.py script.
    These are all preprocessed slices, i.e. they are normalized. Used for training.

    Parameters
    ----------
    Dataset : _type_
        PyTorch class
    """

    def __init__(self, directory: str, my_transforms):
        super(LMDBDataset, self).__init__()
        self.directory = directory
        self.transforms = my_transforms
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
        unpacked = np.expand_dims(unpacked, axis=0)
        return unpacked


class DataLoaderLMDB(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size,
        num_threads_in_multithreaded=0,
        seed_for_shuffle=1234,
        return_incomplete=True,
        shuffle=True,
        infinite=True,
    ):  # infinite has to be true
        super(DataLoaderLMDB, self).__init__(
            dataset,
            batch_size,
            num_threads_in_multithreaded,
            seed_for_shuffle,
            return_incomplete,
            shuffle,
            infinite,
        )

        self.indices = list(range(len(dataset)))

    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]
        return {"data": patients_for_batch}
