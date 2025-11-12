__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"


import ml_collections
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils.dataloaders import MRI_Volume


def check_3D(input_tensor):
    # Initialize variables to keep track of the maximum height and width
    max_h, max_w, max_d = 0, 0, 0

    # List to hold the resulting tensors for each batch and channel
    for sample in input_tensor:
        nonzero = sample.nonzero()
        indicies = torch.zeros((3, 2), dtype=int)
        for j in range(nonzero.shape[1]):
            indicies[j, 0] = nonzero[:, j].min()
            indicies[j, 1] = nonzero[:, j].max()

        sample = sample[
            indicies[0, 0] : indicies[0, 1] + 1,
            indicies[1, 0] : indicies[1, 1] + 1,
            indicies[2, 0] : indicies[2, 1] + 1,
        ]
        # Update the maximum height and width
        max_h = max(max_h, sample.shape[1])
        max_w = max(max_w, sample.shape[2])
        max_d = max(max_d, sample.shape[0])

    return max_h, max_w, max_d


def main():
    config = ml_collections.ConfigDict()
    config.data = data = ml_collections.ConfigDict()
    data.dataset = ""
    data.batch_size = 1
    data.workers = 1
    data.data_type = "T1w"
    df = pd.read_csv(config.data.dataset)
    dataloader = MRI_Volume(config)
    pbar = tqdm(dataloader)
    check_data = False
    percentiles = [25, 50, 75]

    with torch.no_grad():
        my_volume = []
        my_labels = []
        if check_data:
            coordinate_x = []
            coordinate_y = []
            coordinate_z = []
            for i, (image, label) in enumerate(pbar):
                print(f"\n{df.loc[i, df.columns[0]]}")
                x, y, z = check_3D(image[:,:,0])
                coordinate_x.append(x)
                coordinate_y.append(y)
                coordinate_z.append(z)
            print(np.asarray(coordinate_x).max())
            print(np.asarray(coordinate_y).max())
            print(np.asarray(coordinate_z).max())

        else:
            for i, (image, label) in enumerate(pbar):
                my_labels.append(torch.sum(label))
                my_volume.append(df.loc[i, df.columns[0]])
            values = torch.tensor([t.item() for t in my_labels])
            values = values.float()
            percentile_values = torch.quantile(
                values, torch.tensor([p / 100.0 for p in percentiles])
            )
            p25, p50, p75 = percentile_values

            lower_mask = values <= p25
            middle_mask = (values > p25) & (values <= p50)
            upper_mask = (values > p50) & (values <= p75)
            top_mask = values > p75

            lower = [s for i, s in enumerate(my_volume) if lower_mask[i]]
            middle = [s for i, s in enumerate(my_volume) if middle_mask[i]]
            upper = [s for i, s in enumerate(my_volume) if upper_mask[i]]
            top = [s for i, s in enumerate(my_volume) if top_mask[i]]

            with open("ATLAS_lower.csv", "w") as f:
                for line in lower:
                    f.write(f"{line}\n")
            with open("ATLAS_middle.csv", "w") as f:
                for line in middle:
                    f.write(f"{line}\n")
            with open("ATLAS_upper.csv", "w") as f:
                for line in upper:
                    f.write(f"{line}\n")
            with open("ATLAS_top.csv", "w") as f:
                for line in top:
                    f.write(f"{line}\n")


if __name__ == "__main__":
    main()
