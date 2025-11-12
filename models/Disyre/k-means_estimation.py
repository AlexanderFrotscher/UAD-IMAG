__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"


import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.dataloaders import LMDBDataset


def main():
    with torch.no_grad():
        my_transforms = transforms.Compose([transforms.CenterCrop((224, 224))])
        dataset = LMDBDataset(
            "D:\\DokumenteD\\Uni\\Data\\IXI_BIDS_processed\\slices_T1w", my_transforms
        )
        batch_size = dataset.__len__()
        dataloader = DataLoader(dataset, batch_size, shuffle=True)
        my_iter = iter(dataloader)
        images = next(my_iter)
        images = images.view(-1)
        images = images[images.nonzero()]
        images = images.numpy()
        kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(images)
        print(kmeans.cluster_centers_)


if __name__ == "__main__":
    main()
