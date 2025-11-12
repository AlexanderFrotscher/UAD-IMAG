from functools import partial

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torchvision
import torchvision.models as tv_models
import tqdm
from coreset_sampler import ApproximateGreedyCoresetSampler
from feature_merger import Aggregator, Preprocessing

from utils.faiss import FaissNN
from utils.feature_extractor import NetworkFeatureExtractor
from utils.patchmaker import PatchMaker

"""
This code is based on PatchCore
https://github.com/amazon-science/patchcore-inspection/blob/main/src/patchcore/patchcore.py
"""


class PatchCore(torch.nn.Module):
    backbones = {
        "WideResNet50": (partial(tv_models.wide_resnet50_2), "IMAGENET1K_V1"),
        "EfficientNet-B4": (partial(tv_models.efficientnet_b4), "IMAGENET1K_V1"),
    }

    def __init__(self, config):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()

        self.patch_maker = PatchMaker(
            config.model.patchsize, stride=config.model.patchstride
        )
        self.num_neighbors = config.model.num_neighbors
        self.target_size = config.data.image_size
        self.forward_modules = torch.nn.ModuleDict({})
        self.backbone = config.model.backbone
        self.normalize = config.model.normalize

        if self.backbone not in self.backbones:
            raise KeyError(
                f"Invalid backbone {self.backbone}. "
                "You can use the string WideResNet50 or EfficientNet-B4."
            )

        model, weights = self.backbones[self.backbone]
        extractor = NetworkFeatureExtractor(
            model(weights=weights),
            config.model.layers,
        )
        extractor.eval()
        self.layers_to_extract_from = config.model.layers
        feature_dimensions, _ = extractor.feature_dimensions(config.model.input_shape)
        self.forward_modules["feature_aggregator"] = extractor

        preprocessing = Preprocessing(
            feature_dimensions, config.model.pretrain_embed_dim
        )
        self.forward_modules["preprocessing"] = preprocessing

        preadapt_aggregator = Aggregator(target_dim=config.model.target_embed_dim)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.featuresampler = ApproximateGreedyCoresetSampler(config.model.sample_size)
        self.nn_method = FaissNN(config.model.faiss_gpu, config.data.workers)

    def forward(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            normalize = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
                if self.normalize:
                    images = normalize(images)
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data, file):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)
        self.nn_method.save(file)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                # input_image = input_image[None].to(torch.float)
                return self(input_image)

        features = []
        if type(input_data) in (
            torch.utils.data.DataLoader,
            accelerate.data_loader.DataLoaderShard,
        ):
            with tqdm.tqdm(
                input_data,
                desc="Computing support features...",
                position=1,
                leave=False,
            ) as data_iterator:
                for image in data_iterator:
                    if isinstance(image, dict):
                        image = image["image"]
                    features.append(_image_to_features(image))
        else:
            features.append(_image_to_features(input_data))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)
        self.nn_method.fit(features)

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self(images, provide_patch_shapes=True)
            features = np.asarray(features)

            patch_scores = image_scores = np.mean(
                self.nn_method.run(self.num_neighbors, features)[0], axis=-1
            )
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            # masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores
            _scores = _scores.unsqueeze(1)
            _scores = F.interpolate(
                _scores, size=self.target_size, mode="bilinear", align_corners=False
            )
            masks = _scores.cpu()

        return image_scores, masks
