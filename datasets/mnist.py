from typing import Literal
import torchvision.datasets as datasets
from torchvision import transforms
import torch
import numpy as np
from torch.utils.data import Dataset

from datasets.dataset_base import CustomDataset, DOWNLOAD_FOLDER
from datasets.dataset_helpers import TransformToTensorIfNeeded

# zero mean and unit sd?
# https://stackoverflow.com/questions/63746182/correct-way-of-normalizing-and-scaling-the-mnist-dataset
MNIST_NORM = transforms.Normalize((0.1307,), (0.3081,))
IMAGE_NORM = transforms.Normalize([0.5], [0.5])

class DatasetMNIST(CustomDataset):
    '''
    * an image size other than 28 results in the images being rescaled using transforms.Resize
    * normalization:
        * "none": default [0, 1] range.
        * "mean0": transforms.Normalize((0.1307,), (0.3081,)) to get mean 0, variance 1
        * "05": transforms.Normalize([0.5], [0.5]) since that somehow works best for some GANs
    '''
    def __init__(self, normalization: Literal["none", "mean0", "05"], batch_size: int, img_size: int=28, num_workers: int=2, pin_memory=True, train_set_size=-1, train_transform=None):
        super().__init__(ds_id='mnist', batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, train_set_size=train_set_size, train_transform=train_transform)

        assert normalization in ["none", "mean0", "05"], f'{normalization} is no normalization mode'

        self.normalization = normalization
        self.img_size = img_size

        self.options.update({
            "normalization": normalization,
            "img_size": img_size,
        })

    def _get_child_dataset(self, train) -> Dataset:
        return datasets.MNIST(
            root=DOWNLOAD_FOLDER,
            train=train,
            download=True,
            transform=self._get_transform(train=train),
        )
    
    def _get_transform(self, train):
        if train and self.train_transform != None:
            return self.train_transform
        
        return_transform = []
        return_transform.append(TransformToTensorIfNeeded())
        if self.img_size != 28:
            return_transform.append(transforms.Resize((self.img_size, self.img_size), antialias=False))

        if self.normalization == "mean0":
            return_transform.append(MNIST_NORM)
        elif self.normalization == "05":
            return_transform.append(IMAGE_NORM)

        return transforms.Compose(return_transform)

# MARK: Background
class DatasetMNISTBackground(DatasetMNIST):
    '''
    Add a grey shape to the background of the image. Preloading dataset advised.
    '''
    def __init__(self, normalization: Literal['none','mean0','05'], batch_size: int, correlation = 0, output_label: Literal["bg", "digit", "both"] = "bg", img_size: int = 28, num_workers: int = 0, pin_memory=False, train_set_size=-1, neg_label=-1, target_label_noise_prob=0.):
        super().__init__(normalization, batch_size, img_size, num_workers, pin_memory, train_set_size)
        assert output_label in ["bg", "digit", "both"]
        assert neg_label in [-1, 0]

        self.correlation = correlation
        self.output_label = output_label
        self.neg_label = neg_label
        self.target_label_noise_prob = target_label_noise_prob

        self.options.update({
            "correlation": correlation,
            "output_label": output_label,
            "neg_label": neg_label,
            "target_label_noise_prob": target_label_noise_prob,
        })

    def _get_child_dataset(self, train) -> Dataset:
        base_ds = super()._get_child_dataset(train)
        target_label_noise = self.target_label_noise_prob if train else 0
        return _DatasetMNISTBackground(base_ds, self.img_size, self.correlation, self.output_label, self.neg_label, target_label_noise)

class _DatasetMNISTBackground(Dataset):
    def __init__(self, base_ds, img_size, correlation, output_label, neg_label, target_label_noise_prob=0) -> None:
        super().__init__()

        self.output_label = output_label
        self.neg_label = neg_label
        self.ds = base_ds
        self.count = len(base_ds)


        correlated_labels = torch.randint(low=0, high=9, size=(self.count,)) - self.ds.targets + 5
        uncorrelated_labels = torch.randint(low=0, high=10, size=(self.count,))
        self.bg_labels = ((correlation * correlated_labels + (1 - correlation) * uncorrelated_labels) >= 5).to(int)

        self.digit_labels = self.ds.targets
        if target_label_noise_prob > 0:
            random_target_labels = torch.randint(low=0, high=10, size=(self.count,))
            use_random_label = (torch.rand((self.count,)) < target_label_noise_prob).to(int) # type: ignore
            self.digit_labels = (1 - use_random_label) * self.digit_labels + use_random_label * random_target_labels


        self.shapes = [torch.zeros((img_size, img_size), dtype=torch.float) for _ in range(2)]

        # circle
        grid_x, grid_y = np.mgrid[:img_size, :img_size]
        circle_radius = img_size * 4 
        self.shapes[0][torch.tensor((grid_x - img_size//2) ** 2 + (grid_y - img_size//2) ** 2 < circle_radius)] = .3 

        # rectangle
        rectangle_size = int(img_size // 1.2)
        rectangle_padding = (img_size - rectangle_size)
        self.shapes[1][rectangle_padding:rectangle_size, rectangle_padding:rectangle_size] = .3 

    def __len__(self):
        return self.count
    
    def __getitem__(self, index):
        img = self.ds.__getitem__(index)[0]

        bg_label = self.bg_labels[index].item()
        img = np.minimum(img + self.shapes[bg_label], 1)

        if self.neg_label == -1:
            bg_label = bg_label * 2 - 1

        if self.output_label == "bg":
            return img, bg_label
        elif self.output_label == "digit":
            return img, self.digit_labels[index]
        else:
            return img, torch.tensor((bg_label, self.digit_labels[index])) # tensor instead of tuple is more easily compatible with e.g. preloading