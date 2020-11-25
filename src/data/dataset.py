import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as tvtransforms
from PIL import Image

from .gaussian_blur import GaussianBlur
from . import new_transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, x, data_config):
        self.x = x
        self.image_size = data_config['image_size']
        self.s = data_config['s']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x = Image.fromarray(self.x[item])

        color_jitter_ = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        random_resized_crop = transforms.RandomResizedCrop(size=self.image_size)
        random_horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
        color_jitter = transforms.RandomApply([color_jitter_], p=0.8)
        random_grayscale = transforms.RandomGrayscale(p=0.2)
        gaussian_blur = GaussianBlur(kernel_size=int(0.1 * self.image_size))
        to_tensor = transforms.ToTensor()

        t = tvtransforms.Compose([
            random_resized_crop,
            random_horizontal_flip,
            color_jitter,
            random_grayscale,
            gaussian_blur,
            to_tensor
        ])

        x1 = t(x)
        x2 = t(x)

        return x1, x2


