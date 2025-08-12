import os

import numpy as np

import torch

from glob import glob
from torch.utils.data import Dataset


class ct_dataset(Dataset):
    def __init__(self, data_path, min_value=0.0, max_value=2500.0, augmentation=False):
        self.files = sorted(glob(os.path.join(data_path, '*.npz')))
        self.min_value = min_value
        self.max_value = max_value
        self.augmentation = augmentation

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        self.path = self.files[idx]
        self.data = np.load(self.path)

        input_image = self.data['input']
        target_image = self.data['target']

        if self.augmentation:
            seed = np.random.randint(43)
            random_state = np.random.RandomState(seed)

            if random_state.rand() < 0.5:
                input_image = np.fliplr(input_image)
                target_image = np.fliplr(target_image)

        self.input = self.normalize(input_image)
        self.target = self.normalize(target_image)

        self.input = torch.tensor(self.input.astype(np.float32))
        self.target = torch.tensor(self.target.astype(np.float32))

        self.input = self.input.unsqueeze(0)
        self.target = self.target.unsqueeze(0)

        return (self.input, self.target)

    def normalize(self, image):

        return (image - self.min_value) / (self.max_value - self.min_value)
