import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import os
import json

from os.path import join, exists


class BirdsDataset(Dataset):
    def __init__(self, config, device: torch.device):
        birds = {
            1: 'comcuc',
            2: 'cowpig1',
            3: 'eucdov',
            4: 'eueowl1',
            5: 'grswoo',
            6: 'tawowl1'
        }
        dataset = []
        for id, name in birds.items():
            files = []
            for file in os.listdir(join(config.data_path, name)):
                if '.labels' not in file:
                    files.append(file[:-4])
            for file in files:
                # data
                path = join(config.data_path, name, file + '.npy')
                features = np.load(path).T

                # labels
                path = join(config.data_path, name, file + '.labels.npy')
                label = np.load(path)
                count_ones = np.sum(label, axis=1)
                label = np.where(count_ones > label.shape[1] / 2, 1, 0)
                label = label * id
                label = np.eye(7)[label].T

                dataset.append((features, label))

        if len(features) == 0:
            raise AttributeError('Data-path seems to be empty')

        self.device = device
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        features, labels = self.dataset[idx]
        return features.to(self.device), labels.to(self.device)
