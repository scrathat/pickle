"""Speech Dataset class"""

import logging
import os

import numpy as np

import librosa
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler


def get_val_dataset(test_set):
    num_train = len(test_set)
    indices = list(range(num_train))
    split = int(np.floor(0.3 * num_train))
    np.random.seed(42)
    np.random.shuffle(indices)
    test_idx, valid_idx = indices[split:], indices[:split]
    test_sampler = SubsetRandomSampler(test_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    return test_sampler, valid_sampler


class SpeechDataset(Dataset):
    def __init__(self, path):
        self._path = path
        self._filetypes = (".wav", ".flac")
        self._labels = self.list_labels(self._path)
        self._logger = logging.getLogger()
        self._loglevel = self._logger.getEffectiveLevel()
        walker = self.walk_files(self._path)
        self._walker = list(walker)
        self._logger.debug(f"__init__: self.__len__() {self.__len__()}")

    def __getitem__(self, n):
        item = self._walker[n]
        self._logger.debug(f"__getitem__: processing item {n}")
        x, y = self.process(item)
        self._logger.debug(f"__getitem__: x.shape: {x.shape} x: {x}")
        self._logger.debug(f"__getitem__: y.shape: {y.shape} y: {y}")
        return x, y

    def __len__(self):
        return len(self._walker)

    def walk_files(self, path):
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(self._filetypes):
                    yield os.path.join(root, file)

    def list_labels(self, path):
        classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return class_to_idx

    def process(self, item):
        y = self.get_label(item)
        x, sr = librosa.load(item)

        if self._loglevel < logging.INFO:
            mfcc = librosa.feature.mfcc(x, sr=sr, n_mfcc=13)
            self._logger.debug(f"process: mfcc.shape: {mfcc.shape}")
        return x, y

    def get_label(self, item):
        folder = os.path.basename(os.path.dirname(item))
        return self._labels[folder]


class SpokenLanguage(SpeechDataset):
    def __init__(self, path):
        super().__init__(path)
        self.classes = ["en", "de", "es", "m", "f"]
        self._class_to_int = {c: i for i, c in enumerate(self.classes)}

    def get_label(self, item):
        data = os.path.basename(item).split("_")[0:2]
        int_encoded = [self._class_to_int[d] for d in data]
        one_encoded = [0] * len(self.classes)
        for i in int_encoded:
            one_encoded[i] = 1
        return np.asarray(one_encoded)
