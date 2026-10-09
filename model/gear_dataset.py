import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class GearboxDataset(Dataset):
    def __init__(
        self,
        data_path,
        label_path,
        transform=None,
        training=False,
        use_fft=False,
    ):
        self.transform = transform
        self.training = training
        self.use_fft = use_fft

        if not os.path.exists(data_path):
            raise FileNotFoundError(data_path)

        if not os.path.exists(label_path):
            raise FileNotFoundError(label_path)

        try:
            data = pd.read_csv(data_path, header=None).values
            if isinstance(data[0, 0], str):
                data = pd.read_csv(data_path, header=0).values
        except Exception:
            data = pd.read_csv(data_path, header=0).values

        try:
            labels_onehot = pd.read_csv(label_path).values
            if isinstance(labels_onehot[0, 0], str):
                labels_onehot = pd.read_csv(label_path, header=0).values
        except Exception:
            labels_onehot = pd.read_csv(label_path, header=0).values

        labels = np.argmax(labels_onehot, axis=1)

        length = min(len(data), len(labels))
        self.data = data[:length]
        self.labels = labels[:length]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index].astype(np.float32)
        label = np.int64(self.labels[index])

        if self.use_fft:
            fft_values = np.fft.fft(sample)
            sample = np.abs(fft_values) / len(sample)
            sample = sample[: len(sample) // 2]
            sample = np.log1p(sample)
            sample = (
                sample - np.min(sample)
            ) / (
                np.max(sample) - np.min(sample) + 1e-6
            )
        else:
            mean = np.mean(sample)
            std = np.std(sample) + 1e-6
            sample = (sample - mean) / std


        sample = sample.astype(np.float32).reshape(1, -1)

        if self.transform is not None:
            sample = self.transform(sample)

        return torch.from_numpy(sample), torch.tensor(label)


def get_multi_datasets(
    source_data,
    source_label,
    target1_data,
    target1_label,
    target2_data,
    target2_label,
    use_fft=False,
):
    source_dataset = GearboxDataset(
        source_data,
        source_label,
        training=True,
        use_fft=use_fft,
    )

    target1_train = GearboxDataset(
        target1_data,
        target1_label,
        training=True,
        use_fft=use_fft,
    )
    target1_test = GearboxDataset(
        target1_data,
        target1_label,
        training=False,
        use_fft=use_fft,
    )

    target2_train = GearboxDataset(
        target2_data,
        target2_label,
        training=True,
        use_fft=use_fft,
    )
    target2_test = GearboxDataset(
        target2_data,
        target2_label,
        training=False,
        use_fft=use_fft,
    )

    return (
        source_dataset,
        target1_train,
        target1_test,
        target2_train,
        target2_test,
    )
