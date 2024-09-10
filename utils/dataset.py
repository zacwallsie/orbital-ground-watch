import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split


class NPYDataset(Dataset):
    def __init__(self, data1, data2, labels):
        self.data1 = data1
        self.data2 = data2
        self.labels = labels
        self.mean1, self.std1 = self.compute_mean_std(self.data1)
        self.mean2, self.std2 = self.compute_mean_std(self.data2)

    def compute_mean_std(self, data):
        data_tensor = torch.from_numpy(data).float()
        mean = torch.mean(data_tensor, dim=[0, 2, 3])
        std = torch.std(data_tensor, dim=[0, 2, 3])
        return mean, std

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        image1 = self.data1[idx]
        image2 = self.data2[idx]
        label = self.labels[idx]

        image1 = torch.from_numpy(image1).float()
        image2 = torch.from_numpy(image2).float()

        # Convert one-hot encoded mask to class indices
        label = torch.from_numpy(label).float()
        label = torch.argmax(label, dim=0).long()  # Convert to class indices

        image1 = (image1 - self.mean1.view(-1, 1, 1)) / self.std1.view(-1, 1, 1)
        image2 = (image2 - self.mean2.view(-1, 1, 1)) / self.std2.view(-1, 1, 1)

        return image1, image2, label


class NPYDatasetForPrediction(Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
        self.mean1, self.std1 = self.compute_mean_std(self.data1)
        self.mean2, self.std2 = self.compute_mean_std(self.data2)

    def compute_mean_std(self, data):
        data_tensor = torch.from_numpy(data).float()
        mean = torch.mean(data_tensor, dim=[0, 2, 3])
        std = torch.std(data_tensor, dim=[0, 2, 3])
        return mean, std

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        image1 = self.data1[idx]
        image2 = self.data2[idx]

        image1 = torch.from_numpy(image1).float()
        image2 = torch.from_numpy(image2).float()

        image1 = (image1 - self.mean1.view(-1, 1, 1)) / self.std1.view(-1, 1, 1)
        image2 = (image2 - self.mean2.view(-1, 1, 1)) / self.std2.view(-1, 1, 1)

        return image1, image2


def create_datasets(
    data1_path,
    data2_path,
    labels_path,
    test_split=0.15,
    validation_split=0.15,
    random_state=42,
):
    data1 = np.load(data1_path)
    data2 = np.load(data2_path)
    labels = np.load(labels_path)

    # Ensure all arrays have the same number of samples
    assert (
        data1.shape[0] == data2.shape[0] == labels.shape[0]
    ), "Number of samples in all arrays must match"

    # Ensure the spatial dimensions (H and W) match for all arrays
    assert (
        data1.shape[2:] == data2.shape[2:] == labels.shape[2:]
    ), "Spatial dimensions (H and W) must match for all arrays"

    (
        train_data1,
        test_val_data1,
        train_data2,
        test_val_data2,
        train_labels,
        test_val_labels,
    ) = train_test_split(
        data1,
        data2,
        labels,
        test_size=test_split + validation_split,
        random_state=random_state,
    )

    test_data1, val_data1, test_data2, val_data2, test_labels, val_labels = (
        train_test_split(
            test_val_data1,
            test_val_data2,
            test_val_labels,
            test_size=validation_split / (test_split + validation_split),
            random_state=random_state,
        )
    )

    train_dataset = NPYDataset(train_data1, train_data2, train_labels)
    test_dataset = NPYDataset(test_data1, test_data2, test_labels)
    val_dataset = NPYDataset(val_data1, val_data2, val_labels)

    return train_dataset, test_dataset, val_dataset


def create_entire_dataset(data1_path, data2_path, labels_path):
    data1 = np.load(data1_path)
    data2 = np.load(data2_path)
    labels = np.load(labels_path)

    return NPYDataset(data1, data2, labels)


def create_prediction_dataset(data1_path, data2_path):
    data1 = np.load(data1_path)
    data2 = np.load(data2_path)

    return NPYDatasetForPrediction(data1, data2)
