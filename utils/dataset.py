import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split


class NPYDataset(Dataset):
    def __init__(self, data, masks):
        self.data = data
        self.masks = masks

        self.mean, self.std = self.compute_mean_std()

    def compute_mean_std(self):
        # Convert to PyTorch tensor and reshape
        data_tensor = torch.from_numpy(self.data).float()
        mean = torch.mean(data_tensor, dim=[0, 2, 3])
        std = torch.std(data_tensor, dim=[0, 2, 3])
        return mean, std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        mask = self.masks[idx]

        # Convert numpy arrays to PyTorch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        # Normalize images with calculated mean and std
        image = (image - self.mean.view(-1, 1, 1)) / self.std.view(-1, 1, 1)

        return image, mask


class NPYDatasetForPrediction(Dataset):
    def __init__(self, data):
        self.data = data
        self.mean, self.std = self.compute_mean_std()

    def compute_mean_std(self):
        # Convert to PyTorch tensor and reshape
        data_tensor = torch.from_numpy(self.data).float()
        mean = torch.mean(data_tensor, dim=[0, 2, 3])
        std = torch.std(data_tensor, dim=[0, 2, 3])
        return mean, std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]

        # Convert numpy array to PyTorch tensor
        image = torch.from_numpy(image).float()

        # Normalize images with calculated mean and std
        image = (image - self.mean.view(-1, 1, 1)) / self.std.view(-1, 1, 1)

        return image


def create_data_loaders(
    data_path,
    mask_path,
    batch_size,
    test_split=0.15,
    validation_split=0.15,
    random_state=42,
):
    # Load data from .npy files
    all_data = np.load(data_path)
    all_masks = np.load(mask_path)

    permuted_data = np.transpose(all_data, (0, 3, 1, 2))
    permuted_masks = np.transpose(all_masks, (0, 3, 1, 2))

    # First split: separate out the test set
    # Note we take the validation in this split too as a subset will then be taken for val
    train_data, test_val_data, train_masks, test_val_masks = train_test_split(
        permuted_data,
        permuted_masks,
        test_size=test_split + validation_split,
        random_state=random_state,
    )
    # Further split the training data into training and validation sets
    test_data, val_data, test_masks, val_masks = train_test_split(
        test_val_data,
        test_val_masks,
        test_size=1 - (validation_split / (test_split + validation_split)),
        random_state=random_state,
    )

    # Create dataset objects
    train_dataset = NPYDataset(train_data, train_masks)
    test_dataset = NPYDataset(test_data, test_masks)
    val_dataset = NPYDataset(val_data, val_masks)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader


def create_entire_set_loaders(data_path, mask_path, batch_size):
    # Load data from .npy files
    all_data = np.load(data_path)
    all_masks = np.load(mask_path)

    # Create dataset objects
    total_dataset = NPYDataset(all_data, all_masks)

    total_loader = DataLoader(total_dataset, batch_size=batch_size, shuffle=False)

    return total_loader


def create_entire_prediction_loader(data_path, batch_size):
    # Load data from .npy file
    all_data = np.load(data_path)

    permuted_data = np.transpose(all_data, (0, 3, 1, 2))

    # Create dataset object
    prediction_dataset = NPYDatasetForPrediction(permuted_data)

    # Create DataLoader
    prediction_loader = DataLoader(
        prediction_dataset, batch_size=batch_size, shuffle=False
    )

    return prediction_loader
