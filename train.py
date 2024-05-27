import torch
import tomllib
import os
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from model_managers import snunet_model_manager
from utils.misc_utils import AverageMeter, logits_to_mask
from utils.metric_tool import ConfuseMatrixMeter
from utils.dataset import create_data_loaders

PATHS_CONFIG = "run_configs.toml"


def load_paths_config() -> dict:
    with open(PATHS_CONFIG, "rb") as f:
        paths_config = tomllib.load(f)

    required_paths = ["images_path", "masks_path", "output_dir", "model_config_path"]
    missing_paths = [path for path in required_paths if path not in paths_config]
    if missing_paths:
        raise ValueError(
            f"Missing expected path config variables: {', '.join(missing_paths)}"
        )

    return paths_config


# Main script
def main():
    # Load configuration
    path_configs = load_paths_config()
    model_manager = snunet_model_manager(
        path_configs["model_config_path"], path_configs["output_path"]
    )

    # Load dataset and create data loaders
    dataset = create_data_loaders(
        path_configs["images_path"],
        path_configs["masks_path"],
        model_manager.batch_size,
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Define loss function, optimizer, and scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)

    # Train the model
    losses = AverageMeter()
    model_manager.train(
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        config["num_steps"],
        losses,
    )


if __name__ == "__main__":
    main()
