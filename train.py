import torch
import tomllib
import os
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from model_managers.snunet_model_manager import SnuNetModelManager
from utils.misc_utils import AverageMeter
from utils.dataset import create_datasets

PATHS_CONFIG = "run_configs.toml"


def load_paths_config() -> dict:
    with open(PATHS_CONFIG, "rb") as f:
        paths_config = tomllib.load(f)

    required_paths = ["xA_path", "xB_path", "y_path", "output_dir", "model_config_path"]
    missing_paths = [path for path in required_paths if path not in paths_config]
    if missing_paths:
        raise ValueError(
            f"Missing expected path config variables: {', '.join(missing_paths)}"
        )

    return paths_config


def main():
    # Load configuration
    path_configs = load_paths_config()
    model_manager = SnuNetModelManager(
        path_configs["model_config_path"], path_configs["output_dir"]
    )

    # Create datasets
    train_dataset, test_dataset, val_dataset = create_datasets(
        path_configs["xA_path"],
        path_configs["xB_path"],
        path_configs["y_path"],
        test_split=0.15,
        validation_split=0.15,
        random_state=42,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_manager.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=model_manager.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=model_manager.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Define loss function, optimizer, and scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(
        model_manager.model.parameters(),
        lr=model_manager.learning_rate,
        weight_decay=model_manager.weight_decay,
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
        model_manager.num_steps,
        losses,
    )

    # Final evaluation
    final_metrics = model_manager.evaluate(test_loader)
    model_manager.log_metrics(final_metrics)
    model_manager.logger.info("Training completed.")


if __name__ == "__main__":
    main()
