# Import necessary libraries
import tomllib
import logging
import utils.logger_visuals as logger_visuals
from utils.dataset import create_data_loaders
from colorama import init
import warnings
import os
from torch.utils.tensorboard import SummaryWriter
import glob
from utils.misc_utils import AverageMeter, logits_to_mask
from utils.scheduler import WarmupCosineSchedule
from utils.metric_tool import ConfuseMatrixMeter
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.dice_score import dice_loss
import torch.nn.functional as F
import gc
from model_managers.snunet_model_manager import SnuNetModelManager

PATH_CONFIGS = "path_configs.toml"

# Initialize colorama
init(autoreset=True)


def get_path_configs() -> dict:
    logger.info("Getting path configs from current working directory...")

    with open(PATH_CONFIGS, "rb") as f:
        training_paths = tomllib.load(f)

    # Define a list of expected variables
    required_paths = ["images_path", "masks_path", "output_path", "model_configs_path"]

    # Check for each expected variable in the loaded config_data
    missing_vars = [var for var in required_paths if var not in training_paths]
    if missing_vars:
        raise ValueError(
            f"Missing expected path config variables: {', '.join(missing_vars)}"
        )

    logger.info(f"Images Path: {training_paths.get('images_path')}")
    logger.info(f"Masks Path: {training_paths.get('masks_path')}")
    logger.info(f"Output Path: {training_paths.get('output_path')}")
    logger.info(f"Model Configs: {training_paths.get('model_configs_path')}\n")

    return training_paths


def log_training_parameters(config):
    # Log each attribute of the config class
    logger.info("Training parameters")
    for attr in dir(config):
        if not attr.startswith("__") and not callable(getattr(config, attr)):
            logger.info(f"{attr}: {getattr(config, attr)}")

    # Assuming `num_params` is calculated elsewhere in your code
    logger.info("Total Parameter: \t%2.1fM\n" % config.count_parameters())


def valid(config: TrainingConfig, writer, test_loader, global_step, criterion):
    # Validation!
    eval_losses = AverageMeter()
    num_batches = 0

    salEvalVal = ConfuseMatrixMeter(n_class=2)

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", config.get_batch_size)

    config.model.eval()
    epoch_iterator = tqdm(
        test_loader,
        desc="Validating... (loss=X.X)",
        bar_format="{l_bar}{r_bar}",
        dynamic_ncols=True,
    )
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(config.get_device) for t in batch)
        x, y = batch

        with torch.no_grad():
            xA, xB = torch.split(x, 6, dim=1)
            output = config.model(xA, xB)
            masks = y.squeeze(1)

            pred = logits_to_mask(output, config.get_threshold).long()

            # Now compute the loss
            eval_loss = criterion(output.squeeze(1), masks.float())
            eval_loss += dice_loss(
                F.sigmoid(output.squeeze(1)), masks.float(), multiclass=False
            )
            eval_losses.update(eval_loss.item())

            f1 = salEvalVal.update_cm(pr=pred.cpu().numpy(), gt=masks.cpu().numpy())

            num_batches += 1

        epoch_iterator.set_description(
            "Validating... (f1=%3f loss=%2.5f)" % (f1, eval_losses.val)
        )

    scores = salEvalVal.get_scores()

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Kappa: %2.5f" % scores["Kappa"])
    logger.info("IoU: %2.5f" % scores["IoU"])
    logger.info("F1: %2.5f" % scores["F1"])
    logger.info("Overall Accuracy: %2.5f" % scores["OA"])
    logger.info("Recall: %2.5f" % scores["recall"])
    logger.info("Precision: %2.5f" % scores["precision"])

    writer.add_scalar(
        "test/accuracy", scalar_value=scores["OA"], global_step=global_step
    )
    return scores["OA"]


def train_model_from_config(config: TrainingConfig, training_paths: dict):
    os.makedirs(config.get_output_dir(), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", config.get_name()))

    # Prepare dataset
    train_loader, test_loader, val_loader = create_data_loaders(
        data_path=training_paths.get("images_path"),
        mask_path=training_paths.get("masks_path"),
        batch_size=config.get_batch_size(),
    )

    # Prepare optimizer and scheduler
    optimizer = torch.optim.Adam(
        config.model.parameters(),
        lr=config.get_lr(),
        weight_decay=config.get_wd(),
    )

    criterion = nn.BCEWithLogitsLoss()

    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=config.get_warmup_steps(),
        t_total=config.get_num_steps(),
    )

    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", config.get_num_steps())
    logger.info("  Instantaneous batch size per GPU = %d", config.get_batch_size())

    config.model.zero_grad()
    losses = AverageMeter()
    global_step, best_acc = 0, 0

    while True:
        config.model.train()
        epoch_iterator = tqdm(
            train_loader,
            desc="Training (X / X Steps) (loss=X.X)",
            bar_format="{l_bar}{r_bar}",
            dynamic_ncols=True,
        )
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(config.get_device()) for t in batch)
            x, y = batch

            xA, xB = torch.split(x, 6, dim=1)
            output = config.model(xA, xB)
            masks = y.squeeze(1)

            # Now compute the loss
            loss = criterion(output.squeeze(1), masks.float())
            loss += dice_loss(
                F.sigmoid(output.squeeze(1)), masks.float(), multiclass=False
            )

            if (step + 1) % config.get_gradient_accumulation_steps() == 0:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                losses.update(loss.item() * config.get_gradient_accumulation_steps())
                torch.nn.utils.clip_grad_norm_(
                    config.model.parameters(), config.get_max_grad_norm()
                )
                scheduler.step()
                optimizer.step()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)"
                    % (global_step, config.get_num_steps(), losses.val)
                )
                writer.add_scalar(
                    "train/loss", scalar_value=losses.val, global_step=global_step
                )
                writer.add_scalar(
                    "train/lr",
                    scalar_value=scheduler.get_lr()[0],
                    global_step=global_step,
                )

                if global_step % config.get_eval_every() == 0:
                    accuracy = valid(
                        config, writer, test_loader, global_step, criterion
                    )
                    if best_acc < accuracy:
                        config.save_model_checkpoint
                        best_acc = accuracy
                    config.model.train()

            if global_step % config.get_num_steps() == 0:
                break

        losses.reset()
        if global_step % config.get_num_steps() == 0:
            break

    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main():
    # Setup logging
    logger_visuals.setup_logger(__name__, logging.INFO)

    training_paths = get_path_configs()

    logger.info("Getting TOML configs...")
    # Find all TOML files in specified directory
    model_configs_path = training_paths.get("model_configs_path")
    config_files = glob.glob(os.path.join(model_configs_path, "*.toml"))
    logger.info(f"Config Files: {config_files}\n")

    output_dir = os.path.join(training_paths.get("output_path"), "output")

    logger.info("Beginning Bulk Training of Models...\n")
    # Train a model for each configuration
    for config_file in config_files:

        config = TrainingConfig(config_file, output_dir)
        log_training_parameters(config)

        train_model_from_config(config=config, training_paths=training_paths)


if __name__ == "__main__":
    main()
