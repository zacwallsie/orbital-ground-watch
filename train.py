# Future imports for compatibility
from __future__ import absolute_import, division, print_function

# Standard library imports
import argparse
import logging
import os
import random
import warnings
from datetime import timedelta

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from colorama import init

# Local application/library specific imports
from models.sunet import SNUNet_ECAM
from utils.dataset import create_data_loaders
from utils.dice_score import dice_loss
from utils.metric_tool import ConfuseMatrixMeter
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
import utils.logger_visuals as logger_visuals


# Initialize colorama
init(autoreset=True)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


def logits_to_mask(logits, threshold=0.6):
    # Apply sigmoid to convert logits to probabilities
    probabilities = torch.sigmoid(logits)

    # Apply threshold to convert probabilities to binary mask
    mask = probabilities > threshold

    return (
        mask.float()
    )  # Convert to float tensor for further processing or visualization


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(args, model):
    model_to_save = model.module if hasattr(model, "module") else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    model = SNUNet_ECAM(in_channels=6, num_classes=1, base_channel=72)
    if args.pretrained_dir is not None:
        model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step, criterion):
    # Validation!
    eval_losses = AverageMeter()
    num_batches = 0

    salEvalVal = ConfuseMatrixMeter(n_class=2)

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    epoch_iterator = tqdm(
        test_loader,
        desc="Validating... (loss=X.X)",
        bar_format="{l_bar}{r_bar}",
        dynamic_ncols=True,
        disable=args.local_rank not in [-1, 0],
    )
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch

        with torch.no_grad():
            xA, xB = torch.split(x, 6, dim=1)
            output = model(xA, xB)
            masks = y.squeeze(1)

            pred = logits_to_mask(output, args.threshold).long()

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


def train(args, model):
    """Train the model"""
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader, val_loader = create_data_loaders(
        args.images_path, args.mask_path, args.train_batch_size
    )

    # Prepare optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    criterion = nn.BCEWithLogitsLoss()

    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(
            optimizer, warmup_steps=args.warmup_steps, t_total=t_total
        )
    else:
        scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=args.warmup_steps, t_total=t_total
        )

    t_total = args.num_steps

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    while True:
        model.train()
        epoch_iterator = tqdm(
            train_loader,
            desc="Training (X / X Steps) (loss=X.X)",
            bar_format="{l_bar}{r_bar}",
            dynamic_ncols=True,
            disable=args.local_rank not in [-1, 0],
        )
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch

            xA, xB = torch.split(x, 6, dim=1)
            output = model(xA, xB)
            masks = y.squeeze(1)

            # Now compute the loss
            loss = criterion(output.squeeze(1), masks.float())
            loss += dice_loss(
                F.sigmoid(output.squeeze(1)), masks.float(), multiclass=False
            )

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                losses.update(loss.item() * args.gradient_accumulation_steps)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)"
                    % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar(
                        "train/loss", scalar_value=losses.val, global_step=global_step
                    )
                    writer.add_scalar(
                        "train/lr",
                        scalar_value=scheduler.get_lr()[0],
                        global_step=global_step,
                    )

                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    accuracy = valid(
                        args, model, writer, test_loader, global_step, criterion
                    )
                    if best_acc < accuracy:
                        save_model(args, model)
                        best_acc = accuracy
                    model.train()

            if global_step % t_total == 0:
                break

        losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--name", required=True, help="Name of this run. Used for monitoring."
    )
    parser.add_argument(
        "--images_path",
        type=str,
        required=True,
        help="The path to the .npy file containing the un-enhanced image data",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        required=True,
        help="The path to the .npy file containing the un-enhanced image data",
    )
    parser.add_argument(
        "--pretrained_dir",
        type=str,
        default=None,
        help="Where to search for pretrained ViT models.",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        type=str,
        help="The output directory where checkpoints will be written.",
    )
    parser.add_argument("--img_size", default=512, type=int, help="Resolution size")
    parser.add_argument(
        "--train_batch_size",
        default=2,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size", default=2, type=int, help="Total batch size for eval."
    )
    parser.add_argument(
        "--eval_every",
        default=100,
        type=int,
        help="Run prediction on validation set every so many steps."
        "Will always run one evaluation at the end of training.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-3,
        type=float,
        help="The initial learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        default=1e-8,
        type=float,
        help="Weight decay if we apply some.",
    )
    parser.add_argument(
        "--num_steps",
        default=15000,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--decay_type",
        choices=["cosine", "linear"],
        default="linear",
        help="How to decay the learning rate.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=30,
        type=int,
        help="Step of training to perform learning rate warmup for.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for selecting the top class.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n",
    )
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", timeout=timedelta(minutes=60)
        )
        args.n_gpu = 1
    args.device = device

    # Setup logging
    level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN
    logger_visuals.setup_logger(__name__, level)

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training & Validation
    train(args, model)


if __name__ == "__main__":
    main()
