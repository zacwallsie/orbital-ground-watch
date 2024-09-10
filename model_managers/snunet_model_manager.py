import torch
import tomllib
import os
from models.sunet import SNUNet_ECAM
from utils.misc_utils import AverageMeter, logits_to_mask
from ._abstract_model_manager import AbstractModelManager
from utils.metric_tool import ConfuseMatrixMeter


class SnuNetModelManager(AbstractModelManager):

    def __init__(self, toml_file_path, output_dir):
        super().__init__(
            toml_file_path, output_dir
        )  # Call to the abstract class constructor
        self.logger.info(
            f"Initialized SnuNetModelManager with output directory: {output_dir}"
        )

    def _generate_name(self):
        # Generate a unique model name based on configurations
        return f"{self.name}_basechannel-{self.base_channel}_depth-{self.depth}_pretrained-{True if self.pretrained_dir else False}_imgsize-{self.img_size}_lr-{self.learning_rate}_wd-{self.weight_decay}"

    def load_model_config(self, toml_file_path):
        # Load model configuration from a TOML file
        with open(toml_file_path, "rb") as f:
            config_data = tomllib.load(f)

        # Check for required variables in the configuration
        required_vars = [
            "name",
            "in_channels",
            "num_classes",
            "threshold",
            "base_channel",
            "depth",
            "pretrained_dir",
            "img_size",
            "batch_size",
            "eval_every",
            "num_steps",
            "warmup_steps",
            "gradient_accumulation_steps",
            "max_grad_norm",
            "learning_rate",
            "weight_decay",
            "seed",
        ]
        missing_vars = [var for var in required_vars if var not in config_data]
        if missing_vars:
            raise ValueError(
                f"Missing expected config variables: {', '.join(missing_vars)}"
            )

        for key, value in config_data.items():
            setattr(self, key, value)

        return config_data

    def init_model(self):
        # Initialize the model
        model = SNUNet_ECAM(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_channel=self.base_channel,
            depth=self.depth,
        )
        if self.pretrained_dir and os.path.exists(self.pretrained_dir):
            model.load_state_dict(torch.load(self.pretrained_dir), strict=False)
        return model

    def predict(self, xA, xB):
        # Implement model prediction on input data X
        self.set_mode("eval")
        with torch.no_grad():
            return self.model(xA.to(self.device), xB.to(self.device))

    def evaluate(self, test_loader):
        eval_losses = AverageMeter()
        salEval = ConfuseMatrixMeter(n_class=2)

        self.set_mode("eval")
        with torch.no_grad():
            for batch in test_loader:
                batch = tuple(t.to(self.device) for t in batch)
                xA, xB, y = batch
                predictions = self.model(xA, xB)
                loss = self.criterion(predictions, y)
                eval_losses.update(loss.item())
                salEval.add(predictions, y)

        scores = salEval.get_scores()

        return {
            "accuracy": scores["accuracy"],
            "precision": scores["precision"],
            "recall": scores["recall"],
            "f1": scores["f1"],
            "loss": eval_losses.avg,
        }

    def log_metrics(self, metrics):
        # Log metrics during training
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value}")

    def train(
        self,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        scheduler,
        num_steps,
        losses,
    ):
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.set_mode("train")
        best_acc = 0

        for step in range(num_steps):
            self.logger.info(f"Step {step+1}/{num_steps}")
            batch_losses = AverageMeter()

            for batch_idx, batch in enumerate(train_loader):
                batch = tuple(t.to(self.device) for t in batch)
                xA, xB, y = batch

                output = self.model(xA, xB)

                loss = criterion(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                batch_losses.update(loss.item())
                losses.update(loss.item())

                if (batch_idx + 1) % 100 == 0:  # Log every 100 batches
                    self.logger.info(f"  Batch {batch_idx+1}: Loss: {loss.item():.4f}")

            self.logger.info(
                f"Step {step+1} completed. Average Loss: {batch_losses.avg:.4f}"
            )

            if (step + 1) % self.eval_every == 0:
                self.logger.info(f"Performing evaluation at step {step+1}")
                metrics = self.evaluate(valid_loader)
                self.log_metrics(metrics)
                if metrics["accuracy"] > best_acc:
                    best_acc = metrics["accuracy"]
                    self.logger.info(
                        f"New best accuracy: {best_acc:.4f}. Saving best model."
                    )
                    self.save(os.path.join(self.output_dir, "best_model_weights.pth"))
                self.save(os.path.join(self.output_dir, "last_model_weights.pth"))

        losses.reset()
        self.logger.info("Training completed")
        return

    def save(self, path):
        self.logger.info(f"Saving model to {path}")
        torch.save(self.model.state_dict(), path)
        self.logger.info("Model saved successfully")
