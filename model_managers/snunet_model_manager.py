import torch
import tomllib
import os
from models.sunet import SNUNet_ECAM
from abstract_model_manager import AbstractModelManager


class SnuNetModelManager(AbstractModelManager):

    def __init__(self, toml_file_path, output_dir):
        super().__init__(
            toml_file_path, output_dir
        )  # Call to the abstract class constructor

    def _generate_name(self):
        # Generate a unique model name based on configurations
        return f"{self.name}_basechannel-{self.base_channel}_depth-{self.depth}_pretrained-{True if self.pretrained_dir else False}_imgsize-{self.img_size}_lr-{self.learning_rate}_wd-{self.weight_decay}"

    def load_config(self, toml_file_path):
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

    def predict(self, X):
        # Implement model prediction on input data X
        self.set_mode("eval")
        with torch.no_grad():
            return self.model(X.to(self.device))

    def evaluate(self, X, y):
        # Evaluate the model performance on the given dataset
        self.set_mode("eval")
        with torch.no_grad():
            predictions = self.model(X.to(self.device))
            # Assuming some evaluation metric function `evaluate_metric`
            return evaluate_metric(predictions, y)

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
        # Train the model
        self.set_mode("train")
        global_step, best_acc = 0, 0

        for step in range(num_steps):
            for batch in train_loader:
                batch = tuple(t.to(self.device) for t in batch)
                x, y = batch
                output = self.model(x)
                loss = criterion(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.update(loss.item())
                global_step += 1

                if global_step % self.eval_every == 0:
                    metrics = self.evaluate(valid_loader)
                    self.log_metrics(metrics)
                    if metrics["accuracy"] > best_acc:
                        best_acc = metrics["accuracy"]
                        self.save(os.path.join(self.output_dir, "best_model.pth"))
                    self.save(os.path.join(self.output_dir, "last_model.pth"))

        return
