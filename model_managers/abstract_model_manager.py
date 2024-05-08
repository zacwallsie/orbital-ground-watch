import torch
import os
import logging
import random
import numpy as np
import utils.logger_visuals as logger_visuals


class AbstractModelManager:
    def __init__(self, model_config_path, output_dir):
        self.config_params = self.load_config(model_config_path)
        self.model_name = self._generate_name()
        self.output_dir = os.path.join(output_dir, self.model_name)
        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.init_model()
        self.model.to(self.device)
        self.init_logger()

    def load_config(self, model_config_path) -> dict:
        """Loads the configuration from a file."""
        # This will varying depending on the required configuration for the specific model
        raise NotImplementedError

    def _generate_name(self) -> str:
        """Generates a unique name for the model based on configuration."""
        raise NotImplementedError

    def init_model(self) -> torch.nn.Module:
        """Initializes the model using the configuration."""
        raise NotImplementedError

    def init_logger(self) -> None:
        """Initializes logging."""
        self.logger = logger_visuals.setup_logger(
            __name__, logging.INFO, self.output_dir
        )
        return

    def save(self, path) -> None:
        """Saves the model to a given path."""
        torch.save(self.model.state_dict(), path)
        return

    def load(self, path) -> None:
        """Loads the model from a given path."""
        self.model.load_state_dict(torch.load(path))
        return

    def predict(self, X):
        """Runs model inference on the input data."""
        raise NotImplementedError

    def evaluate(self, X, y):
        """Evaluates the model's performance on the provided data."""
        raise NotImplementedError

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
        """Trains the model."""
        raise NotImplementedError

    def set_mode(self, mode="train") -> None:
        """Sets the model to training or evaluation mode."""
        if mode == "train":
            self.model.train()
        elif mode == "eval":
            self.model.eval()
        else:
            raise ValueError(f"Unknown mode {mode}")
        return

    def set_seed(self, seed) -> None:
        """Sets the seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return

    def log_metrics(self, metrics) -> None:
        """Logs training metrics."""
        raise NotImplementedError

    def ensure_output_directory(self) -> None:
        """Ensure that the output directory exists if not create it."""
        os.makedirs(self.output_dir, exist_ok=True)
        return

    def get_path_configs(self) -> dict:
        """Returns the paths for the model."""
        raise NotImplementedError

    def log_training_parameters(self):
        """Logs all training parameters and the number of trainable parameters."""
        self.logger.info("Training parameters:")
        for key, value in self.config_params.items():
            self.logger.info(f"{key}: {value}")

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total Trainable Parameters: {num_params/1e6:.2f}M")
