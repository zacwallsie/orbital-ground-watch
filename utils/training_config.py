import torch
import tomllib
import os
import random
import numpy as np
from models.sunet import SNUNet_ECAM
import shutil


class TrainingConfig:

    def __init__(self, toml_file_path, output_dir):
        self._load_config(toml_file_path)

        self.current_run_number = None
        self.model_name = self._generate_name()
        self.output_dir = os.path.join(output_dir, self.model_name)
        self.model_output_dir = os.path.join(self.output_dir, self.model_name)
        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._set_seed()
        self._initialise_model()

    def _generate_name(self) -> str:
        return f"{self.name}_basechannel-{self.base_channel}_depth-{self.depth}_pretrained-{True if self.pretrained_dir else False}_imgsize-{self.img_size}_lr-{self.learning_rate}_wd-{self.weight_decay}"

    def _initialise_model(self):
        self.model = SNUNet_ECAM(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_channel=self.base_channel,
            depth=self.depth,
        )
        if len(self.pretrained_dir) != 0:
            self.model.load_state_dict(torch.load(self.pretrained_dir), strict=False)
        self.model.to(self.device)

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def _load_config(self, toml_file_path):
        with open(toml_file_path, "rb") as f:
            config_data = tomllib.load(f)

        # Define a list of expected variables
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

        # Check for each expected variable in the loaded config_data
        missing_vars = [var for var in required_vars if var not in config_data]
        if missing_vars:
            raise ValueError(
                f"Missing expected config variables: {', '.join(missing_vars)}"
            )

        # Assuming the TOML structure is flat for simplicity. Adjust as needed.
        for key, value in config_data.items():
            setattr(self, key, value)

    def save_model_checkpoint(self):
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        if self.current_run_number is None:
            # Determine the next run number only if it's not already set
            existing_files = [
                f
                for f in os.listdir(self.output_dir)
                if os.path.isfile(os.path.join(self.output_dir, f))
            ]
            model_files = [
                f
                for f in existing_files
                if f.startswith(f"{self.name}") and f.endswith(".bin")
            ]
            runs = [
                int(f.split("_run-")[1].split(".bin")[0])
                for f in model_files
                if "_run-" in f
            ]
            self.current_run_number = max(runs) + 1 if runs else 1
            # Copy the TOML file to the output directory on the first run
            config_copy_path = os.path.join(self.output_dir, f"{self.name}_config.toml")
            shutil.copy(self.toml_file_path, config_copy_path)

        # Use the current run number in the file name
        model_file_name = (
            f"{self.name}_run-{self.current_run_number}.bin"
            if runs
            else f"{self.name}.bin"
        )

        # Define the full path for the new model checkpoint
        model_checkpoint = os.path.join(self.output_dir, model_file_name)

        # Save the model checkpoint
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        torch.save(model_to_save.state_dict(), model_checkpoint)
        print(f"Model checkpoint saved to {model_checkpoint}")

    def count_parameters(self):
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return params / 1000000

    def get_name(self):
        return self.name

    def get_device(self):
        return self.device

    def get_batch_size(self):
        return self.batch_size

    def get_lr(self):
        return self.learning_rate

    def get_wd(self):
        return self.weight_decay

    def get_eval_every(self):
        return self.eval_every

    def get_num_steps(self):
        return self.num_steps

    def get_warmup_steps(self):
        return self.warmup_steps

    def get_gradient_accumulation_steps(self):
        return self.gradient_accumulation_steps

    def get_max_grad_norm(self):
        return self.max_grad_norm

    def get_threshold(self):
        return self.threshold

    def get_output_dir(self):
        return self.output_dir
