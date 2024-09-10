import logging
from colorama import Fore, Style, init
import os

# Initialize colorama
init(autoreset=True)


class CustomFormatter(logging.Formatter):
    """Custom logging formatter with color"""

    grey = Style.DIM + Fore.WHITE
    green = Fore.GREEN
    yellow = Fore.YELLOW
    red = Fore.RED
    bold_red = Style.BRIGHT + Fore.RED
    reset = Style.RESET_ALL

    FORMATS = {
        logging.DEBUG: grey
        + "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        + reset,
        logging.INFO: green
        + "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        + reset,
        logging.WARNING: yellow
        + "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        + reset,
        logging.ERROR: red
        + "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        + reset,
        logging.CRITICAL: bold_red
        + "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%m/%d/%Y %H:%M:%S")
        return formatter.format(record)


def setup_logger(name, level, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Create and add the custom handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    # Create the log file path
    log_file_path = os.path.join(output_dir, "log.txt")

    # Create the file handler
    file_handler = logging.FileHandler(log_file_path, mode="a")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    logger.info(f"Logging to file: {log_file_path}")

    return logger
