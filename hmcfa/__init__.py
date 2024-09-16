import logging
import yaml
import pathlib
from logging.config import dictConfig

__version__ = "0.0.1"

def get_logger(logger_name: str = "hmcfa"):
    """Get a logger instance."""
    # Define the path to the logging configuration file
    log_config_path = pathlib.Path(__file__).parent / "logger.yaml"

    # Load the logging configuration
    with open(log_config_path, "r") as file:
        config = yaml.safe_load(file)
        dictConfig(config)

    logger = logging.getLogger(logger_name)
    return logger

if __name__ == "__main__":
    logger = get_logger()
    logger.info("This is a test log message.")