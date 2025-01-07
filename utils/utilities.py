import logging
import yaml
import os

class Logger:
    """
    Custom logger for the AdFlux Engine.
    """
    def __init__(self, log_file="adflux.log"):
        self.logger = logging.getLogger("AdFluxEngine")
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log(self, message, level="info"):
        if level == "info":
            self.logger.info(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "warning":
            self.logger.warning(message)
        else:
            self.logger.debug(message)


class ConfigLoader:
    """
    Loads configuration settings from a YAML file.
    """
    @staticmethod
    def load_config(config_file):
        with open(config_file, "r") as file:
            return yaml.safe_load(file)

