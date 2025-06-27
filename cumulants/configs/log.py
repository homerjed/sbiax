import logging
import os

LOG_DIR = os.getenv("LOG_DIR", "logs/")


def get_log_level(default="INFO"):

    level_str = os.getenv("LOG_LEVEL", default).upper()

    return getattr(logging, level_str, logging.INFO)


def setup_module_logger(module_name: str, level=logging.INFO, log_dir=LOG_DIR):

    log_figs_dir = os.path.join(log_dir, "figs/")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_figs_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{module_name}.log")

    try:
        if os.path.exists(log_path):
            os.remove(log_path)
    except Exception as e:
        print(f"LOGS: Failed to delete {log_path}. Reason: {e}")

    logger = logging.getLogger(module_name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger already configured
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)

        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )

        logger.addHandler(file_handler)

    return logger, log_figs_dir