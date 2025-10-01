import sys
import logging
import os

LOG_DIR = os.getenv("LOG_DIR", "logs/")
PRINT_LOGS = os.getenv("PRINT_LOGS", False)


def get_log_level(default="DEBUG"):

    level_str = os.getenv("LOG_LEVEL", default).upper()

    return getattr(logging, level_str, logging.INFO)


def setup_module_logger(
    module_name: str, 
    level=logging.INFO, 
    log_dir=LOG_DIR
) -> tuple[logging.Logger, str]:

    log_figs_dir = os.path.join(log_dir, "figs/")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_figs_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{module_name}.log")

    print("LOG PATH:\n\t{}".format(os.path.abspath(log_path)))

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
                '%(name)s - %(levelname)s \n >> %(message)s' # %(asctime)s - 
            )
        )

        logger.addHandler(file_handler)

    if PRINT_LOGS:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter(
                '%(name)s - %(levelname)s \n >> %(message)s' # %(asctime)s - 
            )
        )
        logger.addHandler(handler)

    return logger, log_figs_dir