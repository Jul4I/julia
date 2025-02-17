import logging
import logging.config

logging.config.fileConfig(
    fname="config.ini",
    disable_existing_loggers=False,
)


# Get the logger specified in the file
def get_logger(python_file_name: str) -> logging.Logger:
    return logging.getLogger(python_file_name)
