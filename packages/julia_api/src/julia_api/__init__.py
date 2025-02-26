from julia_log.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    logger.debug("Hello from julia-api!")
