from .logger import get_logger

log = get_logger(__name__)


def main() -> None:
    log.info("Hello from julia-log!")
