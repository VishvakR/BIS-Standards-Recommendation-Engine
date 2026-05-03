import logging


class Logger:
    """Centralized logging setup for AskTheSite."""

    @staticmethod
    def setup(level=logging.INFO):
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )