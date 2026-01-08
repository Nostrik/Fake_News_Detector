import logging
import sys
from pathlib import Path
from colorlog import ColoredFormatter

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CSV_PATH = DATA_DIR / "fake_news.csv"
LOG_FILE = ROOT_DIR / "project.log"
LOG_LEVEL = logging.INFO  # logging.DEBUG or logging.INFO


def setup_logging():
    LOG_FILE = "app.log"

    # Используем прямые ANSI-коды для стабильности:
    # \033[33m - желтый
    # \033[0m  - сброс цвета
    console_format = (
        "\033[33m%(asctime)s\033[0m "  # Желтая дата
        "%(log_color)s[%(levelname)s] "  # Цветной уровень
        "\033[35m%(name)s\033[0m: "  # Фиолетовый (пурпурный) логгер
        "%(log_color)s%(message)s"  # Сообщение цветом уровня
    )

    formatter = ColoredFormatter(
        console_format,
        datefmt='%Y-%m-%d %H:%M:%S',
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red,bg_white',
        }
    )

    # Остальная часть функции остается без изменений
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))

    logging.basicConfig(
        level=LOG_LEVEL,
        handlers=[stream_handler, file_handler]
    )

    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    # Также рекомендую заглушить sklearn, если он спамит
    logging.getLogger("sklearn").setLevel(logging.ERROR)
