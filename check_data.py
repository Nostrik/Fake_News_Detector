import logging
from src import create_dataframe_from_dataset
from config import setup_logging, CSV_PATH

# Настройка базового логгера для вывода в консоль
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_dataset_integrity(file_path: str, text_col: str = 'text', label_col: str = 'label'):
    """
    Загружает DataFrame и проверяет его целостность и согласованность колонок.
    """
    logger.info(f"Начинаем проверку файла: {file_path}")

    try:
        df = create_dataframe_from_dataset(file_path)
    except FileNotFoundError:
        logger.error("Проверка прервана: Файл не найден.")
        return

    if df.empty:
        logger.info("DataFrame пуст, проверка не требуется.")
        return

    total_rows = len(df)
    logger.info(f"Всего строк в DataFrame: {total_rows}")

    # Проверка наличия нужных колонок
    if text_col not in df.columns or label_col not in df.columns:
        logger.error(f"Колонки '{text_col}' или '{label_col}' не найдены. Доступные колонки: {list(df.columns)}")
        return

    # Проверка пропущенных значений (NaN)
    missing_text = df[text_col].isnull().sum()
    missing_labels = df[label_col].isnull().sum()

    logger.info(f"Пропущенных значений в колонке '{text_col}': {missing_text}")
    logger.info(f"Пропущенных значений в колонке '{label_col}': {missing_labels}")

    if missing_text > 0 or missing_labels > 0:
        logger.warning(f"Найдено {missing_text + missing_labels} строк с NaN. Это потенциальный источник ошибки.")

    # Проверка согласованности типов (если нужно)
    logger.info(f"Тип данных в '{text_col}': {df[text_col].dtype}")
    logger.info(f"Тип данных в '{label_col}': {df[label_col].dtype}")

    if missing_text == 0 and missing_labels == 0:
        logger.info("✅ **Файл CSV цельный. Ошибка не в исходных данных.**")
    else:
        logger.warning("❗️ **Файл CSV содержит пропуски. Требуется очистка данных.**")


if __name__ == "__main__":
    # Укажите здесь актуальный путь к вашему файлу
    check_dataset_integrity(CSV_PATH)
