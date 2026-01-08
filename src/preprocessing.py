import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)


def create_dataframe_from_dataset(file_path: str) -> pd.DataFrame:
    """
    Загружает данные из CSV-файла и преобразует их в объект DataFrame.

    Функция выполняет чтение датасета с новостями, обеспечивая многоуровневую
    проверку ошибок: отсутствие файла, пустой файл или некорректная структура
    (ошибки парсинга). Путь к файлу ожидается по относительному адресу '../data/fake_news.csv'.

    Returns:
        pd.DataFrame: Объект с загруженными данными. В случае, если файл пуст,
            возвращается пустой DataFrame.

    Raises:
        FileNotFoundError: Если файл по указанному пути не найден (с выводом
            абсолютного пути в лог).
        pd.errors.ParserError: Если структура CSV-файла повреждена или используется
            неверный разделитель.
        Exception: В случае других непредвиденных системных ошибок при чтении.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info("Dataset loaded successfully")
        return df
    except FileNotFoundError:
        logger.error(f"File not found in path: {os.path.abspath(file_path)}")
        raise FileNotFoundError(f"Make sure the file {file_path} exists.")
    except pd.errors.EmptyDataError:
        logger.error("The file was found, but it is empty.")
        return pd.DataFrame()
    except pd.errors.ParserError:
        logger.error("Error reading CSV: check delimiters (commas, semicolons).")
        raise
        
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise


def create_labels(dataframe: pd.DataFrame) -> pd.Series:
    """
    Извлекает столбец с метками классов (целевую переменную) из датафрейма.

    Функция обеспечивает безопасный доступ к колонке 'label', проверяя наличие
    данных и корректность структуры переданного объекта. Используется как
    вспомогательный этап перед разделением данных на обучающую и тестовую выборки.

    Args:
        dataframe (pd.DataFrame): Исходный датафрейм, содержащий колонку 'label'.

    Returns:
        pd.Series: Серия данных с метками классов.

    Raises:
        TypeError: Если на вход подан объект, не являющийся датафреймом (ошибка атрибута).
        ValueError: Если в датафрейме отсутствует колонка с названием 'label'.
        Exception: В случае иных непредвиденных ошибок при работе с данными.
    """
    try:
        return dataframe.label
    except AttributeError:
        logger.error("Object has not Dataframe or None")
        raise TypeError('Expect Pandas Dataframe, get: ' + str(type(dataframe)))
    except KeyError:
        logger.error("DataFrame has not required column 'label'")
        raise ValueError("Column 'label' not found in data")
    except Exception as e:
        logger.error(f"An unexpected error when creating labels: {e}")
        raise
