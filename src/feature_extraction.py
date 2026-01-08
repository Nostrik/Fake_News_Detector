import logging
import pandas as pd
from typing import Tuple
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def split_dataset(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42,
                  stratify: bool = True) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Разделяет исходный датафрейм на обучающую и тестовую выборки.

    Функция автоматически определяет признаки (X) и целевую переменную (y). Если в датафрейме
    больше двух колонок, в качестве признаков берутся все, кроме целевой. Если колонок всего две,
    предполагается работа с текстом. Используется стратификация для сохранения баланса классов.

    Args:
        df (pd.DataFrame): Исходный набор данных.
        target_column (str): Название колонки с целевой переменной (метками классов).
        test_size (float): Доля данных, которая пойдет в тестовую выборку (от 0.0 до 1.0).
        random_state (int): Число для фиксации случайности, обеспечивающее повторяемость разбиения.
        stratify (bool): Если True, данные будут разделены так, чтобы сохранить пропорции классов
            из исходного набора в обеих выборках (актуально для 2026 года при дисбалансе классов).

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
            Кортеж, содержащий (x_train, x_test, y_train, y_test).
    """
    logger.info(f"Split data: test_size={test_size}, random_state={random_state}")
    
    # X = df.drop(columns=[target_column]) if len(df.columns) > 1 else df['text']
    # y = df[target_column]

    # X = df
    # y = df[target_column]

    X = df['text']
    y = df[target_column]

    stratify_param = y if stratify else None

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=stratify_param
    )
    
    logger.info(f"Training size: {len(x_train)}, test: {len(x_test)}")
    return x_train, x_test, y_train, y_test


def create_tfidf_vectorizer(stop_words: str = 'english', max_df: float = 0.7,
                            ngram_range: tuple = (1, 1)) -> TfidfVectorizer:
    """
    Инициализирует и настраивает объект TfidfVectorizer для преобразования текста в векторы.

    Создает инструмент, который будет переводить текстовые данные в числовые признаки,
    используя статистику TF-IDF. Настройки подобраны для обеспечения качественной
    предобработки текста в задачах NLP актуальных для 2026 года.

    Args:
        stop_words (str): Язык стоп-слов (например, 'english'). Эти слова будут игнорироваться,
            так как они несут мало смысловой нагрузки.
        max_df (float): Максимальная доля документов (от 0.0 до 1.0). Слова, которые
            встречаются чаще чем в 70% документов, будут исключены как слишком общие.
        ngram_range (tuple): Диапазон n-грамм. (1, 1) означает только отдельные слова,
            (1, 2) добавит комбинации из двух слов подряд.

    Returns:
        TfidfVectorizer: Настроенный, но еще не обученный объект векторизатора с
            поддержкой обработки Unicode-символов.
    """
    logger.info(f"Init TfidfVectorizer: max_df={max_df}, stop_words={stop_words}")
    
    return TfidfVectorizer(
        stop_words=stop_words,
        max_df=max_df,
        ngram_range=ngram_range,
        # Добавляем strip_accents для лучшей очистки текста
        strip_accents='unicode',
        # Анализ на уровне слов (стандарт для 2025)
        analyzer='word'
    )


def vectorize_data(x_train: list | pd.Series, x_test: list | pd.Series,
                   vectorizer: TfidfVectorizer) -> Tuple[csr_matrix, csr_matrix, TfidfVectorizer]:
    """
    Преобразует текстовые данные в разреженные матрицы признаков с помощью TF-IDF векторизации.

    Процесс включает обучение векторизатора на тренировочных данных и последующую
    трансформацию как тренировочного, так и тестового наборов. Тестовый набор
    трансформируется строго на основе словаря, полученного из тренировочных данных.

    Args:
        x_train (list | pd.Series): Текстовые данные для обучения модели.
        x_test (list | pd.Series): Текстовые данные для тестирования модели.
        vectorizer (TfidfVectorizer): Необученный или готовый к настройке объект TfidfVectorizer.

    Returns:
        Tuple[csr_matrix, csr_matrix, TfidfVectorizer]:
            - tfidf_train: Матрица признаков для обучающей выборки.
            - tfidf_test: Матрица признаков для тестовой выборки.
            - vectorizer: Обученный векторизатор с зафиксированным словарем.

    Raises:
        Exception: Если в процессе векторизации возникла ошибка (например, из-за некорректных типов данных).
    """
    logger.info("Beginning of text vectorization...")
    try:
        # Рекомендуется убедиться, что нет NaN и данные — строки
        if isinstance(x_train, pd.Series):
            x_train = x_train.fillna('').astype(str)
        if isinstance(x_test, pd.Series):
            x_test = x_test.fillna('').astype(str)

        logger.debug(f"Длина x_train перед векторизацией: {len(x_train)}")
        logger.debug(f"Первые 3 элемента x_train: {x_train[:3]}")

        # Обучаем векторизатор на TRAIN
        tfidf_train = vectorizer.fit_transform(x_train)
        logger.debug(f"tfidf_train shape: {tfidf_train.shape}")

        # Только трансформируем TEST
        tfidf_test = vectorizer.transform(x_test)

        logger.info(
            f"Vectorization complete. "
            f"Dict size: {len(vectorizer.vocabulary_)} words."
        )

        return tfidf_train, tfidf_test, vectorizer

    except Exception as e:
        logger.error(f"Vectorization error: {e}")
        raise
