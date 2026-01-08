import logging
from typing import Union
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)


def train_passive_aggressive(x_train: csr_matrix, y_train: Union[pd.Series, list], max_iter: int = 50,
                             random_state: int = 7, c: float = 1.0) -> PassiveAggressiveClassifier:
    """
    Выполняет обучение классификатора Passive Aggressive (PAC).

    Алгоритм PAC эффективен для крупномасштабного обучения и задач классификации текстов.
    Функция настраивает модель с применением стратегии ранней остановки (early stopping)
    для предотвращения переобучения, если качество на валидационной подвыборке
    перестает расти. Данные для валидации (10%) выделяются автоматически.

    Args:
        x_train (csr_matrix): Разреженная матрица признаков (результат векторизации).
        y_train (Union[pd.Series, list]): Целевая переменная (метки классов).
        max_iter (int): Максимальное количество эпох обучения (проходов по данным).
        random_state (int): Параметр для обеспечения воспроизводимости результатов.
        c (float): Параметр агрессивности. Определяет степень влияния каждой ошибки
            на изменение весов модели. Большее значение делает обучение более "агрессивным".

    Returns:
        PassiveAggressiveClassifier: Обученная модель классификатора.

    Raises:
        Exception: Если в процессе обучения возникла ошибка (например, несовпадение размерностей
            или некорректные типы данных).
    """
    logger.info(f"Start training PAC: max_iter={max_iter}, C={c}")

    model = PassiveAggressiveClassifier(
        max_iter=max_iter,  # Максимальное число итераций
        random_state=random_state,  # Контроль случайности
        C=c,  # Параметр агрессивности
        early_stopping=True,  # Ранняя остановка
        validation_fraction=0.1,  # Доля валидационной выборки
        n_iter_no_change=5  # Количество итераций без улучшений
    )

    try:
        model.fit(x_train, y_train)
        logger.info(f"The model is trained. Iterations completed: {model.n_iter_}")
        return model
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise


def train_sgd_classifier(x_train: csr_matrix, y_train: Union[pd.Series, list], max_iter: int = 50,
                         random_state: int = 7, eta0: float = 1.0) -> SGDClassifier:
    """
    Выполняет обучение линейного классификатора SGDClassifier
    в режиме Passive-Aggressive (PA-I).

    Данная реализация является современной заменой
    PassiveAggressiveClassifier, который был объявлен устаревшим
    в sklearn >= 1.8.

    Используется hinge-loss (SVM), стратегия PA-I и ранняя остановка
    для предотвращения переобучения. 10% обучающих данных
    автоматически выделяются под валидацию.

    Args:
        x_train (csr_matrix): Разреженная матрица признаков (TF-IDF).
        y_train (Union[pd.Series, list]): Целевая переменная (метки классов).
        max_iter (int): Максимальное количество эпох обучения.
        random_state (int): Фиксация случайности.
        eta0 (float): Параметр агрессивности (эквивалент C в PAC).

    Returns:
        SGDClassifier: Обученная модель классификатора.

    Raises:
        Exception: Если возникла ошибка в процессе обучения.
    """
    logger.info(
        f"Start training SGDClassifier (PA-I): "
        f"max_iter={max_iter}, eta0={eta0}"
    )

    model = SGDClassifier(
        loss="hinge",
        penalty=None,
        learning_rate="pa1",
        eta0=eta0,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5
    )

    try:
        model.fit(x_train, y_train)
        logger.info(
            f"The model is trained. Iterations completed: {model.n_iter_}"
        )
        return model

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise


def evaluate_model(model: any, x_test: any, y_test: pd.Series | list) -> float:
    """
    Проводит комплексную оценку обученной модели на тестовых данных.

    Функция выполняет предсказание на основе предоставленных признаков, вычисляет
    общую точность (Accuracy) и формирует детальный отчет по метрикам для каждого
    класса (Precision, Recall, F1-score). Результаты выводятся в лог и консоль
    в структурированном виде, актуальном для анализа классификации в 2026 году.

    Args:
        model (any): Обученная модель (например, PassiveAggressiveClassifier),
            поддерживающая метод .predict().
        x_test (any): Тестовая матрица признаков (обычно csr_matrix).
        y_test (pd.Series | list): Истинные метки классов для проверки качества.

    Returns:
        float: Значение метрики Accuracy (общая точность) в диапазоне от 0.0 до 1.0.

    Output:
        В консоль выводится визуально оформленный блок с процентом точности
        и таблица classification_report с метриками для каждого отдельного класса.
    """
    logger.info("Running model evaluation...")

    # 1. Предсказание
    y_pred = model.predict(x_test)

    # 2. Расчет метрик
    accuracy = accuracy_score(y_test, y_pred)

    # 3. Формирование детального отчета
    # classification_report дает точность, полноту и F1 для каждого класса
    report = classification_report(y_test, y_pred)

    # Логирование результатов
    logger.info(f"Accuracy: {accuracy:.4f}")

    # Вывод в консоль в красивом формате (традиционно оставляем для читаемости)
    print("\n" + "="*30)
    print(f"TOTAL ACCURACY: {accuracy * 100:.2f}%")
    print("="*30)
    print("Detailed report by class:")
    print(report)
    print("="*30)

    return accuracy
