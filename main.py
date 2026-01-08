import logging
from time import sleep
from config import setup_logging, CSV_PATH
from src import (
    split_dataset,
    create_tfidf_vectorizer,
    vectorize_data,
    train_passive_aggressive,
    evaluate_model,
    create_dataframe_from_dataset,
    train_sgd_classifier,
    plot_confusion_matrix,
    plot_class_distribution,
)

setup_logging()
MODEL = "PassiveAggressiveClassifier"
logger = logging.getLogger(__name__)


def run_pipeline(data_path: str):
    """
    Основной процесс: Загрузка -> Подготовка -> Обучение -> Оценка.
    """
    try:
        # 1. Загрузка данных
        logger.info(f"Загрузка данных из {data_path}")
        df = create_dataframe_from_dataset(data_path)
        sleep(0.5)

        # 2. Разделение датасета
        x_train, x_test, y_train, y_test = split_dataset(
            df, target_column='label', test_size=0.2
        )
        sleep(0.5)

        # 3. Подготовка признаков (TF-IDF)
        vectorizer = create_tfidf_vectorizer(max_df=0.7)
        tfidf_train, tfidf_test, _ = vectorize_data(x_train, x_test, vectorizer)
        sleep(0.5)

        # 4. Обучение модели
        # model = train_passive_aggressive(tfidf_train, y_train, max_iter=50)
        model = train_sgd_classifier(tfidf_train, y_train, max_iter=50)
        sleep(0.5)

        # 5. Оценка и вывод результата
        y_pred = model.predict(tfidf_test)
        accuracy = evaluate_model(model, tfidf_test, y_test)
        logger.info(f"Точность модели (accuracy): {round(accuracy * 100, 2)}")
        sleep(0.5)

        # 6. Построение графика матрицы ошибок
        plot_confusion_matrix(y_test, y_pred, filename="confusion_matrix_fake_news.png")

        # 7. Дополнительный график: распределение классов в данных
        plot_class_distribution(labels=y_train, filename="class_distribution_train.png")

        logger.info("Пайплайн успешно завершен.")
        return model, vectorizer

    except FileNotFoundError:
        logger.error(f"Файл {data_path} не найден.")
    except Exception as e:
        logger.error(f"Критическая ошибка в пайплайне: {e}", exc_info=True)


def main(data_path):
    try:
        result = run_pipeline(data_path)
        logger.info("End")
    except Exception as e:
        logger.critical(f"Critical failure in main: {e}")


if __name__ == "__main__":
    main(CSV_PATH)
