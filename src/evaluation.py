from typing import List
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import confusion_matrix
import pandas as pd

logger = logging.getLogger(__name__)


def plot_confusion_matrix(y_true: pd.Series | list, y_pred: pd.Series | list, labels: List[str] = ["FAKE", "REAL"],
                          title: str = "Матрица ошибок (Confusion Matrix)", figsize: tuple = (8, 6),
                          cmap: str = "Blues", save: bool = True, filename: str = "confusion_matrix.png") -> None:
    """
    Строит матрицу ошибок и автоматически сохраняет график
    в каталог results/figures/.

    Если каталог не существует, он будет создан автоматически.

    Args:
        y_true (pd.Series | list): Истинные метки классов.
        y_pred (pd.Series | list): Предсказанные метки классов.
        labels (List[str]): Порядок меток классов.
        title (str): Заголовок графика.
        figsize (tuple): Размер изображения.
        cmap (str): Цветовая схема.
        save (bool): Сохранять ли изображение.
        filename (str): Имя файла изображения.

    Returns:
        None
    """
    logger.info("Building confusion matrix plot...")

    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            xticklabels=labels,
            yticklabels=labels
        )

        plt.title(title)
        plt.ylabel("Истинные значения")
        plt.xlabel("Предсказанные значения")
        plt.tight_layout()

        if save:
            # src/*.py → project root → results/figures
            project_root = Path(__file__).resolve().parents[1]
            figures_dir = project_root / "results" / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)

            save_path = figures_dir / filename
            plt.savefig(save_path, dpi=300)
            logger.info(f"Confusion matrix saved to: {save_path}")

        plt.show()

    except Exception as e:
        logger.error(f"Error while plotting confusion matrix: {e}")
        raise


def plot_class_distribution(labels: pd.Series | list, title: str = "Распределение типов новостей в датасете",
                            xlabel: str = "Тип новости", ylabel: str = "Количество", figsize: tuple = (6, 4),
                            palette: str = "viridis", save: bool = True,
                            filename: str = "class_distribution.png") -> None:
    """
    Строит график распределения классов в датасете
    и сохраняет его в results/figures/ без deprecated-параметров seaborn.
    """
    logger.info("Building class distribution plot...")

    try:
        plt.figure(figsize=figsize)

        sns.countplot(
            x=labels,
            hue=labels,
            palette=palette,
            legend=False
        )

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()

        if save:
            project_root = Path(__file__).resolve().parents[1]
            figures_dir = project_root / "results" / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)

            save_path = figures_dir / filename
            plt.savefig(save_path, dpi=300)
            logger.info(f"Class distribution plot saved to: {save_path}")

        plt.show()

    except Exception as e:
        logger.error(f"Error while plotting class distribution: {e}")
        raise
