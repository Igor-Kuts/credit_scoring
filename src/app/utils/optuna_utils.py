from typing import List

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

def classifier_stratkf_loop(
    X, y,
    classifier_obj,
    metric = accuracy_score,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = None,
    fold_normalize: bool = False,
    numerical_columns: List = [],
    ):
    """
    Оценка классификатора с использованием стратифицированной кросс-валидации.

    Аргументы:
        X (DataFrame): Признаки для всего набора данных.
        y (Series): Целевые метки для всего набора данных.
        classifier_obj (объект): Модель классификатора, которую необходимо оценить.
        metric: Метрика качества для оценки модели.
        n_splits (int): Количество разбиений в стратифицированной кросс-валидации.
        shuffle (bool): Флаг для перемешивания данных перед разбиением.
        random_state (int): Зерно для генерации случайных чисел для обеспечения воспроизводимости.
        fold_normalize (bool): Нормализовать ли числовые столбцы внутри каждого разбиения.
        numerical_columns (список): Список имен числовых столбцов для нормализации,
            если `fold_normalize` установлен в True.

    Возвращает:
        float: Среднее значение выбранной метрики по всем разбиениям.
    """

    # Подбор гипер-параметров модели на кросс-валидации
    score_list = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        Y_train, Y_val = y.iloc[train_idx], y.iloc[val_idx]

        if fold_normalize:
            # ----------------Нормализация-------------------- #
            scaler = StandardScaler()
            X_train.loc[:, numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
            X_val.loc[:, numerical_columns] = scaler.transform(X_val[numerical_columns])
            # ------------------------------------------------ #
        
        # Обучение модели
        classifier_obj.fit(X_train, Y_train)
        Y_hat = classifier_obj.predict(X_val)
        score = metric(Y_val, Y_hat)
        score_list.append(score)

    # Среднее значение метрики по фолдам
    return np.mean(score_list)
