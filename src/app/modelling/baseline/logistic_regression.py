import joblib
import sys
sys.path.extend(['../../../config', '../../../app/utils'])

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import optuna

from preprocessing_utils import (
    one_hot_encoding, 
    get_column_types,
    load_and_split_data,
)
from optuna_utils import classifier_stratkf_loop

RANDOM_STATE = 123


def main():

    def logistuna(trial):
        """
        Функция осуществляет подбор гиперпараметров с целью оптимизации заданной функции
        """
        # Область поиска гипер-параметров
        param_logr = {
            "solver":          trial.suggest_categorical("solver", ["newton-cholesky", "lbfgs"]),
            "max_iter":        trial.suggest_int("max_iter", 100, 200),
            "C":               trial.suggest_float("C", 0.01, 10.01, step=0.1),
            "class_weight":    trial.suggest_categorical("class_weight", ["balanced"]),
            "random_state":    RANDOM_STATE
        }
        # Сохранение словаря гиперпараметров в информацию о триале
        trial.set_user_attr("params", param_logr)

        # Инициализация модели
        classifier_obj = LogisticRegression(**param_logr,
                                            n_jobs=-1)

        mean_rocauc = classifier_stratkf_loop(
            x_train, y_train,
            classifier_obj,
            metric=roc_auc_score,
            n_splits=5,
            shuffle=True,
            fold_normalize=True,
            numerical_columns=numerical_columns,
            random_state=RANDOM_STATE,
        )

        return mean_rocauc

    # Загрузка данных
    X_train, y_train = load_and_split_data(data_path='./train_test_cleaned.csv')

    # Определение типов признаков (cat, num)
    numerical_columns, categorical_columns = get_column_types(X_train)

    # One-Hot Encoding категориальных признаков
    x_train = one_hot_encoding(X_train, categorical_columns)

    # Создадим объект для оптимизации: study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))

    # Оптимизируем
    study.optimize(logistuna, n_trials=10, n_jobs=1, show_progress_bar=True, gc_after_trial=True)

    # Инициализируем LogisticRegression с подобранными гиперпараметрами
    logistic_regression = LogisticRegression(**study.best_trial.user_attrs['params'])

    # Transformers и preprocessor для численных и категориальных признаков
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='if_binary', handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns),
    ])

    # Pipeline: Логистическая Регрессия с подобранными гиперпараметрами и One-Hot
    logistic_regression = LogisticRegression(**study.best_trial.user_attrs['params'])
    logr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', logistic_regression)])

    # Обучение модели на всей тренировочной выборке
    logr_pipeline.fit(X_train, y_train)

    # Выгрузка модели
    joblib.dump(logr_pipeline, 'logistic_regression_pipe.pkl')

if __name__ == "__main__":
    main()
