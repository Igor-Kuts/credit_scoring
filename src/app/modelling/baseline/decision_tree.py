import joblib
import sys
sys.path.extend(['../../../config', '../../../app/utils'])

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
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

    def treetuna(trial):
        """
        Функция осуществляет подбор гиперпараметров с целью оптимизации заданной функции
        """
        # Область поиска гипер-параметров
        param_dt = {
            "criterion":            trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "max_depth":            trial.suggest_int("max_depth", 5, 13),
            "min_samples_split":    trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":     trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features":         trial.suggest_float("max_features", 0.05, 1.0, step=0.05),
             
            "class_weight":    trial.suggest_categorical("class_weight", ["balanced"]),
            "random_state":    RANDOM_STATE
        }
        # Сохранение словаря гиперпараметров в информацию о триале
        trial.set_user_attr("params", param_dt)

        # Инициализация модели
        classifier_obj = DecisionTreeClassifier(**param_dt)

        mean_rocauc = classifier_stratkf_loop(
            x_train, y_train,
            classifier_obj,
            metric=roc_auc_score,
            n_splits=5,
            shuffle=True,
            fold_normalize=False,
            random_state = RANDOM_STATE,
        )

        return mean_rocauc

    # Загрузка данных
    X_train, y_train = load_and_split_data(data_path='./train_test_cleaned.csv')

    # Определение типов признаков (cat, num)
    numerical_columns, categorical_columns = get_column_types(X_train)

    # One-Hot Encoding категориальных признаков для Optuna
    x_train = one_hot_encoding(X_train, categorical_columns)

    # Создадим объект для оптимизации: study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))

    # Оптимизируем
    study.optimize(treetuna, n_trials=10, n_jobs=10, show_progress_bar=True, gc_after_trial=True)

    # Трансформер для категориальных признаков
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='if_binary', handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[('cat', categorical_transformer, categorical_columns)])

    # Pipeline: Дерево Решений с подобранными гиперпараметрами и One-Hot
    decision_tree = DecisionTreeClassifier(**study.best_trial.user_attrs['params'])
    dt_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', decision_tree)])

    # Обучение модели на всей тренировочной выборке
    dt_pipeline.fit(X_train, y_train)

    # Выгрузка модели
    joblib.dump(dt_pipeline, 'decision_tree_pipe.pkl')

if __name__ == "__main__":
    main()
