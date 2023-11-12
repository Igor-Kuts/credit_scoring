import joblib
import sys
import re
sys.path.extend(['../../../config', '../../../app/utils'])

import pandas as pd
import numpy as np

import optuna
import sklearn
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from preprocessing_utils import one_hot_encoding

RANDOM_STATE = 123

def main():
        
    # Загрузка данных
    df = pd.read_csv('../features/train_test_cleaned.csv', low_memory=False)
    df_train = df[df.target.notna()].copy()
    df_train['target'] = df_train['target'].astype('int')
    
    x_train = df_train.drop(['sk_id_curr', 'target'], axis=1)
    y_train = df_train.target
    
    # Загрузка моделей
    base_rf = joblib.load('tuned_randomforest.pkl')
    base_lgbm = joblib.load('tuned_lightgbm.pkl') 
    
    categorical_columns = []
    for column in df_train.drop(['sk_id_curr', 'target'], axis=1).columns:
        if pd.api.types.is_object_dtype(df_train[column]) or pd.api.types.is_bool_dtype(df_train[column]):
            categorical_columns.append(column)
    
    # ---------------- Функции ---------------- #
    def split_data(X, y, test_size=0.2, val_size=0.25, random_state=None):
        """
        Обёртка на train_test_split из SKLearn. 
        Делит данные на тренировочную, валидационную и тестовую выборки
        со стратификацией.
        """
        x_temp, X_test, y_temp, Y_test = train_test_split(X, y, 
                                                          test_size = test_size, 
                                                          stratify=y_train,
                                                          random_state=random_state)
        
        X_train, X_val, Y_train, Y_val = train_test_split(x_temp, y_temp, 
                                                          test_size=val_size, 
                                                          stratify=y_temp,
                                                          random_state=random_state)
        
        return X_train, X_val, X_test, Y_train, Y_val, Y_test
    
    def make_blending_dfs(features, target, models):
        """
        Формирует blending датасет для обучения мета модели
        
        Параметры:
        features (pd.DataFrame): Исходный датафрейм с признаками для обучения
        target (pd.DataFrame): Исходный датафрейм с целевой переменной
        models (list(base_models)): Список базовых моделей для обучения и предсказаний
    
        Возвращает:
        val_blending_df (pd.DataFrame): Датафрейм с предсказаниями базовых моделей
                                        на валидационной выборке
        test_blending_df (pd.DataFrame): Датафрейм с предсказаниями базовых моделей
                                         на тестовой выборке
        """        
        X = features.copy()
        Y = target.copy()
        
        # Закодируем категориальные признаки с помощью One-Hot кодирования
        # LGBTClassifier ругается на названия фичей после кодирования
        X = one_hot_encoding(X, categorical_columns, drop_policy='if_binary')
        X.columns = [re.sub(r'[^\w\s]', '_', col) for col in X.columns]
        
        # Разделим данные на тренировочную, валидационную и тестовую выборки
        X_train, X_val, X_test, \
        Y_train, Y_val, Y_test = split_data(X, Y, random_state=RANDOM_STATE)
        
        # Создадим дотафреймы для предсказаний базовых моделей
        val_blending_df = pd.DataFrame()
        test_blending_df = pd.DataFrame()

        # Пройдемся циклом по списку моделей
        for model in models:

            # Если модель RandomForestClassifier
            if type(model) == sklearn.ensemble._forest.RandomForestClassifier:

                # Обучим модель на тренировочной выборке и получим предсказания
                # на валидационной и тестовой выборке и сохраним в датафреймы
                model.fit(X_train, Y_train)
                val_blending_df['base_rf'] = model.predict_proba(X_val)[:,1]
                test_blending_df['base_rf'] = model.predict_proba(X_test)[:,1]
    
            # Если модель LightGBMClassifier
            elif type(model) == lightgbm.sklearn.LGBMClassifier:

                # Обучим модель на тренировочной выборке и получим предсказания
                # на валидационной и тестовой выборке и сохраним в датафреймы
                model.fit(X_train, Y_train, eval_set=(X_val, Y_val),
                          callbacks=[lightgbm.early_stopping(50)])
                val_blending_df['base_lgbm'] = model.predict_proba(X_val)[:,1]
                test_blending_df['base_lgbm'] = model.predict_proba(X_test)[:,1]

            # Добавим индексы и целевую в итоговые датафреймы
            val_blending_df.index, test_blending_df.index  = Y_val.index, Y_test.index
            val_blending_df['target'], test_blending_df['target'] = Y_val.values, Y_test.values
            
        return val_blending_df, test_blending_df
    
    def logistuna(trial):
        """
        Функция осуществляет подбор гиперпараметров мета-модели 
        с целью оптимизации заданного функционала
        """
        # Области поиска гипер-параметров
        param_logr = {
            "solver":          trial.suggest_categorical("solver", ["lbfgs"]),
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
        # Обучение и предсказания модели
        classifier_obj.fit(X_train, Y_train)
        Y_hat = classifier_obj.predict(X_val)
        rauc = roc_auc_score(Y_val, Y_hat)

        # Возвращение значения roc-auc для оптимизации
        return rauc
    # ----------------------------------------- #
    
    # получим предсказания базовых моделей на валидационной и тестовой выборках
    val_df, test_df = make_blending_dfs(x_train, y_train, [base_rf, base_lgbm])
    
    # Разделим датасет с предсказаниями базовых моделей
    # На тренировочную и валидационную выборки для обучения мета-модели
    X_train, X_val, Y_train, Y_val = train_test_split(val_df.drop('target',axis=1),
                                                      val_df.target, 
                                                      test_size = 0.25,
                                                      stratify=val_df.target,
                                                      random_state=RANDOM_STATE)
    # Создадим сэмплер и объект для оптимизации: study
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    # Оптимизируем
    study.optimize(logistuna, n_trials=100, n_jobs=5, gc_after_trial=True)  
    
    # Инициализируем и обучим LogisticRegression с подобранными гиперпараметрами
    logistic_regression = LogisticRegression(**study.best_trial.user_attrs['params'])
    logistic_regression.fit(X_train, Y_train)
    
    # Добавим в датафрейм с предсказаниями результаты полученные
    # С помощью мета-модели: Логистической Регрессии 
    test_df['blending'] = logistic_regression.predict_proba(test_df.drop('target', axis=1))[:,1]
    
    # Экстра: Bagging :)
    test_df['bagging'] = test_df[['base_rf', 'base_lgbm']].sum(axis=1) / 2
    
    # Сохраним результаты
    test_df.to_csv('blending_predictions.csv')

if __name__ == "__main__":
    main()
