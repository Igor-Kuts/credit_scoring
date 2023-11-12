import joblib
import sys
sys.path.extend(['../../../config', '../../../app/utils'])

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from lightgbm import LGBMClassifier
import optuna

RANDOM_STATE = 123

def main():
    def lgbtuna(trial):
        """ 
        Осуществляет подбор оптимальных гиперпараметров с целью максимизации ROC-AUC
        """
        # Настройка областей поиска гиперпараметров
        param_lgbm = {
            "objective":           "binary",
            "metric":              "auc",
            "is_unbalance":        True,
            
            "learning_rate":       trial.suggest_float("learning_rate", 0.001, 0.5, step=0.001), # def:0.1
            "max_depth":           trial.suggest_int("max_depth", 3, 10),
            "num_boost_round":     trial.suggest_int("num_iterations", 100, 1000),

            "num_leaves":          trial.suggest_int("num_leaves", 20, 62), # def:31
            "min_samples_leaf":    trial.suggest_int("min_samples_leaf", 10, 100), # def:20
            
            "feature_fraction":    trial.suggest_float("feature_fraction", 0.03, 1.0, step=0.01), # def:1
            "lambda_l1":           trial.suggest_int("lambda_l1", 0, 100), # def:0
            "lambda_l2":           trial.suggest_int("lambda_l2", 0, 100), # def:0
            
            "random_seed":         RANDOM_STATE, # имеет приоритет над остальными random_seed
            "device_type":         "gpu", # cpu, cuda
            "verbose":             -1, # <0:Fatal, =0:Error(Warning), = 1:Info, >1:Debug
            "early_stopping_round": 20,
            "feature_pre_filter": False
        }
        # Сохранение словаря гиперпараметров в информацию о триале
        trial.set_user_attr("params", param_lgbm)

        # Инициализация и обучение модели
        classifier_obj = lgb.train(param_lgbm,
                                   train_set=train_data, 
                                   valid_sets=[validation_data],
                                   callbacks=[lgb.early_stopping(param_lgbm["early_stopping_round"])]
                                   )
        # Предсказания на валидационной выборке
        rauc = roc_auc_score(Y_val, classifier_obj.predict(X_val))   
        return rauc
    
    
    # Загрузка данных
    df = pd.read_csv('../features/train_test_cleaned.csv', low_memory=False)
    df_train = df[df.target.notna()].copy()
    df_train['target'] = df_train['target'].astype('int')
    x_train = df_train.drop(['sk_id_curr', 'target'], axis=1)
    y_train = df_train.target
    
    numerical_columns = []
    categorical_columns = []
    for column in df_train.drop(['sk_id_curr', 'target'], axis=1).columns:
        if pd.api.types.is_object_dtype(df_train[column]) or pd.api.types.is_bool_dtype(df_train[column]):
            categorical_columns.append(column)
        else: numerical_columns.append(column)

    # Переведем в 'category' чтобы принял lightgbm
    x_train[categorical_columns]=x_train[categorical_columns].astype('category')
    
    # Разделение данных на тренировочную, валидационную и тестовую выборки
    x_temp, X_test, y_temp, Y_test = train_test_split(x_train, y_train, test_size = 0.2, stratify=y_train,
                                                      random_state=RANDOM_STATE)
    X_train, X_val, Y_train, Y_val = train_test_split(x_temp, y_temp, test_size = 0.25, stratify=y_temp,
                                                      random_state=RANDOM_STATE)

    # Z - Нормализация 
    scaler = StandardScaler()
    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_val[numerical_columns] = scaler.transform(X_val[numerical_columns])
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

    # Создадим LightGBM datasets для тренировки и валидации
    train_data = lgb.Dataset(X_train, Y_train, 
                             categorical_feature=categorical_columns,
                             free_raw_data=False)
    validation_data = train_data.create_valid(X_val, Y_val)
    
    # Создадим объект study
    study_lgbm = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    # Оптимизируем
    study_lgbm.optimize(lgbtuna, n_trials=100)

    # Инициализируем и сохраним модель с подобранными гиперпараметрами
    lightgbm = LGBMClassifier(**study_lgbm.best_trial.user_attrs['params'])
    joblib.dump(lightgbm, 'tuned_lightgbm.pkl')

if __name__ == "__main__":
    main()
