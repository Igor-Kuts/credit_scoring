import joblib
import sys
sys.path.extend(['../../../config', '../../../app/utils'])

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import optuna

from preprocessing_utils import one_hot_encoding

RANDOM_STATE = 123

def main():
    def forestuna(trial):
        """ 
        Осуществляет подбор оптимальных гиперпараметров с целью максимизации ROC-AUC
        """
        # Область поиска гипер-параметров
        param_rf = {
            "n_estimators":         trial.suggest_int("n_estimators", 100, 200, step=5),
            "criterion":            trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "max_depth":            trial.suggest_int("max_depth", 3, 10),
            "min_samples_split":    trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":     trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features":         trial.suggest_float("max_features", 0.03, 0.2, step=0.01),
            "class_weight":         trial.suggest_categorical("class_weight", ["balanced"]),
            "random_state":         RANDOM_STATE
        }
        
        # Сохранение словаря гиперпараметров в информацию о триале
        trial.set_user_attr("params", param_rf)
        
        # Инициализация и обучение модели
        classifier_obj = RandomForestClassifier(**param_rf, n_jobs=-1)
        classifier_obj.fit(X_train, Y_train)
        Y_hat = classifier_obj.predict(X_val)
        rauc = roc_auc_score(Y_val, Y_hat)
        
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
            
    # One-Hot Encoding
    x_train = one_hot_encoding(x_train, categorical_columns, drop_policy='if_binary')
    
    # Разделение данных на тренировочную, валидационную и тестовую выборки
    x_temp, X_test, y_temp, Y_test = train_test_split(x_train, y_train, test_size = 0.2, stratify=y_train)
    X_train, X_val, Y_train, Y_val = train_test_split(x_temp, y_temp, test_size = 0.25, stratify=y_temp)
    
    # Z - Нормализация 
    scaler = StandardScaler()
    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_val[numerical_columns] = scaler.transform(X_val[numerical_columns])
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])
    
    # Создадим объект study
    study_rf = optuna.create_study(direction="maximize",
                                   sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    # Оптимизируем
    study_rf.optimize(forestuna, n_trials=100)

    # Инициализируем и сохраним модель c подобранными гиперпараметрами
    random_forest = RandomForestClassifier(**study_rf.best_trial.user_attrs['params'])
    joblib.dump(random_forest, 'tuned_randomforest.pkl')

if __name__ == "__main__":
    main()
