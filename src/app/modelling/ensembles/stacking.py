import joblib
import sys
import re
sys.path.extend(['../../../config', '../../../app/utils'])

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

from preprocessing_utils import one_hot_encoding

RANDOM_STATE = 123

def main():

    # Загрузка, предобработка и разделение данных
    df = pd.read_csv('../features/train_test_cleaned.csv', low_memory=False)
    
    df_train = df[df.target.notna()].copy()
    df_train.target = df_train.target.astype('int')
    
    categorical_columns = []
    for column in df_train.drop(['sk_id_curr', 'target'], axis=1).columns:
        if pd.api.types.is_object_dtype(df_train[column]) or pd.api.types.is_bool_dtype(df_train[column]):
            categorical_columns.append(column)
    
    # One-Hot кодирование категориальных признаков
    df_train = one_hot_encoding(df_train, categorical_columns, drop_policy='if_binary')
    
    # LGBTClassifier ругается на названия фичей после кодирования
    df_train.columns = [re.sub(r'[^\w\s]', '_', col) for col in df_train.columns]
    
    features = df_train.drop(columns=["target", "sk_id_curr"])
    target = df_train["target"]
    
    # Разделение данных на тренировочную и валидационную выборки со стратификацией
    X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, 
                                                      random_state=RANDOM_STATE, 
                                                      stratify=target)
    # Инициализация базовых и мета моделей
    model_logr = LogisticRegression(max_iter = 150, C=5, 
                                    solver = "lbfgs",
                                    class_weight = "balanced", 
    )
    model_rf = RandomForestClassifier(max_depth = 9,
                                      min_samples_leaf = 10, 
                                      class_weight = "balanced", 
                                      n_jobs = -1,                        
    )
    model_lgbm = LGBMClassifier(max_depth = 8, 
                                min_samples_leaf = 10,
                                class_weight = "balanced",
                                device_type = "gpu"
    )
    estimators = [("lgbm", model_lgbm), ("rf", model_rf)]
    
    # Инициализация, обучение и предсказания StackingClassifier
    # --- пробовал подобрать гиперпараметры через RandomizedSearchCV,
    # --- но так и не дождался пока он выполнится :)
    stacking = StackingClassifier(estimators=estimators, final_estimator=model_logr, cv=5)
    stacking.fit(X_train, Y_train)
    
    # Предсказания стэкинга
    y_predict = stacking.predict_proba(X_val)[:, 1]
    print(roc_auc_score(Y_val, y_predict))
    
    # Датафрейм с результатами стэкинга
    pred_df = pd.DataFrame(y_predict, index=X_val.index, columns=(['stacking']))
    
    # Сохраним результирующий датафрейм
    pred_df.to_csv('stacking_predictions.csv')

    # Сохраним модель для Kaggle
    joblib.dump(stacking, "stacking_classifier.pkl")
                
if __name__ == "__main__":
    main()
    