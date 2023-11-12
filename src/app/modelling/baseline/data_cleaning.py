"""" 
Скрипт загружает датафрейм со значимыми признаками из базы данных и
проводит их предобработку, впоследствии сохраняя обработанные данные локально.
Скрипт не производит Ohe-Hot кодирование и Стандартизацию данных - данные операции
будут производиться в скриптах с обучением моделей. 
"""

import sys
sys.path.extend(['../../../config', '../../../app/utils'])

import pandas as pd
import numpy as np

from preprocessing_utils import find_outliers, drop_outliers
from database_manager import PGSQL_Manager
from config import db_credentials

PATH = "./train_test_cleaned.csv"

def main():
    
    # Установим гостевое соединение с базой данных
    pgsql_guest = PGSQL_Manager()
    DB_ARGS = db_credentials(user_type="guest")
    engine = pgsql_guest.connect(**DB_ARGS)
    
    # Загрузим данные
    df_orig = pgsql_guest.get_df_from_query('SELECT * FROM features_train_test')
    df_train = df_orig[df_orig.target.notna()].copy()
    df_test = df_orig[df_orig.target.isna()].copy()
    
    # Обработка выбросов
    numerical_columns = []
    categorical_columns = []
    for column in df_train.drop(['sk_id_curr', 'target'], axis=1).columns:
        if pd.api.types.is_object_dtype(df_train[column]) or pd.api.types.is_bool_dtype(df_train[column]):
            categorical_columns.append(column)
        else: numerical_columns.append(column)
    
    # Определим выбросы, не строго. Иначе потеряем много данных
    outliers_arr_train = find_outliers(df_train[numerical_columns], low=0.05, high=0.95)
    df_train[numerical_columns] = drop_outliers(df_train[numerical_columns], outliers_arr_train)
    
    # Удалим строки с выбросами
    outliers_arr_test = find_outliers(df_test[numerical_columns], low=0.05, high=0.95)
    df_test[numerical_columns] = drop_outliers(df_test[numerical_columns], outliers_arr_test)
    
    # Обработка пропущенных значений
    # "name_type_suite"
    for df in [df_train, df_test]:  
        df['name_type_suite'].fillna('<missing>', inplace=True)
    
    # "amt_income_total"
    # Ограничим сверху доход по 500к
    for df in [df_train, df_test]:
        df.loc[df.amt_income_total > 5e5, 'amt_income_total'] = 5e5
    
    # "cnt_children"
    # Ограничим сверху кол-во детей по 10
    for df in [df_train, df_test]:
        df.loc[df.cnt_children > 10, 'cnt_children'] = 10
    
    # Заполним пропуски нулём
    for df in [df_train, df_test.drop('target', axis=1)]:
        df.fillna(0, inplace=True)
    
    # Замена inf на 0
    for df in [df_train, df_test]:
        df.loc[np.isinf(df.avg_pct_instalment_paid), 'avg_pct_instalment_paid'] = 0
    
    # Сохранение результата
    data = pd.concat([df_train, df_test])
    data.to_csv(PATH, index=False)

if __name__ == "__main__":
    main()