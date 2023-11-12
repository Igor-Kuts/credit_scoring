"""
Скрипт:
- Загружает исходный application_train_test.csv и дополнительные сгенерированные признаки.
- Отбирает среди них статистически значимые признаки с помощью статистических тестов.
- Формирует и загружает в PostgreSQL итоговый датафрейм.
"""
import sys
sys.path.extend(['../../../config', '../../../app/utils'])

import pandas as pd
import numpy as np
from tqdm import tqdm

from database_manager import PGSQL_Manager
from config import db_credentials
from stat_utils import (anderson_darling_test,
                        students_ttest,
                        mannwhitneyu_test,
                        bootstrap,
                        chi_squared_test)

def main():
    
    # Загрузка сгенерированных признаков
    df_temp1 = pd.read_csv(str(input("Input PATH to ftrs_application_train_test.csv: ")))
    df_temp2 = pd.read_csv(str(input("Input PATH to ftrs_installment_payments.csv: ")))
    df_temp3 = pd.read_csv(str(input("Input PATH to ftrs_bureau_bureaubalance.csv: ")))
    df_temp4 = pd.read_csv(str(input("Input PATH to ftrs_previous_application.csv: ")))
    df_temp5 = pd.read_csv(str(input("Input PATH to ftrs_credit_card_balance.csv: ")))
    
    # Установим административное соединение с базой данных
    pgsql_admin = PGSQL_Manager()
    DB_ARGS = db_credentials(user_type="admin")
    engine = pgsql_admin.connect(**DB_ARGS)
    
    # Загрузим исходные данные
    df_main = pgsql_admin.get_df_from_query('SELECT * FROM application_train_test')
    
    # Объединим таблицы
    for temp_df in [df_temp1, df_temp2, df_temp3, df_temp4, df_temp5]:#, df_temp2, df_temp3, df_temp4, df_temp5]:
        df_main = df_main.merge(temp_df, on='sk_id_curr', how='left')
    
    numerical_columns = []
    categorical_columns = []

    df = df_main
    # Оставим только признаки с количеством пропусков < 50%
    for column in df.drop(['sk_id_curr', 'target', 'ext_source_1',
                                                   'ext_source_2',
                                                   'ext_source_3'], axis=1).columns:
        if df[column].notna().sum() / df.shape[0] >= 0.5:
            if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_bool_dtype(df[column]):
                categorical_columns.append(column)
            else: numerical_columns.append(column)
    
    # Отберём только важные признаки
    significant_features = {}
    print('Обработка численных признаков..')
    for feature in tqdm(numerical_columns):
        tmp = df[[feature, 'target']].fillna(df[feature].mean())
    
        significant_features[feature] = {}
    
        if anderson_darling_test(tmp, feature, 'target'):
            significant_features[feature]['ttest_norm'] = students_ttest(tmp, feature)
            significant_features[feature]['bootstrap_norm'] = bootstrap(tmp, feature, 'target')
    
        else:
            significant_features[feature]['mann_whitney_notnorm'] = mannwhitneyu_test(tmp, feature)
            significant_features[feature]['bootstrap_notnorm'] = bootstrap(tmp, feature, 'target')
    
    print('Обработка категориальных признаков..')
    for feature in tqdm(categorical_columns):
        tmp = df[[feature, 'target']]
    
        significant_features[feature] = {}
        significant_features[feature]['chisquared_test'] = chi_squared_test(tmp, feature, 'target')
    
    # Если оба теста == 0,- тогда 0. Иначе, - 1.
    significant_features = {i: int(sum(significant_features[i].values()) / \
                                   len(significant_features[i].values()) >= 0.5) \
                            for i in significant_features}
    
    # Отберем важные признаки
    significant_features = [i for i, j in significant_features.items()\
                            if significant_features[i] == 1]
    df = df[['sk_id_curr', 'target', 'ext_source_1',
                                     'ext_source_2',
                                     'ext_source_3'] + significant_features]
    
    # Запишем датафрейм в базу данных
    print('Загрузка данных в PosgreSQL')
    df.to_sql('features_train_test', engine, if_exists='replace', index=False)
    # Дадим права пользователю гость
    pgsql_admin.send_sql_query("""GRANT SELECT ON ALL TABLES IN SCHEMA public TO guest;""")

if __name__=="__main__":
    main()
