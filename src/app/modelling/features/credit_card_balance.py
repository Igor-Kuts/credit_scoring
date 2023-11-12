import sys
sys.path.extend(['../../../config', '../../../app/utils'])

import pandas as pd
import numpy as np

from database_manager import PGSQL_Manager
from config import db_credentials

def main():
    
    # Путь для сохранения новых признаков
    PATH = str(input('Input path to save file: '))
    
    # Установим гостевое соединение с базой данных
    pgsql_guest = PGSQL_Manager()
    DB_ARGS = db_credentials(user_type="guest")
    engine = pgsql_guest.connect(**DB_ARGS)
    
    print('Загрузка данных из БД..')
    # Загрузим данные
    df = pgsql_guest.get_df_from_query('SELECT * FROM credit_card_balance')
    # DataFrame для новых признаков
    features = df[['sk_id_prev']]
    features = features[~features.sk_id_prev.duplicated()].sort_values(by='sk_id_prev')\
                                                          .reset_index(drop=True)

    print('Извлечение признаков..')
    # Посчитайте все возможные аггрегаты по картам
    aggregations = ['min', 'max', 'median', 'mean', 'sum', 'std']
    numeric_columns = df.drop(['sk_id_curr', 'months_balance', 'name_contract_status'], axis=1)
    tmp = numeric_columns.groupby('sk_id_prev').agg(aggregations)
    tmp.columns = tmp.columns.map('_'.join)
    features = features.merge(tmp, on='sk_id_prev', how='left')

    # Посчитайте как меняются аггрегаты. например отношение аггрегата за все время
    # к аггрегату за последние 3 месяца или к данных за последний месяц.
    numeric_columns = df.drop(['sk_id_curr','name_contract_status'], axis=1)
    features_alltime = features

    for month in [-1, -3]:
        period = numeric_columns[numeric_columns.months_balance.isin(np.arange(-1, month - 1, -1))] \
                                                               .drop(['months_balance'], axis = 1)
        tmp = period.groupby('sk_id_prev').agg(aggregations)
        tmp.columns = tmp.columns.map('_'.join)
        tmp = tmp.reset_index()
        tmp = features[['sk_id_prev']].merge(tmp, on='sk_id_prev', how='left')

        sk_id_prev = tmp[['sk_id_prev']]
        tmp
        tmp = features_alltime.drop('sk_id_prev', axis=1).divide(tmp.drop('sk_id_prev', axis=1))
        tmp.columns = [f'part_all_{i}_to_{-month}M' for i in tmp.columns]
        tmp['sk_id_prev'] = sk_id_prev
        features = features.merge(tmp, how='left', on='sk_id_prev')
        
    # Запись признаков в файл
    features.set_index('sk_id_prev', inplace=True)
    features.to_csv(path_or_buf=PATH, index_label=features.index.name)
    print('Файл успешно сохранен')
    
if __name__ == '__main__':
    main()
    