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
    df = pgsql_guest.get_df_from_query('SELECT * FROM previous_application')
    features = df[['sk_id_curr']]
    features = features.drop_duplicates().sort_values(by='sk_id_curr') \
                                         .reset_index(drop=True)
    
    print('Извлечение признаков..')
    # 1. Количество прошлых кредитов по типам
    tmp = pd.pivot_table(
        data = df,
        index = 'sk_id_curr', 
        columns = 'name_contract_type', 
        values = 'sk_id_prev', 
        aggfunc = 'count',
        fill_value = 0)
    tmp.columns = [f'num_prev_{i.lower()}' for i in tmp.columns]
    tmp = tmp.reset_index()
    features = features.merge(tmp, on='sk_id_curr', how='left') 

    # 2. Средний процент одобренной суммы кредита
    df['pct_approve'] = df.amt_credit / df.amt_application
    tmp = df.groupby('sk_id_curr').agg(avg_pct_approved=pd.NamedAgg(column='pct_approve', aggfunc='mean')).reset_index()
    tmp.avg_pct_approved = tmp.avg_pct_approved.round(2)
    tmp.loc[tmp.avg_pct_approved == np.inf, 'avg_pct_approved'] = 0
    features = features.merge(tmp, on='sk_id_curr', how='left') 

    # 3. Средняя частота обращений (меньше = чаще)
    tmp = df.groupby('sk_id_curr').agg(avg_inq_freq=pd.NamedAgg(column='days_decision', aggfunc='mean')).reset_index()
    tmp.avg_inq_freq = -tmp.avg_inq_freq.astype('int')
    features = features.merge(tmp, on='sk_id_curr', how='left') 

    # 4. Количество заявок по времени суток
    day = {
        "early": [5,6,7,8],
        "daytime": [9,10,11,12,13,14,15,16,17,18],
        "afterwork": [19,20,21,22],
        "late": [23,0,1,2,3,4]
    }
    for key, period in day.items():
        tmp = df[df.hour_appr_process_start.isin(period)]
        tmp = tmp.groupby('sk_id_curr').agg(period=pd.NamedAgg(column='sk_id_prev', aggfunc='count'))
        tmp.columns = [f'cnt_{key}_applications' for i in range(len(tmp.columns))]
        tmp.reset_index(inplace=True)
        features = features.merge(tmp, on='sk_id_curr', how='left')

    # Запись признаков в файл
    features = features.fillna(0)
    features.set_index('sk_id_curr', inplace=True)
    features.to_csv(path_or_buf=PATH, index_label=features.index.name)
    print('Файл успешно сохранен')
    
if __name__ == '__main__':
    main()
    