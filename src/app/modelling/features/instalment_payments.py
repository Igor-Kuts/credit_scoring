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
    df = pgsql_guest.get_df_from_query('SELECT * FROM installments_payments')
    features = df[['sk_id_curr']]
    features = features.drop_duplicates().sort_values(by='sk_id_curr') \
                                         .reset_index(drop=True)
    
    df[['days_instalment', 'days_entry_payment']] *= -1 

    # 1. Количество просрочек по платежам
    df['diff_instalments_payment'] = df.days_instalment - df.days_entry_payment
    tmp = df[df['diff_instalments_payment'] > 0]
    tmp = tmp.groupby('sk_id_curr')[['sk_id_prev']].agg(cnt_late_payments=pd.NamedAgg(column='sk_id_prev', aggfunc='count'))
    features = features.merge(tmp, on='sk_id_curr', how='left')

    # 2. Количество платежей раньше срока
    tmp = df[df['diff_instalments_payment'] < 0]
    tmp = tmp.groupby('sk_id_curr')[['sk_id_prev']].agg(early_payments=pd.NamedAgg(column='sk_id_prev', aggfunc='count'))
    features = features.merge(tmp, on='sk_id_curr', how='left')
    features.early_payments = features.early_payments.fillna(0)

    # 3. Средний процент платежа от необходимого
    df['pct_instalment_paid'] = df.amt_payment / df.amt_instalment
    tmp = df.groupby('sk_id_curr').agg(avg_pct_instalment_paid=pd.NamedAgg(column='pct_instalment_paid', aggfunc='mean'))
    tmp.avg_pct_instalment_paid = tmp.avg_pct_instalment_paid.round(3)
    features = features.merge(tmp, on='sk_id_curr', how='left')
    
    # Запись признаков в файл
    features = features.fillna(0)
    features.set_index('sk_id_curr', inplace=True)
    features.to_csv(path_or_buf=PATH, index_label=features.index.name)
    print('Файл успешно сохранен')
    
if __name__ == '__main__':
    main()
    