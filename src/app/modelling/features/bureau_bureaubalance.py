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
    df = pgsql_guest.get_df_from_query('SELECT * FROM bureau')
    # DataFrame для новых признаков
    features = df[['sk_id_curr']]
    features = features.drop_duplicates().sort_values(by='sk_id_curr').reset_index(drop=True)
    
    print('Извлечение признаков..')
    # 1. Максимальная сумма просрочки
    tmp = df.groupby('sk_id_curr').agg(
        max_overdue_amt=pd.NamedAgg(column='amt_credit_sum_overdue', aggfunc='max')).reset_index()
    features = features.merge(tmp, on='sk_id_curr', how='left')
    
    # 2. Минимальная сумма просрочки
    # Напишем пользовательскую функцию
    def custom_min(group):
        if (group > 0).any(): return group[group != 0].min()
        else: return 0
    
    tmp = df.groupby('sk_id_curr').agg(
        min_overdue_amt=pd.NamedAgg(column='amt_credit_sum_overdue', aggfunc=custom_min)).reset_index()
    features = features.merge(tmp, on='sk_id_curr', how='left')
    
    # 3. Доля просрочки активного займа
    tmp = df.loc[df.credit_active == 'Active', ['sk_id_curr', 'amt_credit_sum_overdue', 'amt_credit_sum']]
    tmp = tmp.groupby('sk_id_curr').sum().reset_index()
    tmp['actv_overdue_rate'] = tmp.amt_credit_sum_overdue / tmp.amt_credit_sum
    features = features.merge(tmp[['sk_id_curr','actv_overdue_rate']], on='sk_id_curr', how='left')
    features.loc[features['actv_overdue_rate'] == np.inf, 'actv_overdue_rate'] = 0
    
    # 4. Количество кредитов определённого типа
    tmp = pd.pivot_table(
        data = df,
        index='sk_id_curr',
        columns='credit_type',
        values='sk_id_bureau',
        aggfunc=pd.Series.count,
        fill_value=0
    )\
    .reset_index()
    
    features = features.merge(tmp, on='sk_id_curr', how='left')
    
    # 5. Кол-во просрочек кредитов определенного типа
    tmp = pd.pivot_table(
        data = df.loc[(df['amt_credit_max_overdue'] > 0) | (df['amt_credit_sum_overdue'] > 0)], 
        index='sk_id_curr', 
        columns='credit_type', 
        values='sk_id_bureau', 
        aggfunc=pd.Series.count,        
        fill_value=0
    )
    
    tmp.columns = ['cnt_overdue_' + i for i in tmp.columns]
    features = features.merge(tmp, on='sk_id_curr', how='left')
    features[tmp.columns] = features[tmp.columns].fillna(0)
    
    # 6. Кол-во закрытых кредитов определенного типа
    tmp = pd.pivot_table(
        data = df[df.credit_active=='Closed'],
        index = 'sk_id_curr',
        columns = 'credit_type',
        values = 'sk_id_bureau',
        aggfunc = pd.Series.count,
        fill_value = 0
    )
    tmp.columns = ['cnt_closed_' + i for i in tmp.columns]
    features = features.merge(tmp, on='sk_id_curr', how='left')
    features[tmp.columns] = features[tmp.columns].fillna(0)

    # Загрузим данные из bureau_balance
    df_bur_bal = pgsql_guest.get_df_from_query('SELECT * FROM bureau_balance')
    # Сохраним ключи из bureau в df_bur_bal
    df_bur_bal = df_bur_bal.merge(df[['sk_id_curr', 'sk_id_bureau']], on='sk_id_bureau', how = 'left')
    
    # 1. Количество открытых кредитов
    tmp = df_bur_bal[~df_bur_bal.status.isin(['X', 'C', '5'])]
    tmp = tmp.groupby(['sk_id_curr','sk_id_bureau']).status.count().reset_index()
    tmp = tmp.groupby('sk_id_curr').agg(
        cnt_open_credits=pd.NamedAgg(column='status', aggfunc='sum')).reset_index()
    features = features.merge(tmp, on='sk_id_curr', how='left')
    
    # 2. Количество закрытых кредитов
    tmp = df_bur_bal[df_bur_bal.status.isin(['C'])]
    tmp = tmp.groupby(['sk_id_curr','sk_id_bureau']).status.count().reset_index()
    tmp = tmp.groupby('sk_id_curr').agg(
        cnt_closed_credits=pd.NamedAgg(column='status', aggfunc='sum')).reset_index()
    features = features.merge(tmp, on='sk_id_curr', how='left')
    
    # 3. Кол-во просроченных кредитов по разным дням просрочки (смотреть дни по колонке STATUS)
    tmp = df_bur_bal[~df_bur_bal.status.isin(['C', 'X', '0'])].groupby(['sk_id_curr', 'sk_id_bureau', 'status'])\
                                                              .count()\
                                                              .unstack()\
                                                              .reset_index()
    tmp.columns = tmp.columns.droplevel(level=0)
    tmp.columns = ['sk_id_curr', 'sk_id_bureau', '1', '2', '3', '4', '5']
    tmp = tmp.groupby('sk_id_curr')[['1', '2', '3', '4', '5']].sum()
    tmp.columns = ['cnt_overdue_'+str(i)+'_dpd' for i in tmp.columns]
    features = features.merge(tmp, on='sk_id_curr', how='left')
    features[tmp.columns] = features[tmp.columns].fillna(0)
    
    # 4. Кол-во кредитов
    tmp = df_bur_bal.groupby('sk_id_curr')[['sk_id_bureau']].count()
    tmp.columns = ['cnt_credits']
    features = features.merge(tmp, on='sk_id_curr', how='left')
    
    # 5. Доля закрытых кредитов
    features['rate_closed_credits'] = features['cnt_closed_credits'] / features['cnt_credits']
    
    # 6. Доля открытых кредитов
    features['rate_open_credits'] = features['cnt_open_credits'] / features['cnt_credits']
    
    # 7. Доля просроченных кредитов по разным дням просрочки(смотреть дни по колонке STATUS)
    cols_dpd = ['cnt_overdue_' + str(i) + '_dpd' for i in np.arange(1,6)]
    cols_rate = ['rate_' + str(i) + '_dpd' for i in np.arange(1,6)]
    features[cols_rate] = features[cols_dpd].divide(features[['cnt_credits']].values)
    
    # 8. Интервал между последним закрытым кредитом и текущей заявкой
    tmp = df_bur_bal.loc[df_bur_bal['status'] == 'C'].groupby(['sk_id_curr','sk_id_bureau'])[['months_balance']].max().reset_index()
    tmp.rename(columns={'months_balance': 'diff_last_closed_credit'}, inplace=True)
    tmp = tmp.groupby('sk_id_curr')[['diff_last_closed_credit']].max().reset_index()
    tmp['diff_last_closed_credit'] = tmp['diff_last_closed_credit'] * 30 # переведём в дни
    features = features.merge(tmp, on='sk_id_curr', how='left')
    
    # 9. Интервал между взятием последнего активного займа и текущей заявкой
    actv_bur_idx = df[df.credit_active == 'Active'].sk_id_bureau
    tmp = df_bur_bal[df_bur_bal.sk_id_bureau.isin(actv_bur_idx)].groupby('sk_id_bureau', as_index=False)[['months_balance']].min()
    tmp = df[['sk_id_curr', 'sk_id_bureau', 'days_credit']].merge(tmp, on='sk_id_bureau', how='left')
    tmp = tmp.groupby('sk_id_curr')[['days_credit','months_balance']].max()
    tmp = tmp.rename(columns={'days_credit':'days_since_last_actv_credit',
                              'months_balance':'months_since_last_actv_credit'})
    features = features.merge(tmp, on='sk_id_curr', how='left')
    
    # Запись признаков в файл
    features.set_index('sk_id_curr', inplace=True)
    features.to_csv(path_or_buf=PATH, index_label=features.index.name)
    print('Файл успешно сохранен')

if __name__ == '__main__':
    main()
