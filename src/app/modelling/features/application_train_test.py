import sys
sys.path.extend(['../../../config', '../../../app/utils'])

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

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
    df = pgsql_guest.get_df_from_query('SELECT * FROM application_train_test')
    # DataFrame для новых признаков
    features = df[['sk_id_curr']].sort_values(by='sk_id_curr').reset_index(drop=True)
    
    print('Извлечение признаков..')
    # 1. Количество документов
    flag_document_columns = list(df.columns[df.columns.str.contains('flag_document_')])
    df['num_documents'] = df[flag_document_columns].sum(axis=1) + 1 # +1 для учёта ID
    features = features.merge(right=df[['sk_id_curr','num_documents']], how='left', left_on='sk_id_curr', right_on='sk_id_curr')

    # 2. Есть ли полная информация о доме
    building_info_columns = list(df.columns[df.columns.str.contains('_avg|_mode|_medi')])             
    df['building_info'] = (df[building_info_columns].isna().sum(axis=1) < 30).astype(int)
    features = features.merge(right=df[['sk_id_curr','building_info']], how='left', left_on='sk_id_curr', right_on='sk_id_curr')

    # 3. Количество полных лет
    df['age_years'] = np.floor(df.days_birth / 365.25).astype('int16')
    features = features.merge(right=df[['sk_id_curr','age_years']], how='left', left_on='sk_id_curr', right_on='sk_id_curr')

    # 4. Год смены документа
    df['id_change_age'] = np.floor((df.days_birth + df.days_id_publish) / 365.25).astype('int16')
    features = features.merge(right=df[['sk_id_curr','id_change_age']], how='left', left_on='sk_id_curr', right_on='sk_id_curr')

    # 5. Разница во времени между сменой документа и возрастом на момент смены документа
    features['id_age_change_diff'] = features[['id_change_age']].map(lambda age: age - 45 if age >= 45 \
                                                                     else (age - 20 if age >= 20 \
                                                                     else age - 14))

    # 6. Признак задержки смены документа. Документ выдается или меняется в 14, 20 и 45 лет
    features['flag_late_id_change'] = features[['id_age_change_diff']].map(lambda flag: 1 if flag > 0 else 0) 

    # 7. Доля денег, которые клиент отдает на займ за год
    df['annuity_to_income_rate'] = df.amt_annuity / df.amt_income_total
    features = features.merge(right=df[['sk_id_curr','annuity_to_income_rate']], how='left', left_on='sk_id_curr', right_on='sk_id_curr')

    # 8. Среднее количество детей в семье на одного взрослого
    df['children_per_adult'] = df.cnt_children / (df.cnt_fam_members - df.cnt_children)
    features = features.merge(right=df[['sk_id_curr','children_per_adult']], how='left', left_on='sk_id_curr', right_on='sk_id_curr')

    # 9. Средний доход на ребенка
    # В случае если нет детей укажем полный доход в данном поле, 
    # так как отсутствие детей не несёт дополнительной финансовой нагрузки на заёмщика.
    df.loc[df['cnt_children'] > 0, 'inc_per_child'] = df[df['cnt_children'] > 0].amt_income_total / \
                                                      df[df['cnt_children'] > 0].cnt_children
    df.loc[df['cnt_children'] == 0, 'inc_per_child'] = df[df['cnt_children'] == 0].amt_income_total
    features = features.merge(right=df[['sk_id_curr','inc_per_child']], how='left', left_on='sk_id_curr', right_on='sk_id_curr')

    # 10. Средний доход на взрослого
    df['inc_per_adult'] = df.amt_income_total / (df.cnt_fam_members - df.cnt_children)
    features = features.merge(right=df[['sk_id_curr','inc_per_adult']], how='left', left_on='sk_id_curr', right_on='sk_id_curr')

    # 11. Процентная ставка
    # Исходим из предположения, что `amt_annuity`,- это <начисления по кредиту за год>
    df['annuity_rate'] = round(((df.amt_annuity / df.amt_credit) * 100), 3)
    features = features.merge(right=df[['sk_id_curr','annuity_rate']], how='left', left_on='sk_id_curr', right_on='sk_id_curr')

    # 12 Взвешенный score внешних источников
    # Посчитаем важность источников с помощью CatBoost-a
    # В зависимости от того присутствует ли в записи ext_score_N - посчитаем взвешенный score
    clf = CatBoostClassifier(iterations=100, auto_class_weights='Balanced', silent=True)
    df_temp = df[['ext_source_1', 'ext_source_2', 'ext_source_3', 'target']].dropna()
    clf.fit(df_temp.drop(['target'], axis=1), df_temp.target)
    weights = np.round((clf.get_feature_importance() / 100), 2) # >>> array([0.3 , 0.25, 0.45])
    df['weighted_ext_scores'] = df[['ext_source_1', 'ext_source_2', 'ext_source_3']].notna() @ weights
    features = features.merge(right=df[['sk_id_curr','weighted_ext_scores']], how='left', left_on='sk_id_curr', right_on='sk_id_curr')

    # 13. Разница между доходом заявителя и средним доходом по группе.
    train_grouped_means = df[df.target.notna()].groupby(['code_gender', 'name_education_type'])['amt_income_total'].mean()
    df = df.merge(train_grouped_means, on=['code_gender', 'name_education_type'], how='left', suffixes=('', '_mean'))
    df['composite_metric'] = df['amt_income_total'] - df['amt_income_total_mean']
    features = features.merge(right=df[['sk_id_curr','composite_metric']], how='left', left_on='sk_id_curr', right_on='sk_id_curr')
    
    # Запись признаков в файл
    features.set_index('sk_id_curr', inplace=True)
    features.to_csv(path_or_buf=PATH, index_label=features.index.name)
    print('Файл успешно сохранен')

if __name__ == '__main__':
    main()
    