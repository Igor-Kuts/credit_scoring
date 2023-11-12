import sys
sys.path.extend(['../../../config', '../../../app/utils'])

import pandas as pd
import numpy as np
from tqdm import tqdm

from database_manager import PGSQL_Manager
from config import db_credentials

from stat_utils import (evaluate_test_results,
                        get_notnone_features,
                        anderson_darling_test,
                        students_ttest,
                        mannwhitneyu_test,
                        chi_squared_test,
                        bootstrap
)    

# Установим гостевое соединение с базой данных
pgsql_guest = PGSQL_Manager()
DB_ARGS = db_credentials(user_type="guest")
engine = pgsql_guest.connect(**DB_ARGS)

path_output = str(input('Input path to save file: '))

def main(path_output):

    df = pgsql_guest.get_df_from_query('SELECT * FROM application_train_test')
    df = df.dropna(subset='target')

    columns_toignore=['sk_id_curr', 'target']
    numerical_columns, categorical_columns = get_notnone_features(df, 
                                                                  notnone_pct=0.7, 
                                                                  columns_toignore=columns_toignore)
    significant_features = {}
    
    print('Обработка численных признаков..')
    for feature in tqdm(numerical_columns):
        tmp = df[[feature, 'target']].fillna(df[feature].median())

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

    # Фильтрация признаков на основе результатов тестов
    significant_features = evaluate_test_results(significant_features)
    
    # Выгрузим данные
    df[['sk_id_curr'] + significant_features].to_csv(path_output, index=False)
    
if __name__=='__main__':
    main(path_output)
    