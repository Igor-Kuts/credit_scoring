"Вспомогательные функции для предобработки данных"
from typing import List

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

# Вспомогательная функция
def one_hot_encoding(df, ohe_col_list, categories='auto', drop_policy=None):
    """
    Производит one-hot кодирование указанных столбцов в датафрейме и удаляет оригинальные столбцы.

    Параметры:
    df (pandas.DataFrame): Исходный датафрейм.
    ohe_col_list (список): Список столбцов, подлежащих one-hot кодированию.
    drop_policy (строка, по умолчанию None): Политика удаления исходных столбцов.
        None - не удалять исходные столбцы.
        'first' - удалить первый исходный столбец после one-hot кодирования.

    Возвращает:
    pandas.DataFrame: Датафрейм с one-hot кодированными данными.

    Примечание:
    Функция сохраняет исходный датафрейм и возвращает новый с one-hot кодированными данными.
    """
    # Создадим копию датафрейма перед работой
    df_copy = df.copy()
    
    encoder = OneHotEncoder(sparse_output=False, categories=categories, drop=drop_policy)
        
    onehot_encoded_array = encoder.fit_transform(df_copy[ohe_col_list])

    # Создадим новый датафрейм с закодированными данными
    onehot_encoded_df = pd.DataFrame(
        onehot_encoded_array,
        columns=encoder.get_feature_names_out(ohe_col_list),
        index = df_copy.index,
        dtype='int8')

    # Удалим оригинальные столбцы 
    df_copy.drop(ohe_col_list, axis=1, inplace=True)

    # Сконкатенируем датафреймы
    df_copy = pd.concat([df_copy, onehot_encoded_df], axis=1)
      
    return df_copy

def find_outliers(data, low=0.25, high=0.75, coef=1.5):
    """
    Идентифицирует выбросы в числовых столбцах датафрейма с использованием квантилей и межквартильного размаха.

    Параметры:
    data (pandas.DataFrame): Датафрейм с данными.
    low (float, по умолчанию 0.25): Нижний квантиль для расчета IQR.
    high (float, по умолчанию 0.75): Верхний квантиль для расчета IQR.
    coef (float, по умолчанию 1.5): Множитель для определения выбросов.

    Возвращает:
    numpy.ndarray: Массив с информацией о столбцах, количестве выбросов и индексах выбросов.
    """
    columns = data.columns
    outliers_arr = np.zeros((len(data),3), dtype='object')
    outliers_num = 0
    
    for idx, col in enumerate(columns):
        q1 = np.quantile(data[col], low)
        q3 = np.quantile(data[col], high)
        
        iqr = q3 - q1
        
        # Calculate the lower and upper cutoffs for outliers
        lower = q1 - coef * iqr
        upper = q3 + coef * iqr
        
        outliers = data[(data[col] < lower) | (data[col] > upper)]
        
        if len(outliers) > 0:
            
            # Subset df to find outliers
            outliers_arr[idx,0] = col
            outliers_arr[idx,1] = outliers.shape[0]
            outliers_arr[idx,2] = list(outliers.index)
            outliers_num += 1

    nonzero_rows = np.any(outliers_arr != 0, axis=1)
    outliers_arr = outliers_arr[nonzero_rows]  
    
    print(f'В данных {outliers_arr.shape[0]} столбцов с выбросами')
    print(f'В данных {np.sum(outliers_arr[:,1])} Выбросов')

    return outliers_arr

def drop_outliers(data, outliers_arr):
    """
    Удаляет выбросы из датафрейма на основе информации, предоставленной find_outliers.

    Параметры:
    data (pandas.DataFrame): Датафрейм с данными.
    outliers_arr (numpy.ndarray): Массив с информацией о столбцах, выбросах и их индексах.

    Возвращает:
    pandas.DataFrame: Датафрейм без удаленных строк, содержащих выбросы.
    """
    outliers_set = set([])

    for row in range(outliers_arr.shape[0]):
        outliers_set.update(set(outliers_arr[row,2]))
       
    print(f'Будет удалено {len(outliers_set)} строк')
    print(f'Будет удалено {np.sum(outliers_arr[:,1])} выбросов')
    
    if (input('Удаляем выбросы? Y/n: ')) == 'Y':
        
        data_clean = data.loc[~data.index.isin(outliers_set)]
        print('Готово')
        return data_clean
        
    else: 
        print('Отменено')
        return data

def get_column_types(data: pd.DataFrame, columns_to_ignore: List = []) -> List:
    """
    Проверяет типы данных в столбцах pandas.DataFrame.

    Параметры:
    data (pandas.DataFrame): Датафрейм с данными.
    columns_to_ignore (List): Список с названиями столбцов, которые не нужно учитывать.

    Возвращает:
    numerical_columns: Список названий столбцов с численными признаками.
    categorical_columns: Список названий столбцов с категориальными признаками.
    """
    numerical_columns, categorical_columns = [], []
    for column in data.drop(columns_to_ignore, axis=1).columns:
        if pd.api.types.is_object_dtype(data[column]) or pd.api.types.is_bool_dtype(data[column]):
            categorical_columns.append(column)
        else: numerical_columns.append(column)

    return numerical_columns, categorical_columns

def load_and_split_data(data_path: str) -> pd.DataFrame:
    """Convinience function: загружает данные и делит их на train и test"""
    data = pd.read_csv(data_path, low_memory=False)
    data_train = data[data.target.notna()].copy()
    data_train['target'] = data_train['target'].astype('bool')
    
    x_train = data_train.drop(['sk_id_curr', 'target'], axis=1)
    y_train = data_train.target

    return x_train, y_train
class DtypeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List, new_dtype: str):
        """
        Пользовательская реализация SKLearn трансформера
        для преобразования типов данных в pipeline.

        Параметры:
        - columns: Список столбцов для смены типа данных.
        - new_dtype: Новый тип данных для смены.
        """
        self.columns = columns
        self.new_dtype = new_dtype

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit необходим для реализации трансформера.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Смена типа данных в столбцах.

        Параметры:
        - X: Входные данные.

        Возвращает:
        - X_transformed: Трансформированный датафрейм со сменёнными типами данных.
        """

        # Создадим копию датафрейма чтобы избежать модификации исходных данных
        X_transformed = X.copy()  
        for column in self.columns:
            X_transformed[column] = X_transformed[column].astype(self.new_dtype)
        
        return X_transformed
        
