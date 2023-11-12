from typing import List

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from scipy.stats import mannwhitneyu, chi2_contingency, ttest_ind
from scipy.stats import anderson
from scipy.stats import chi2

def evaluate_test_results(significant_features):
    """
    Оценка результатов тестов и возврат списка значимых признаков.
    
    Args:
    значимые_признаки (dict): Словарь, содержащий результаты тестов.

    Returns:
    list: Список значимых признаков.
    """
    # Если оба теста == 1, то 1. В противном случае, 0.
    significant_features = {i: 1 if sum(significant_features[i].values()) /
                                    len(significant_features[i].values()) > 0.5
                            else 0 for i, j in significant_features.items()}

    # Фильтрация значимых признаков
    significant_features = [i for i, j in significant_features.items()
                            if significant_features[i] == 1]
    
    return significant_features

def get_notnone_features(data: pd.DataFrame, notnone_pct: float, columns_toignore: List) -> List:
    """
    Проверка количества пропущенных значений в признаках и возвращение списков,
    (численные и категориальные) содержащих названия признаков 
    с количество пропусков < (1 - notnone_pct)

    Параметры:
    data: Данные.
    notnone_pct: Доля значений без пропусков - условие фильтрации
    columns_toignore: Список названий столбцов для игнорирования при фильтрации

    Возвращает:
    numerical_column_names: Список с названиями численных признаков удоволетворяющих условию
    categorical_column_names: Список с названиями категориальных признаков удоволетворяющих условию
    """
    
    # Списки для записи результатов фильтрации
    numerical_column_names = []
    categorical_column_names = []

    for column in data.drop(columns_toignore, axis=1).columns:

        # Оставим только признаки с количеством пропусков < (1 - notnone_pct)
        if data[column].notna().sum() / data.shape[0] >= notnone_pct:
            
            # Проверки типов данных в столбцах
            if pd.api.types.is_object_dtype(data[column]) or pd.api.types.is_bool_dtype(data[column]):
                categorical_column_names.append(column)
            else: numerical_column_names.append(column)
                
    return numerical_column_names, categorical_column_names

def anderson_darling_test(data: pd.DataFrame, feature: str, target: str) -> bool:
    """
    Проверка Н_0, о том что выборка (признак) была получена из генеральной совокупности
    следующей нормальному распределению для всех вариантов целевой переменной.

    Параметры:
    data: Данные.
    feature: Признак для проверки.
    target: Целевая переменная.

    Возвращает
    True: Нормальное распределение, False: Ненормальное распределение.
    """
    statistics = np.zeros(data.target.nunique(dropna=False))
    critical_values = np.zeros(data.target.nunique(dropna=False))

    for i, target_val in enumerate(data[target].unique()):
        target_data = data[data[target] == target_val][feature]
        sample_size = target_data.shape[0]

        # Проверка, достаточно ли данных для теста
        if sample_size < 42:
            continue  # Пропустим группу если в ней недостаточно элементов

        stat, crit_value, sign_level = anderson(target_data, "norm")
        statistics[i] = stat
        critical_values[i] = crit_value[4]

    return any(statistics <= critical_values)


def students_ttest(data: pd.DataFrame, feature: str) -> int:
    """
    Student's T-Test.
    Возвращает решение о значимости признака.
    
    Параметры
    data: Данные
    feature: Признак для проверки.
    
    Возвращает:
    1: Признак значимый, 0: Незначимый.
    """

    t_stat, p_ttest = ttest_ind(data[data['target'] == 0][feature], data[data['target'] == 1][feature], nan_policy='omit')
    if p_ttest < 0.05:
        return 1
    elif p_tt >= 0.05:
        return 0
    
def mannwhitneyu_test(data:pd.DataFrame, feature:str) -> int:
    """
    Определение значимости признаков тестом Манна-Уитни

    Параметры:
    data: Данные.
    feature: Признак для проверки.

    Возвращает:
    1: Признак значимый, 0: Незначимый.
    """

    u_stat, p_mann = mannwhitneyu(data[data['target'] == 0][feature],
                                  data[data['target'] == 1][feature],
                                  nan_policy='omit')
    if p_mann < 0.05: return 1
    elif p_mann >= 0.05: return 0

def verdict(ci_diff):
    """
    Возращает решение о важности признака на основе доверительного
    интервала для метода Bootstrap.
    """
    # Устанавливаем минимальный порог для разницы доверительных интервалов на уровне 0,1%.
    cidiff_min=0.001
    
    # Приведём все значения к модулю
    ci_diff_abs = [abs(ele) for ele in ci_diff]

    # Проверяем условия, чтобы определить вердикт.
    if (min(ci_diff) <= cidiff_min <= max(ci_diff)):
        # Если минимальное значение ci_diff меньше или равно порогу 
        # и порог находится в пределах ci_diff, возвращаем 0 (не важно).
        return 0
    if (cidiff_min >= max(ci_diff_abs) >= 0) or (cidiff_min >= min(ci_diff_abs) >= 0):
        # Если порог больше или равен максимальной абсолютной разнице 
        # или порог больше или равен минимальной абсолютной разнице, возвращаем 0 (не важно).
        return 0
    else:
        # Если ни одно из вышеперечисленных условий не выполняется, возвращаем 1 (важно).
        return 1

def bootstrap(
        data: pd.DataFrame,
        feature: str,
        target: str,
        stat_func=np.mean,
        iterations=1000,
        alpha=0.05 # Уровень значимости для 95% доверительного интервала
) -> int:
    """
    Возвращает решение о значимости признака методом Bootstrap
    
    Параметры:
    data: Данные.
    feature: Проверяемый признак.
    target: Целевая переменная.
    stat_func: Проверяемая статистика.
    iterations: Количество итераций.
    alpha: Уровень значимости.
    
    Возвращает:
    1: Признак значимый, 0: Незначимый
    """
    data1 = data[(data[target] == 0) & (data[feature].notna())][feature]
    data2 = data[(data[target] == 1) & (data[feature].notna())][feature]

    sample_1, sample_2 = [], []

    for iter_ in range(iterations):
        iter_sample_1 = data1.sample(data1.shape[0], replace=True)
        sample_1.append(stat_func(iter_sample_1))
        iter_sample_2 = data2.sample(data2.shape[0], replace=True)
        sample_2.append(stat_func(iter_sample_2))
    sample_1.sort()
    sample_2.sort()
    
    # Convidence Interval для разницы
    bootdiff = np.subtract(sample_2, sample_1)
    bootdiff.sort()
    ci_diff = (np.round(bootdiff[np.round(iterations * alpha / 2).astype(int)], 3),
               np.round(bootdiff[np.round(iterations * (1 - alpha / 2)).astype(int)], 3))

    return verdict(ci_diff)

def chi_squared_test(data: pd.DataFrame, feature: str, target: str) -> int:
    """
    Chi-Squared тест. Возвращает решение о значимости бинарных
    и категориальных признаков.
    
    Параметры:
    data: Данные.
    feature: Проверяемый признак.
    target: Целевая переменная.
    
    Возвращает:
    1: Признак значимый, 0: Незначимый
    """
    cross_tab = pd.concat([
            pd.crosstab(data[feature], data[target], margins=False),
            data.groupby(feature)[target].agg(['count', 'mean']).round(4)
        ], axis=1).rename(columns={False: f"target=0", True: f"target=1", "mean": 'probability_of_default'})

    cross_tab['probability_of_default'] = np.round(cross_tab['probability_of_default'].astype('float') * 100, 2)
    cross_tab

    chi2_stat, p, dof, expected = chi2_contingency(cross_tab.values)
    prob = 0.95
    critical = chi2.ppf(prob, dof)
    return abs(chi2_stat) >= critical
