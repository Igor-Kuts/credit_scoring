"""Скрипт для создания признаков в dvc pipeline"""
import os
import sys
import yaml

import pandas as pd

def main():

    root_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../../.."))

    # Путь до params.yaml от корневого каталога
    params_path = os.path.join(root_dir, "params.yaml")

    # Входные данные - второй аргумент после скрипта в командной строке.
    input_data = sys.argv[1]

    # Пути до результирующих файлов
    output_path = os.path.join("data", "featurized", "data_featurized.csv")

    # Создание директорий для результирующих файлов
    os.makedirs(os.path.join("data", "featurized"), exist_ok=True)

    # Параметры для сплита
    params = yaml.safe_load(open(params_path))["featurize"]

    # Загрузка .csv с помощью pandas
    df = pd.read_csv(input_data)

    # Генерация нового признака на основе параметров 
    df['weighted_sum'] = sum([
        df['ext_source_1'] * params['ext_1_multi'],
        df['ext_source_2'] * params['ext_2_multi'],
        df['ext_source_3'] * params['ext_3_multi'],
    ])
    # Выгрузка датасета с новым признаком
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    main()
