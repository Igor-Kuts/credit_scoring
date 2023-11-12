"""Скрипт для автоматизации разделения данных на train и val в dvc pipeline"""
import os
import sys
import yaml

import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    root_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../../.."))

    # Путь до params.yaml от корневого каталога
    params_path = os.path.join(root_dir, "params.yaml")

    # Входные данные - второй аргумент после скрипта в командной строке.
    input_data = sys.argv[1]

    # Пути до результирующих файлов
    output_train = os.path.join("data", "prepared", "train_featurized.csv")
    output_val = os.path.join("data", "prepared", "val_featurized.csv")

    # Создание директорий для результирующих файлов
    os.makedirs(os.path.join("data", "prepared"), exist_ok=True)

    # Параметры для сплита
    params = yaml.safe_load(open(params_path))["prep_splits"]

    # Размер тестовой выборки и random_state
    split = params['test_size']
    random_state = params['random_state']

    # Загрузка .csv с помощью pandas
    df = pd.read_csv(input_data)
    features, target  = df.drop('target', axis=1), df.target

    # Разделение на train и val
    x_train, x_val, y_train, y_val = train_test_split(
        features, target, 
        test_size=split, 
        random_state=random_state,
    )
    train_data = pd.concat([x_train, y_train], axis=1)
    val_data = pd.concat([x_val, y_val], axis=1)

    # Выгрузка результатов сплита
    train_data.to_csv(output_train, index=False)
    val_data.to_csv(output_val, index=False)

if __name__ == '__main__':
    main()
