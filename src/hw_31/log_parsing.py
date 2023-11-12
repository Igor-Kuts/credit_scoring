from dataclasses import dataclass, asdict
import json

import pandas as pd
from tqdm import tqdm

# Вспомогательная функция
def parse_bureau(json_row):
    """
    Считывает и извлекает соответствующую информацию из json строки.

    Параметры:
    json_row (dict): Словарь, содержащий json данные.

    Возвращает:
    list: Список извлечённой информации из json строки для последующей записи.
    """
    # Манипуляция данными в строке
    json_row['data']['record']['CREDIT_TYPE'] = json_row['data']['CREDIT_TYPE']

    # Чтение и извлечение информации из вложенного объекта
    json_row['data']['record']['AmtCredit'] = eval(json_row['data']['record']['AmtCredit'])
    for param, value in asdict(json_row['data']['record']['AmtCredit']).items():
        json_row['data']['record'][param] = value

    del json_row['data']['CREDIT_TYPE']
    del json_row['data']['record']['AmtCredit']

    # Преобразование данных в список
    bureau_data_row = list(json_row['data']['record'].values())
    
    return bureau_data_row

# Вспомогательная функция
def parse_poscash(json_row):
    """
    Считывает и извлекает соответствующую информацию из json строки.

    Параметры:
    json_row (dict): Словарь, содержащий json данные.

    Возвращает:
    list: Список извлечённой информации из json строки для последующей записи.
    """
    # Из-за особенности структуры логов pos_cash_balance
    # будем формировать и сохранять строки
    # в ходе прохождения по списку с вложенными объектами.
    poscash_data = []
    for record in tqdm(json_row['data']['records']):

        # Составление строки для записи
        poscash_data_row = []
        poscash_data_row.extend([
            json_row['data']['CNT_INSTALMENT'],
            record['CNT_INSTALMENT_FUTURE'],
            record['MONTHS_BALANCE']])

        # Извлечение информации из вложенного объекта
        record['PosCashBalanceIDs'] = eval(record['PosCashBalanceIDs'])
        for value in asdict(record['PosCashBalanceIDs']).values():
            poscash_data_row.append(value)

        # Составление строки данных
        poscash_data_row.extend([
            record['SK_DPD'],
            record['SK_DPD_DEF']
        ])
        
        poscash_data.append(poscash_data_row)
        
    return poscash_data


# Пользовательский ввод путей до файла логов
# и файлов, в которые будет производиться запись
log_filename = str(input('Введите путь до файла с логами '))
bureau_csv_destination = str(input('Записать данные о bureau в: '))
poscash_csv_destination = str(input('Записать данные о pos_cash_balance в: '))

# Заголовок для DataFrame содержащего информацию о bureau
bureau_header = [
    'SK_ID_CURR', 'SK_ID_BUREAU', 'CREDIT_ACTIVE', 'DAYS_CREDIT', 'CREDIT_DAY_OVERDUE',
    'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT', 'CNT_CREDIT_PROLONG', 'DAYS_CREDIT_UPDATE',
    'CREDIT_TYPE', 'CREDIT_CURRENCY', 'AMT_CREDIT_MAX_OVERDUE', 'AMT_CREDIT_SUM',
    'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE','AMT_ANNUITY'
]

# Заголовок для DataFrame содержащего информацию о pos_cash_balance
poscash_header = [
    'CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE', 'MONTHS_BALANCE', 'SK_ID_PREV',
    'SK_ID_CURR', 'NAME_CONTRACT_STATUS', 'SK_DPD', 'SK_DPD_DEF'
]

# Структура вложенного объекта
@dataclass
class AmtCredit:
    CREDIT_CURRENCY: str
    AMT_CREDIT_MAX_OVERDUE: float
    AMT_CREDIT_SUM: float
    AMT_CREDIT_SUM_DEBT: float
    AMT_CREDIT_SUM_LIMIT: float
    AMT_CREDIT_SUM_OVERDUE: float
    AMT_ANNUITY: float

# Структура вложенного объекта
@dataclass
class PosCashBalanceIDs:
    SK_ID_PREV: int
    SK_ID_CURR: int
    NAME_CONTRACT_STATUS: str

    
def main(log_filename=log_filename, 
         bureau_csv_destination=bureau_csv_destination, 
         poscash_csv_destination=poscash_csv_destination):
    
    # Списки для записи строк
    bureau_rows = []
    poscash_rows = []
    
    # Открытие .log файла для чтения
    with open('POS_CASH_balance_plus_bureau-001.log', 'r') as log_file:

        # Построчное чтение .log файла
        for line in tqdm(log_file):
            json_row = json.loads(line)

            # Блок обработки данных из таблицы bureau
            if json_row['type'] == 'bureau':

                # Обработка и сохранение строки данных в виде списка
                bureau_data_row = parse_bureau(json_row)
                bureau_rows.append(bureau_data_row)

                
            # Блок обработки данных из таблицы posh_cash_balance
            if json_row['type'] == 'POS_CASH_balance':

                # Обработка и сохранение строки данных в виде списка
                poscash_data = parse_poscash(json_row)
                poscash_rows += poscash_data
    
    # Формирование DataFrame-ов и выгрузка их в виде .csv файлов
    bureau_df = pd.DataFrame(data=bureau_rows, columns=bureau_header)
    poscash_df = pd.DataFrame(data=poscash_rows, columns=poscash_header)
    
    bureau_df.to_csv(bureau_csv_destination, index=False)
    poscash_df.to_csv(poscash_csv_destination, index=False)

    print(f'\nУспех! \nДанные из {log_filename} были записаны в {bureau_csv_destination} и {poscash_csv_destination}')

if __name__ == "__main__":
    main()
    