import pandas as pd


def antifraud(data: pd.DataFrame) -> pd.DataFrame:
    """
    Функция для фильтрации фродовых заявок, на основе правил.

    Параметры:
    data: Датафрейм с заявками

    Возвращает:
    df: Отфильтрованный датафрейм.
    """
    df = data.copy()
    # Отказ в предоставлении контактных данных
    df = df[~((df.flag_cont_mobile <= 0) & (df.flag_email <= 0) & (df.flag_phone <= 0))]
    # Слишком большой трудовой стаж по отношению к возрасту
    df = df[~((df.days_birth - df.days_employed) / 365 < 18)] 
    # Только открытые микро-займы
    df = df[~((df.cnt_closed_Microloan == 0) & (df.Microloan > 1))]
    # Только просроченные микро-займы
    df = df[~((df.cnt_closed_Microloan == 0) & (df.cnt_overdue_Microloan >= 1))]
    # Слишком большой размер займа по отношению к доходу 
    df = df[~(df.amt_credit > df.amt_income_total * 20)]

    print(f'Правилами антифрода будет отфильтровано {len(data) - len(df)} заявок, продолжить?')
    flag = str(input('\n ДА/нет: '))
    if flag == 'ДА': 
        print('Антифрод фильтр применён')
        return df
    else: 
        print('Применение антифрод фильтра отменено')
        return data
