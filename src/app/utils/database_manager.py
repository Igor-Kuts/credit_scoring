from sqlalchemy import create_engine, text
from sqlalchemy import SmallInteger, Integer, REAL, Text, Boolean
import psycopg2
import pandas as pd
import numpy as np

class PGSQL_Manager:
    def __init__(self):
        self.engine = None
        
    def connect(self, user, pswd, host, port, database):
        """
        Устанавливает соединение с базой данных PostgreSQL
        используя предоставленные учётные данные.

        Параметры:
        user (str):     Имя пользователя PostgreSQL.
        pswd (str):     Пароль PostgreSQL.
        host (str):     Имя хоста или IP-адрес сервера базы данных.
        port (str):     Номер порта сервера базы данных.
        database (str): Название базы данных PostgreSQL.

        Возвращает:
        sqlalchemy.engine.base.Connection: SQLAlchemy соединение с базой данных.
        """

        # Создание URL для подключения к базе данных
        connection_url = f"postgresql+psycopg2://{user}:{pswd}@{host}:{port}/{database}"

        # Создание SQLAlchemy движка
        self.engine = create_engine(connection_url)

        # Тестирование соединения:
        """ 
        Движок SQLAlchemy создается в "ленивом" режиме и не подключается
        до его непосредственного вызова.
        """
        try:
            """ 
            Контекстный менеджер (with) автоматически закрывает подключение
            при выходе из его блока.
            """
            with self.engine.connect() as connection:
                print("Соединение установлено")

                # Возврат исправного соединения
                return self.engine

        # Сообщение об ошибке
        except Exception as e:
            print(f"Не удалось установить соединение: {e}")
            
            
    def evaluate_pandas_dtypes(self, dataframe: pd.DataFrame) -> dict:
        """
        Функция находит более оптимальные типы данных SQLAlchemy для столбцов pd.DataFrame.
        Используется перед записью таблицы в базу данных PostgreSQL

        Параметры:
        dataframe (pd.DataFrame): Pandas DataFrame

        Возвращает:
        dtypes (dict): Словарь содержащий новые соответствия column:dtype. 
        """
        dtypes = {}  # Создаем пустой словарь для хранения типов данных

        for col in dataframe.columns:
            # Проверяем, является ли столбец числовым
            if pd.api.types.is_numeric_dtype(dataframe[col]):
                # Если столбец содержит только 0 и 1, то устанавливаем Boolean
                if set(dataframe[col].dropna().unique()) == {0, 1}:
                    dtypes[col] = Boolean
                # Если значения в столбце входят в диапазон int16, то устанавливаем SmallInteger
                elif dataframe[col].min() >= np.iinfo('int16').min and dataframe[col].max() <= np.iinfo('int16').max:
                    dtypes[col] = SmallInteger
                # Если значения в столбце являлись целыми числами
                # перед импортом из .csv, то устанавливаем Integer
                elif (dataframe[col].dropna() == dataframe[col].dropna().round(0)).all():
                    dtypes[col] = Integer
                # В остальных случаях устанавливаем REAL
                else:
                    dtypes[col] = REAL
            else:
                # Если столбец не числовой и содержит 'Y' и 'N', то заменяем их на 1 и 0 и устанавливаем Boolean
                if set(dataframe[col].dropna().unique()) == {'Y', 'N'}:
                    dataframe.loc[:, col] = dataframe[col].map({'Y': 1, 'N': 0})
                    dtypes[col] = Boolean
                # В остальных случаях устанавливаем Text
                else:
                    dtypes[col] = Text

        return dtypes  # Возвращаем словарь с соответствиями типов данных

    
    
    def get_df_from_query(self, query: str) -> pd.DataFrame:
        """
        Выполняет запрос к базе на основе строкового запроса.
        
        Параметры:
        query (str): Строка с sql запросом.
        
        Возвращает:
        df: Датафрейм с результатом.
        
        Raises:
        ValueError: Сообщение об ошибке, если engine не определён.
        """
        # Проверяем, инициализированно ли соединение
        # Если инициализированно, - посылаем запрос на данные
        if self.engine:
            
            # pd.read_sql автоматически закроет подключение.
            df = pd.read_sql(query, self.engine)
            return df
        
        # Если нет, то возвращаем ValueError с комментарием.
        else:
            raise ValueError("SQLAlchemy engine не определен. Установите соединение с базой данных.")
            
            
    def send_sql_query(self, query: str):
        """
        Выполняет sql запрос к базе.

        Параметры:
        query: строка с sql запросом.
        
        Возвращает:
        resul: Результат выполнения sql запроса
        """
        # Проверяем, инициализированно ли соединение
        if self.engine is not None:
            try:
                with self.engine.connect() as connection:
                    result = connection.execute(text(query))

                    # Коммит нужен в том случае, если мы вносим изменения в базу данных  
                    connection.commit() 
                    
                    # Возвращаем результат выполнения запроса
                    return result

            except Exception as e:
                print(f"Ошибка при выполнении SQL запроса: {e}")
                # Можно выбросить исключение или вернуть None в случае ошибки.
                raise e

        else:
            raise ValueError("SQLAlchemy engine не определен. Установите соединение с базой данных.")