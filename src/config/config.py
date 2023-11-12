import os

def db_credentials(user_type: str) -> dict:
    """
    Возвращает учетные данные для подключения к базе данных в виде словаря.

    :param user_type: Тип пользователя ('admin' или другой).
    :return: Словарь с учетными данными.
    """
    if user_type == 'admin':
        DB_ARGS = {
            "user": os.getenv('DB_USER'),
            "pswd": os.getenv('DB_PASSWORD'),
            "host": "82.147.71.130",
            "port": "5432",
            "database": "home_credit"
        }
    else:
        DB_ARGS = {
            "user": "guest",
            "pswd": 1312546,
            "host": "82.147.71.130",
            "port": "5432",
            "database": "home_credit"
        }
    return DB_ARGS
