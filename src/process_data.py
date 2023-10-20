import pandas as pd

def read_datasets():
    '''
    Читает датасеты, используемые в процессе обучения.

    Returns
    --------
    Словарь со следующими ключами:
        * source_train
        * source_test
        * weather_parsed
        * holidays
        * population
    '''
    return {
        'source_train': pd.read_csv('./data/train_dataset.csv'),
        'source_test': pd.read_csv('./data/test_dataset.csv'),
        'weather_parsed': pd.read_csv('./data/weather.csv', delimiter=';', encoding="UTF-8", skiprows=6),
        'holidays': pd.read_csv('./data/holidays.csv'),
        'population': pd.read_csv('./data/population.csv'),
    }

def prepare_dataset(df):
    """
    Вся подготовка датасета и feature engineering

    Аргументы:
        df (pd.DataFrame): Датасет в формате data/train_dataset.csv
    """
    result = df.copy()
    result['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'].astype(str) + ':00:00')
    return result