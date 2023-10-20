import pandas as pd

def prepare_dataset(df):
    """
    Вся подготовка датасета и feature engineering

    Аргументы:
        df (pd.DataFrame): Датасет в формате data/train_dataset.csv
    """
    result = df.copy()
    result['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'].astype(str) + ':00:00')
    return result