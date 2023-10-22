import pandas as pd


def read_datasets() -> dict[str, pd.DataFrame]:
    """
    Читает датасеты, используемые в процессе обучения.

    Returns
    --------
    Словарь со следующими ключами:
        * source_train
        * source_test
        * weather_parsed
        * holidays
        * population
    """
    result = {
        "source_train": pd.read_csv("./data/train_dataset.csv"),
        "source_test": pd.read_csv("./data/test_dataset.csv"),
        "weather_parsed": pd.read_csv(
            "./data/weather.csv", delimiter=";", encoding="UTF-8", skiprows=6, index_col=False
        ),
        "holidays": pd.read_csv("./data/holidays.csv"),
        "population": pd.read_csv("./data/population.csv"),
    }

    train = result["source_train"].copy()
    test = result["source_test"].copy()
    train['is_train'] = True
    test['is_train'] = False
    result["source_full"] = pd.concat([train, test])
    return result

def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вся подготовка датасета и feature engineering

    Аргументы:
        df (pd.DataFrame): Датасет в формате data/train_dataset.csv
    """
    datasets = read_datasets()
    holidays = prepare_holidays(datasets["holidays"].copy()).set_index("day")
    weather_parsed = prepare_parsed_weather(datasets["weather_parsed"].copy()).set_index("datetime")
    population = datasets["population"].copy()

    result = df.copy()
    result["datetime"] = pd.to_datetime(df["date"] + " " + df["time"].astype(str) + ":00:00")
    result["season"] = result["datetime"].dt.month % 12 // 3 + 1

    # Копии колонок для prophet
    result["ds"] = result["datetime"]
    result["y"] = result["target"]

    # исправляем data leak для weather_pred и weather_fact
    result.loc[~result.is_train, 'weather_fact'] = result.loc[~result.is_train, 'weather_pred']
    # result.loc[~result.is_train, 'temp_fact'] = result.loc[~result.is_train, 'temp_pred']
    result = process_weather(result).drop(['weather_fact', 'weather_pred'], axis=1).drop(["precipitation_fact", "cloudiness_fact", "cloudiness_pred"], axis=1)
    # ["precipitation_pred", "cloudiness_pred"]

    # Начинаем джойнить основной датасет с дополнительными
    result = result.set_index("datetime")

    result = result.join(holidays, how="left")
    result["holiday_type"] = result["holiday_type"].fillna(0).astype(int)
    
    result.loc[:, 'is_weekend'] = 0
    result.loc[result.index.day_of_week.isin([5, 6]), 'is_weekend'] = 1

    result = result.reset_index().merge(population, left_on = result.index.year, right_on="year", how = "left").drop("year", axis=1).set_index('datetime')

    result = result.join(weather_parsed)
    result.loc[:, weather_parsed.columns] = result.loc[:, weather_parsed.columns].bfill()

    # колонки, которые на момент на момент вызова модели известны только за вчера
    data_leak_columns = weather_parsed.columns.to_list()
    assert set(result.groupby('date').size().value_counts().index) == {24}
    result = result.join(result.loc[:, data_leak_columns].shift(24).add_suffix('_yesterday'))

    # заполнение пропусков
    result["temp_pred"] = result["temp_pred"].fillna(result["temp"].shift(24))

    assert result.shape[0] == datasets['source_train'].shape[0] + datasets['source_test'].shape[0]
    result = result.reset_index()
    result.loc[:24, [col + '_yesterday' for col in data_leak_columns]] = result.loc[:24, [col + '_yesterday' for col in data_leak_columns]].bfill()
    return result


def process_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Обработка признаков с погодой

    Аргументы:
        df (pd.DataFrame): Датасет
    """

    result = df.copy()
    result["weather_fact"] = result["weather_fact"].fillna("")
    result["weather_pred"] = result["weather_pred"].fillna("")

    result = process_precipitation(result, "pred")
    result = process_precipitation(result, "fact")
    result = process_cloudiness(result, "pred")
    result = process_cloudiness(result, "fact")

    return result


def process_precipitation(df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    result = df.copy()

    weather_col = f"weather_{suffix}"
    result[f"precipitation_{suffix}"] = "Без осадков"
    result.loc[result[weather_col].str.contains("дожд"), f"precipitation_{suffix}"] = "Дождь"
    result.loc[result[weather_col].str.contains("дожь"), f"precipitation_{suffix}"] = "Дождь"
    result.loc[result[weather_col].str.contains("лив"), f"precipitation_{suffix}"] = "Ливень"
    result.loc[result[weather_col].str.contains("гроза"), f"precipitation_{suffix}"] = "Ливень"
    result.loc[result[weather_col].str.contains("шторм"), f"precipitation_{suffix}"] = "Ливень"
    result.loc[result[weather_col].str.contains("град"), f"precipitation_{suffix}"] = "Град"
    result.loc[result[weather_col].str.contains("морос"), f"precipitation_{suffix}"] = "Морось"
    result.loc[result[weather_col].str.contains("снег"), f"precipitation_{suffix}"] = "Снег"
    result.loc[result[weather_col].str.contains("снеж"), f"precipitation_{suffix}"] = "Снег"

    return result


def process_cloudiness(df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    result = df.copy()

    weather_col = f"weather_{suffix}"
    cloud_col = f"cloudiness_{suffix}"

    result.loc[result[weather_col].str.contains("ясн"), cloud_col] = "Ясно"
    result.loc[result[weather_col].str.contains("обл"), cloud_col] = "Облачно"
    result.loc[result[weather_col].str.contains("проясн"), cloud_col] = "Прояснения"
    result.loc[result[weather_col].str.contains("обл с пр"), cloud_col] = "Прояснения"
    result.loc[result[weather_col].str.contains("обл. с  пр."), cloud_col] = "Прояснения"
    result.loc[result[weather_col].str.contains("обл.с пр"), cloud_col] = "Прояснения"
    result.loc[result[weather_col].str.contains("облачно с пр"), cloud_col] = "Прояснения"
    result.loc[result[weather_col].str.contains("обл сп"), cloud_col] = "Прояснения"
    result.loc[result[weather_col].str.contains("обл. с пр"), cloud_col] = "Прояснения"
    result.loc[result[weather_col].str.contains("п/обл"), cloud_col] = "Переменная обл"
    result.loc[result[weather_col].str.contains("пер.обл"), cloud_col] = "Переменная обл"
    result.loc[result[weather_col].str.contains("пер. обл"), cloud_col] = "Переменная обл"
    result.loc[result[weather_col].str.contains("перем.обл"), cloud_col] = "Переменная обл"
    result.loc[result[weather_col].str.contains("малообл"), cloud_col] = "Малооблачно"
    result.loc[result[weather_col].str.contains("малобл"), cloud_col] = "Малооблачно"
    result.loc[result[weather_col].str.contains("пасм"), cloud_col] = "Пасмурно"
    return result

def prepare_parsed_weather(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    col_rename = {
        'Местное время в Калининграде': 'datetime',
        'T': 'temp_parsed',
        'Po': 'atm_pressure',
        'U': 'humidity',
        'DD': 'wind_direction',
        'Ff': 'wind_speed',
        'N': 'cloudiness_percent',
        # 'Cl': 'cloudiness_category',
        'WW': 'weather_category_parsed'
    }
    result = result.rename(columns = col_rename)
    result['datetime'] = pd.to_datetime(result['datetime'])
    result = result.sort_values('datetime')

    num_features = ['temp_parsed', 'atm_pressure', 'humidity', 'wind_speed']
    cat_features = ['wind_direction', 'weather_category_parsed']

    assert len(set(num_features) & set(cat_features)) == 0
    return result[['datetime'] + num_features + cat_features]
    for col in cat_features:
        result[col] = pd.Categorical(result[col]).codes
    return result
        
    return pd.get_dummies(result[cat_features])

    return result[col_rename.keys() + cat_features]

def prepare_holidays(df):
    result = df.copy()
    result['day'] = pd.to_datetime(result['day'])
    result = result.sort_values('day').rename(columns = {'type': 'holiday_type'})
    return result[['day', 'holiday_type']]