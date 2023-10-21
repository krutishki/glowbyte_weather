from abc import ABC, abstractmethod

import pandas as pd


class Model(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.api.extensions.ExtensionArray:
        ...


class BaselineYearAgo(Model):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Обучение модели (запоминание истории)

        Аргументы:
            X (pd.DataFrame): Временной ряд, содержаший колонку datetime
            y (pd.Series): целевая переменная. Длины X и y должны совпадать.
        """
        assert "datetime" in X.columns
        self.timeline = pd.concat([X, y.rename("target")], axis=1)
        self.timeline = self.timeline.sort_values("datetime")

    def predict(self, X: pd.DataFrame) -> pd.api.extensions.ExtensionArray:
        """
        Обучение модели (запоминание истории)

        Аргументы:
            X (pd.DataFrame): Временной ряд, содержаший колонку datetime

        Возвращаемое значение:
            np.array, предсказание модели.
        """
        assert "datetime" in X.columns
        X["source_datetime"] = X.datetime - pd.DateOffset(years=1)

        result = X.merge(
            self.timeline.copy()[["datetime", "target"]].rename(
                columns={"datetime": "source_datetime", "target": "predict"}
            ),
            on=["source_datetime"],
        )["predict"]
        assert not result.isna().any()
        assert result.shape[0] == X.shape[0]

        return result.array


def aggregated_daily_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """df - датафрейм с predict, target и datetime"""
    return df.groupby(df["datetime"].dt.date)[["predict", "target"]].sum().reset_index()
