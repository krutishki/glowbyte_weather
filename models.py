from abc import ABC
import pandas as pd
import numpy as np

class Model(ABC):
    def fit(self, df):
        pass

    def predict(self, df):
        pass

class BaselineYearAgo(Model):
    def fit(self, X, y):
        self.timeline = pd.concat([X, y.rename('target')], axis=1)
        self.timeline = self.timeline.sort_values('datetime')

    def predict(self, X):
        assert 'datetime' in X.columns
        X['source_datetime'] = X.datetime - pd.DateOffset(years=1)

        result = X.merge(self.timeline[['datetime', 'target']].rename(columns = {'datetime': 'source_datetime', 'target': 'predict'}), on = ['source_datetime'])['predict']
        assert not result.isna().any()

        return result