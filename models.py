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
        self.timeline['predict'] = self.timeline.groupby([self.timeline['datetime'].dt.month, self.timeline['datetime'].dt.day, self.timeline['datetime'].dt.hour])['target'].shift()
        self.timeline['predict'] = self.timeline['predict'].fillna(np.mean(y)) # костыль, надо понять откуда наны, может високосный год

    def predict(self, df):
        assert 'datetime' in df.columns
        
        return self.timeline.merge(df[['datetime']], how = 'right')['predict']