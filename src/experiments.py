import numpy as np
import pandas as pd
from prophet import Prophet
from datetime import datetime
from .metrics import evaluate
import pickle
import os
from IPython.display import display
# import statsmodels
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
import lightgbm

def get_predict_function(model):
    if isinstance(model, Prophet):
        return lambda df: model.predict(df).reset_index()["yhat"]
    elif isinstance(model, ARIMAResultsWrapper):
        return lambda df: model.forecast(len(df.set_index('datetime')), exog = df.set_index('datetime'))
    elif isinstance(model, lightgbm.basic.Booster):
        return lambda df: model.predict(df.reset_index(drop=True).drop('datetime', axis=1, errors='ignore'))
    else:
        return model.predict

class ExperimentTracker:
    def __init__(self) -> None:
        self.experiments = []
        self.tracker_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._current_serial = 0

    def add_experiment(self, model, train, test, name=None, predict_function = None) -> None:
        if name is None:
            name = "experiment" + str(self._current_serial)
        # assert name not in [item['name'] for item in self.experiments], "Name must be unique"

        train = train.copy().reset_index()
        test = test.copy().reset_index()

        # if 'datetime' in train.columns:

        predict_function = get_predict_function(model)
        train["predict"] = np.array(predict_function(train.drop("target", axis=1)))
        test["predict"] = np.array(predict_function(test.drop("target", axis=1)))

        experiment = {
            "name": name if name != "" else str(model),
            "serial": self._current_serial,
            "model": model,
            "train": train.copy(),
            "test": test.copy(),
            "datetime": datetime.now().strftime("%Y%m%d-%H%M%S")
        }

        experiment.update(
            {f"train_{k}": v for k, v in evaluate(train["target"], train["predict"]).items()}
        )
        experiment.update(
            {f"test_{k}": v for k, v in evaluate(test["target"], test["predict"]).items()}
        )
        
        daily_train = train.groupby(pd.Grouper(key = "datetime", freq = "D"))[["target", "predict"]].sum()
        daily_test = test.groupby(pd.Grouper(key = "datetime", freq = "D"))[["target", "predict"]].sum()

        experiment.update(
            {f"daily_train_{k}": v for k, v in evaluate(daily_train["target"], daily_train["predict"]).items()}
        )
        experiment.update(
            {f"daily_test_{k}": v for k, v in evaluate(daily_test["target"], daily_test["predict"]).items()}
        )

        self.experiments.append(experiment)
        self._current_serial += 1

    def get_experiment(self, name):
        for item in self.experiments:
            if item["name"] == name:
                return item

    def metrics_df(self) -> pd.DataFrame:
        return pd.json_normalize(self.experiments).drop(["train", "test"], axis=1)
    
    def display_metrics(self, list_of_metrics = ["train_MAE", "train_R^2", "daily_train_MAE", "daily_train_R^2", "test_MAE", "test_R^2", "test_MAPE", "test_RMSE", "daily_test_MAE", "daily_test_MAPE", "daily_test_R^2", "daily_test_RMSE"]) -> None:
        metrics = self.metrics_df()[["name", "serial", "datetime"] + list_of_metrics]
        
        display(metrics.style
                .highlight_min(
                    subset = metrics.columns.intersection(["test_MAE", "daily_test_MAE"]), color="green"
                ).highlight_max(subset=metrics.columns.intersection(["daily_test_R^2", "test_R^2"]), color="green"))

    def get_best_experiment(self, metric = 'test_MAE'):
        assert len(self.experiments) > 0, "You haven't run any experiment yet"
        return pd.json_normalize(self.experiments) \
            .drop(['train', 'test'], axis=1).sort_values(metric, ascending='RMSE' in metric).iloc[0]
    
    def save(self, storage_path = './experiments/'):
        if not os.path.exists(storage_path):
            os.mkdir(storage_path)
        path = os.path.join(storage_path, self.tracker_id + '.pickle')
        with open(storage_path + self.tracker_id + '.pickle', 'wb+') as f:
            pickle.dump(self, f)
        return path
    
    @staticmethod
    def load_tracker(tracker_id = "last", storage_path = './experiments/'):
        assert os.path.exists(storage_path), "There is no such storage"
        if tracker_id == "last":
            tracker_id = sorted(os.listdir('./experiments'))[-1].split('.')[0]
        
        pickle_path = os.path.join(storage_path, tracker_id + '.pickle')
        assert os.path.exists(pickle_path), "There is no such experiment id"

        with open(pickle_path, 'rb') as f:
            experiment = pickle.load(f)
            print(f'Loaded tracker_id {tracker_id}')
            return experiment