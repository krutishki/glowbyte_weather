import pandas as pd
from prophet import Prophet
from datetime import datetime
from .metrics import evaluate
import pickle
import os
from IPython.display import display
class ExperimentTracker:
    def __init__(self) -> None:
        self.experiments = []
        self.experiment_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._current_serial = 0

    def add_experiment(self, model, train, test, name=None, predict_function=None) -> None:
        if name is None:
            name = "experiment" + str(self._current_serial)
        # assert name not in [item['name'] for item in self.experiments], "Name must be unique"

        if isinstance(model, Prophet):
            predict_function = lambda df: model.predict(df)["yhat"]
        if predict_function is None:
            predict_function = model.predict

        train = train.copy()
        test = test.copy()

        train["predict"] = predict_function(train.drop("target", axis=1))
        test["predict"] = predict_function(test.drop("target", axis=1))

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
        experiment.update(
            pd.json_normalize(evaluate(test["target"], test["predict"]))
            .add_prefix("test_")
            .iloc[0]
            .to_dict()
        )

        self.experiments.append(experiment)
        self._current_serial += 1

    def get_experiment(self, name):
        for item in self.experiments:
            if item["name"] == name:
                return item

    def metrics_df(self) -> pd.DataFrame:
        return pd.json_normalize(self.experiments).drop(["train", "test"], axis=1)
    
    def display_metrics(self) -> None:
        display(self.metrics_df().style
                .highlight_min(
                    subset=["test_MAE", "test_MSE", "test_MAPE", "test_RMSE"], color="green"
                ).highlight_max(subset=["test_R^2"], color="green"))

    def get_best_experiment(self, metric = 'test_MAE'): 
        assert len(self.experiments) > 0, "You haven't run any experiment yet"
        return pd.json_normalize(self.experiments) \
            .drop(['train', 'test'], axis=1).sort_values(metric, ascending='RMSE' in metric).iloc[0]
    
    def save_to_pickle(self, storage_path = './experiments/'):
        if not os.path.exists(storage_path):
            os.mkdir(storage_path)
        path = os.path.join(storage_path, self.experiment_id + '.pickle')
        with open(storage_path + self.experiment_id + '.pickle', 'wb+') as f:
            pickle.dump(self, f)
        return path
    
    @staticmethod
    def load_experiment(experiment_id = "last", storage_path = './experiments/'):
        assert os.path.exists(storage_path), "There is no such storage"
        if experiment_id == "last":
            experiment_id = sorted(os.listdir('./experiments'))[-1].split('.')[0]
        
        pickle_path = os.path.join(storage_path, experiment_id + '.pickle')
        assert os.path.exists(pickle_path), "There is no such experiment id"

        with open(pickle_path, 'rb') as f:
            experiment = pickle.load(f)
            print(f'Loaded experiment_id {experiment_id}')
            return experiment