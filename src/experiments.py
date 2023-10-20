import pandas as pd
from prophet import Prophet
from .metrics import evaluate

class ExperimentTracker:
    def __init__(self) -> None:
        self.experiments = []
        self._current_run_id = 0

    def add_experiment(self, model, train, test, name = None, predict_function = None):
        if name == None:
            name = "experiment" + str(self._current_run_id)
        # assert name not in [item['name'] for item in self.experiments], "Name must be unique"

        if isinstance(model, Prophet):
            predict_function = lambda df: model.predict(df)['yhat']
        if predict_function == None:
            predict_function = model.predict
        
        train = train.copy()
        test = test.copy()

        train['predict'] = predict_function(train.drop('target', axis=1))
        test['predict'] = predict_function(test.drop('target', axis=1))

        experiment = {
            'name': name if name != "" else str(model),
            'run_id': self._current_run_id,
            'model': model,
            'train': train.copy(),
            'test': test.copy()
        }

        experiment.update({f"train_{k}": v for k, v in evaluate(train['target'], train['predict']).items()})
        experiment.update({f"test_{k}": v for k, v in evaluate(test['target'], test['predict']).items()})
        # experiment.update(pd.json_normalize(evaluate(test['target'], test['predict'])).add_prefix('test_').iloc[0].to_dict())

        self.experiments.append(experiment)
        self._current_run_id += 1

    def get_experiment(self, name):
        for item in self.experiments:
            if item['name'] == name:
                return item

    def metrics_df(self):
        return pd.json_normalize(self.experiments).drop(['train', 'test'], axis=1)