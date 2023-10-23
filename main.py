import argparse
import os
import pandas as pd
import pickle
import lightgbm

from src.experiments import ExperimentTracker, get_predict_function
from src.models import aggregated_daily_predictions
from src.process_data import prepare_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an energy consumption prediction model for Glowbyte by Krutishki team (team 21)"
    )
    parser.add_argument("filename", type=str, help="Path to input CSV file")
    parser.add_argument("-m", type=str, help="Path to experiment storage", required=False, default='./experiments')
    args = parser.parse_args()

    file_path = args.filename
    model_path = args.m

    assert os.path.exists(file_path), 'Path does not exist'
    data = pd.read_csv(file_path, index_col=False)
    # for col in ['is_train', 'Unnamed: 0']:
    #     if col in data.columns:
    #         data = data.drop(col, axis=1)
    print(f'Read data from {file_path} successfully')
    data = prepare_dataset(data).set_index('datetime')

    last_experiment = ExperimentTracker().load_tracker()
    model = last_experiment.get_best_experiment()['model']

    removed_cols = ['date', 'ds', 'wind_direction', 'wind_direction_yesterday']
    needed_cols = ['time', 'temp', 'temp_pred', 'season', 'holiday_type',
       'is_weekend', 'population', 'temp_parsed', 'atm_pressure', 'humidity',
       'wind_speed', 'temp_parsed_yesterday', 'atm_pressure_yesterday',
       'humidity_yesterday', 'wind_speed_yesterday', 'lag1', 'lag2', 'lag3',
       'lag4', 'lag10', 'rolling2', 'rolling3', 'rolling4',
       'precipitation_pred_downpour', 'precipitation_pred_no',
       'precipitation_pred_rain', 'precipitation_pred_snow']
    if isinstance(model, lightgbm.basic.Booster): # bad code, todo: move to prepare_dataset
        data = pd.get_dummies(data.drop(removed_cols, axis=1))[needed_cols]
    print('Calling predict...')
    prediction = pd.Series(get_predict_function(model)(data)).rename('predict')
    result = data.reset_index().join(prediction)
    result = aggregated_daily_predictions(result.reset_index()).rename(columns={'datetime': 'date'})[['date', 'predict']]
    result.to_csv('prediction_team21.csv')

    print('Successfully saved predictions to prediction_team21.csv')