import argparse
import os
import pandas as pd
import pickle

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
    data = pd.read_csv(file_path)
    print(f'Read data from {file_path} successfully')
    data = prepare_dataset(data).set_index('datetime')

    last_experiment = ExperimentTracker().load_tracker()
    model = last_experiment.get_best_experiment()['model']
    result = get_predict_function(model)(data)
    result = aggregated_daily_predictions(result)
    result.to_csv('prediction_team21.csv')

    print('Successfully saved predictions to prediction_team21.csv')