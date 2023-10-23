import argparse
import os
import pandas as pd
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an energy consumption prediction model for Glowbyte by Krutishki team"
    )
    parser.add_argument("-p", type=str, help="Path to input CSV file", required=True)
    parser.add_argument("-m", type=str, help="Path to model pickle", required=False, default='./best_model.pickle')
    args = parser.parse_args()

    file_path = args.p
    model_path = args.m

    assert os.path.exists(file_path), 'Path does not exist'
    data = pd.read_csv(file_path)
    print(f'Read data from {file_path} successfully')

    assert os.path.exists('./best_model.pickle'), "Model is not serialized. Run ml_model.ipynb first"

        

