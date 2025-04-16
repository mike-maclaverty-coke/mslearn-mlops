# Import libraries
import argparse
import glob
import os
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Define functions
def main(args):
    mlflow.autolog()

    # Read data
    df = get_csvs_df(args.training_data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Train model
    train_model(args.reg_rate, X_train, X_test, y_train, y_test)


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


def split_data(df):
    """Split the dataframe into features (X) and target (y)."""
    X = df[['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness',
            'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age']].values
    y = df['Diabetic'].values
    print(np.unique(y, return_counts=True))
    return train_test_split(X, y, test_size=0.30, random_state=0)


def train_model(reg_rate, X_train, X_test, y_train, y_test):
    """Train a logistic regression model with the given data and regularization rate."""
    model = LogisticRegression(C=1 / reg_rate, solver="liblinear")
    model.fit(X_train, y_train)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--training_data", dest='training_data', type=str, required=True)
    parser.add_argument("--reg_rate", dest='reg_rate', type=float, default=0.01)

    # Parse args
    return parser.parse_args()


# Run script
if __name__ == "__main__":
    # Add space in logs
    print("\n\n")
    print("*" * 60)

    # Parse args
    args = parse_args()

    # Run main function
    main(args)

    # Add space in logs
    print("*" * 60)
    print("\n\n")
