import pandas as pd
from sklearn.model_selection import train_test_split
from .flate import *


def features_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    exist_features = [col for col in COLUMN_FEATURES if col in df.columns]

    X = df[exist_features]
    y = df[COLUMN_TARGET]

    print(f" [Données] Features sélectionnées : {exist_features} | Cible : {COLUMN_TARGET}")
    return X, y


def train_test_process(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    print(f" [Découpage] Train set : {X_train.shape} | Test set : {X_test.shape}")
    return X_train, X_test, y_train, y_test