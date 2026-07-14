"""
dataset.py

Utility functions for loading and preparing the processed
satellite land cover dataset.

Author: Akshara Bharath
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split



DATASET_PATH = Path(
    "../data/processed/satellite_landcover_features.csv"
)



FEATURE_NAMES = [

    "Mean_R",

    "Mean_G",

    "Mean_B",

    "Std_R",

    "Std_G",

    "Std_B",

    "Brightness",

]



def load_dataset(csv_path=DATASET_PATH):
    """
    Load the processed feature dataset.

    Parameters
    ----------
    csv_path : str or pathlib.Path

    Returns
    -------
    pandas.DataFrame
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():

        raise FileNotFoundError(

            f"\nDataset not found:\n{csv_path}\n\n"

            "Run Notebook 02 first."

        )

    return pd.read_csv(csv_path)



def dataset_summary(df):
    """
    Print a summary of the dataset.

    Parameters
    ----------
    df : pandas.DataFrame
    """

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    print()

    print("Feature Columns")

    for feature in FEATURE_NAMES:

        print(f"  • {feature}")

    print()

    print("Class Distribution")

    print(df["Label"].value_counts())

    print()

    print("=" * 60)



def get_feature_names():
    """
    Return the feature column names.

    Returns
    -------
    list
    """

    return FEATURE_NAMES.copy()



def prepare_features(
    df,
    target_column="Label",
):
    """
    Separate features and labels.

    Parameters
    ----------
    df : pandas.DataFrame

    target_column : str

    Returns
    -------
    X : pandas.DataFrame

    y : pandas.Series
    """

    X = df[get_feature_names()]

    y = df[target_column]

    return X, y



def split_dataset(
    X,
    y,
    test_size=0.20,
    random_state=42,
):
    """
    Split the dataset into training and testing sets.

    Stratification preserves the class distribution.

    Parameters
    ----------
    X : pandas.DataFrame

    y : pandas.Series

    test_size : float

    random_state : int

    Returns
    -------
    X_train

    X_test

    y_train

    y_test
    """

    return train_test_split(

        X,

        y,

        test_size=test_size,

        random_state=random_state,

        stratify=y,

    )



def dataset_info(df):
    """
    Return useful information about the dataset.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    dict
    """

    return {

        "rows": len(df),

        "columns": len(df.columns),

        "classes": sorted(df["Label"].unique()),

        "num_classes": df["Label"].nunique(),

        "features": get_feature_names(),

    }