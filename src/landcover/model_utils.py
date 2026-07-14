"""
model_utils.py

Utilities for saving and loading trained machine
learning models.

Author: Akshara Bharath
"""

from pathlib import Path
from datetime import datetime
import json

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIRECTORY = PROJECT_ROOT / "models"

MODEL_FILENAME = "best_landcover_model.pkl"


def get_model_path():

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    return MODEL_DIRECTORY / MODEL_FILENAME


def save_model(
    model,
    model_path=None,
):

    if model_path is None:
        model_path = get_model_path()
    else:
        model_path = Path(model_path)

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    joblib.dump(
        model,
        model_path,
    )

    return model_path


def load_model(
    model_path=None,
):

    if model_path is None:
        model_path = get_model_path()
    else:
        model_path = Path(model_path)

    if not model_path.exists():

        raise FileNotFoundError(

            f"Model not found:\n{model_path}\n\n"

            "Run Notebook 03 to train and save the model."

        )

    return joblib.load(model_path)


def model_exists(
    model_path=None,
):

    if model_path is None:
        model_path = get_model_path()
    else:
        model_path = Path(model_path)

    return model_path.exists()


def save_dataset_split(
    X_train,
    X_test,
    y_train,
    y_test,
    output_directory=None,
):

    if output_directory is None:
        output_directory = MODEL_DIRECTORY
    else:
        output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_train.to_csv(
        output_directory / "X_train.csv",
        index=False,
    )

    X_test.to_csv(
        output_directory / "X_test.csv",
        index=False,
    )

    y_train.to_csv(
        output_directory / "y_train.csv",
        index=False,
    )

    y_test.to_csv(
        output_directory / "y_test.csv",
        index=False,
    )


def load_dataset_split(
    output_directory=None,
):

    if output_directory is None:
        output_directory = MODEL_DIRECTORY
    else:
        output_directory = Path(output_directory)

    X_train = pd.read_csv(
        output_directory / "X_train.csv"
    )

    X_test = pd.read_csv(
        output_directory / "X_test.csv"
    )

    y_train = pd.read_csv(
        output_directory / "y_train.csv"
    ).squeeze("columns")

    y_test = pd.read_csv(
        output_directory / "y_test.csv"
    ).squeeze("columns")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def save_training_metadata(
    best_model,
    cv_accuracy,
    k,
    cv_folds,
):

    metadata = {

        "best_model": best_model,

        "cv_accuracy": float(cv_accuracy),

        "k_neighbors": k,

        "cross_validation_folds": cv_folds,

        "training_date": datetime.now().strftime("%Y-%m-%d"),

    }

    metadata_path = MODEL_DIRECTORY / "training_metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )