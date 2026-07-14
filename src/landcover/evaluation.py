"""
evaluation.py

Evaluate trained machine learning models for the
Satellite Land Cover Classification project.

Author: Akshara Bharath
"""

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def calculate_accuracy(
    model,
    X_test,
    y_test,
):
    """
    Calculate test accuracy.

    Parameters
    ----------
    model : sklearn estimator

    X_test : pandas.DataFrame

    y_test : pandas.Series

    Returns
    -------
    float
    """

    predictions = model.predict(X_test)

    return accuracy_score(
        y_test,
        predictions,
    )


def generate_classification_report(
    model,
    X_test,
    y_test,
):
    """
    Generate a classification report.

    Parameters
    ----------
    model : sklearn estimator

    X_test : pandas.DataFrame

    y_test : pandas.Series

    Returns
    -------
    pandas.DataFrame
    """

    predictions = model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
    )

    return pd.DataFrame(report).transpose()


def generate_confusion_matrix(
    model,
    X_test,
    y_test,
):
    """
    Generate a confusion matrix.

    Parameters
    ----------
    model : sklearn estimator

    X_test : pandas.DataFrame

    y_test : pandas.Series

    Returns
    -------
    numpy.ndarray
    """

    predictions = model.predict(X_test)

    return confusion_matrix(
        y_test,
        predictions,
    )


def prediction_summary(
    model,
    X_test,
):
    """
    Return model predictions.

    Parameters
    ----------
    model : sklearn estimator

    X_test : pandas.DataFrame

    Returns
    -------
    pandas.Series
    """

    predictions = model.predict(X_test)

    return pd.Series(
        predictions,
        name="Prediction",
    )


def evaluate_model(
    model,
    X_test,
    y_test,
):
    """
    Complete evaluation pipeline.

    Parameters
    ----------
    model : sklearn estimator

    X_test : pandas.DataFrame

    y_test : pandas.Series

    Returns
    -------
    dict
    """

    accuracy = calculate_accuracy(
        model,
        X_test,
        y_test,
    )

    report = generate_classification_report(
        model,
        X_test,
        y_test,
    )

    matrix = generate_confusion_matrix(
        model,
        X_test,
        y_test,
    )

    predictions = prediction_summary(
        model,
        X_test,
    )

    return {

        "accuracy": accuracy,

        "classification_report": report,

        "confusion_matrix": matrix,

        "predictions": predictions,

    }