"""
training.py

Train and compare machine learning models for the
Satellite Land Cover Classification project.

Author: Akshara Bharath
"""

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def create_models(k=5):
    """
    Create candidate machine learning models.
    """

    return {

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=5000,
                random_state=42,
            ))
        ]),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=k,
            ))
        ])

    }


def evaluate_models(
    models,
    X_train,
    y_train,
    cv=5,
):
    """
    Evaluate each model using cross-validation.
    """

    rows = []

    for name, model in models.items():

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
        )

        rows.append({

            "Model": name,

            "CV Accuracy": scores.mean(),

            "CV Std": scores.std(),

        })

    results = pd.DataFrame(rows)

    results = results.sort_values(
        by=["CV Accuracy", "CV Std"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return results


def train_best_model(
    models,
    best_model_name,
    X_train,
    y_train,
):
    """
    Train the best model on the full training dataset.
    """

    model = models[best_model_name]

    model.fit(
        X_train,
        y_train,
    )

    return model


def train_pipeline(
    X_train,
    y_train,
    k=5,
    cv=5,
):
    """
    Complete model training pipeline.
    """

    models = create_models(k=k)

    results = evaluate_models(
        models,
        X_train,
        y_train,
        cv=cv,
    )

    best_model_name = results.iloc[0]["Model"]

    best_model = train_best_model(
        models,
        best_model_name,
        X_train,
        y_train,
    )

    return (
        best_model_name,
        best_model,
        results,
    )