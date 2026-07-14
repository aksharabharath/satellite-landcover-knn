"""
prediction.py

Utilities for making predictions on new satellite images.

Author: Akshara Bharath
"""

import random
from pathlib import Path

import pandas as pd


from PIL import Image

from .feature_extraction import (
    image_to_dataframe,
)

from .model_utils import (
    load_model,
)



def load_image(image_path):
    """
    Load an image from disk.

    Parameters
    ----------
    image_path : str or pathlib.Path

    Returns
    -------
    PIL.Image.Image
    """

    image_path = Path(image_path)

    if not image_path.exists():

        raise FileNotFoundError(

            f"Image not found:\n{image_path}"

        )

    return Image.open(image_path).convert("RGB")


def predict_image(
    image,
    model=None,
):
    """
    Predict the land cover class of an image.

    Parameters
    ----------
    image : PIL.Image.Image

    model : sklearn estimator, optional

    Returns
    -------
    str
    """

    if model is None:
        model = load_model()

    features = image_to_dataframe(image)

    prediction = model.predict(features)[0]

    return prediction


def predict_probabilities(
    image,
    model=None,
):
    """
    Predict class probabilities.

    Parameters
    ----------
    image : PIL.Image.Image

    model : sklearn estimator, optional

    Returns
    -------
    pandas.DataFrame
    """

    if model is None:
        model = load_model()

    if not hasattr(model, "predict_proba"):

        raise AttributeError(

            "This model does not support probability predictions."

        )

    features = image_to_dataframe(image)

    probabilities = model.predict_proba(features)[0]

    return pd.DataFrame({

        "Class": model.classes_,

        "Probability": probabilities,

    }).sort_values(

        "Probability",

        ascending=False,

    ).reset_index(drop=True)


def predict_from_path(
    image_path,
    model=None,
):
    """
    Predict directly from an image path.

    Parameters
    ----------
    image_path : str or pathlib.Path

    model : sklearn estimator, optional

    Returns
    -------
    str
    """

    image = load_image(image_path)

    return predict_image(
        image,
        model=model,
    )


def print_prediction(
    prediction,
):
    """
    Print a formatted prediction.

    Parameters
    ----------
    prediction : str
    """

    print("=" * 60)

    print("LAND COVER PREDICTION")

    print("=" * 60)

    print(f"Predicted Class : {prediction}")

    print("=" * 60)

def get_random_image(
    class_name=None,
    image_directory="../data/raw/eurosat_subset",
):
    """
    Select a random image from the dataset.

    Parameters
    ----------
    class_name : str or None
        If provided, selects a random image from that class.
        Otherwise selects a random class.

    image_directory : str or Path

    Returns
    -------
    pathlib.Path
        Path to the selected image.
    """

    image_directory = Path(image_directory)

    if class_name is None:

        classes = [
            folder for folder in image_directory.iterdir()
            if folder.is_dir()
        ]

        selected_class = random.choice(classes)

    else:

        selected_class = image_directory / class_name

        if not selected_class.exists():
            raise ValueError(f"Unknown class: {class_name}")

    images = list(selected_class.glob("*"))

    if len(images) == 0:
        raise FileNotFoundError(
            f"No images found in {selected_class}"
        )

    return random.choice(images)

def predict_probabilities(
    image,
    model,
):
    """
    Predict class probabilities for a single image.

    Parameters
    ----------
    image : PIL.Image

    model : sklearn estimator

    Returns
    -------
    pandas.DataFrame
    """

    import pandas as pd

    features = image_to_dataframe(image)

    probabilities = model.predict_proba(features)[0]

    return pd.DataFrame({
        "Class": model.classes_,
        "Probability": probabilities,
    }).sort_values(
        "Probability",
        ascending=False,
    ).reset_index(drop=True)