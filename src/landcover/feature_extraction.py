"""
feature_extraction.py

Feature extraction utilities for the Satellite Land Cover
Classification project.

This module converts satellite images into numerical
features that can be used by machine learning models.

Author: Akshara Bharath
"""

from PIL import Image
import numpy as np
import pandas as pd


def validate_image(image):
    """
    Ensure the image is RGB.

    Parameters
    ----------
    image : PIL.Image.Image

    Returns
    -------
    PIL.Image.Image
    """

    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image.")

    return image.convert("RGB")



def extract_features(image):
    """
    Extract handcrafted RGB features from an image.

    Features
    --------
    Mean_R
    Mean_G
    Mean_B
    Std_R
    Std_G
    Std_B
    Brightness

    Parameters
    ----------
    image : PIL.Image.Image

    Returns
    -------
    dict
    """

    image = validate_image(image)

    img = np.asarray(image).astype(np.float32)

    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    features = {

        "Mean_R": red.mean(),

        "Mean_G": green.mean(),

        "Mean_B": blue.mean(),

        "Std_R": red.std(),

        "Std_G": green.std(),

        "Std_B": blue.std(),

        "Brightness": img.mean(),

    }

    return features



def features_to_dataframe(features):
    """
    Convert a feature dictionary into a one-row DataFrame.

    Parameters
    ----------
    features : dict

    Returns
    -------
    pandas.DataFrame
    """

    return pd.DataFrame([features])


def image_to_dataframe(image):
    """
    Convert an image directly into a one-row DataFrame.

    Parameters
    ----------
    image : PIL.Image.Image

    Returns
    -------
    pandas.DataFrame
    """

    features = extract_features(image)

    return features_to_dataframe(features)