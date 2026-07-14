"""
visualization.py

Visualization utilities for the Satellite Land Cover
Classification project.

Author: Akshara Bharath
"""

from pathlib import Path

import matplotlib.pyplot as plt


def show_image(
    image,
    title=None,
):
    """
    Display an image.

    Parameters
    ----------
    image : PIL.Image.Image

    title : str, optional
    """

    plt.figure(figsize=(5, 5))

    plt.imshow(image)

    plt.axis("off")

    if title is not None:
        plt.title(title)

    plt.show()


def plot_probabilities(
    probabilities,
):
    """
    Plot prediction probabilities.

    Parameters
    ----------
    probabilities : pandas.DataFrame
    """

    plt.figure(figsize=(8, 4))

    plt.bar(

        probabilities["Class"],

        probabilities["Probability"]

    )

    plt.ylabel("Probability")

    plt.xlabel("Land Cover Class")

    plt.title("Prediction Probabilities")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()


def plot_confusion_matrix(
    matrix,
    class_names,
):
    """
    Plot a confusion matrix.

    Parameters
    ----------
    matrix : numpy.ndarray

    class_names : list
    """

    plt.figure(figsize=(7, 6))

    plt.imshow(matrix)

    plt.colorbar()

    plt.xticks(

        range(len(class_names)),

        class_names,

        rotation=45

    )

    plt.yticks(

        range(len(class_names)),

        class_names

    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.show()


def save_figure(
    filename,
    dpi=300,
):
    """
    Save the current matplotlib figure.

    Parameters
    ----------
    filename : str

    dpi : int
    """

    output = Path(filename)

    output.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    plt.savefig(

        output,

        dpi=dpi,

        bbox_inches="tight"

    )