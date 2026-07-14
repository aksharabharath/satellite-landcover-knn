"""
LandCover

A machine learning package for satellite land cover
classification using satellite imagery.

Author: Akshara Bharath
"""

__version__ = "1.0.0"

from .dataset import (
    load_dataset,
    dataset_summary,
    prepare_features,
    split_dataset,
    get_feature_names,
    dataset_info,
)

from .feature_extraction import (
    extract_features,
    features_to_dataframe,
    image_to_dataframe,
)

from .training import (
    create_models,
    evaluate_models,
    train_best_model,
    train_pipeline,
)

from .evaluation import (
    calculate_accuracy,
    generate_classification_report,
    generate_confusion_matrix,
    prediction_summary,
    evaluate_model,
)

from .model_utils import (
    get_model_path,
    save_model,
    load_model,
    model_exists,
    save_dataset_split,
    load_dataset_split,
    save_training_metadata,
)
from .prediction import (
    load_image,
    get_random_image,
    predict_image,
    print_prediction,
)

from .visualization import (
    show_image,
    plot_probabilities,
    plot_confusion_matrix,
    save_figure,
)

from .prediction import (
    load_image,
    get_random_image,
    predict_image,
    predict_probabilities,
    print_prediction,
)