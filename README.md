# Satellite Land Cover Classification using K-Nearest Neighbors (KNN)

A machine learning project that classifies satellite images into **Forest**, **River**, and **Residential** land cover types using engineered RGB image features and a K-Nearest Neighbors (KNN) classifier.

**Final Model Accuracy:** **96.76%**

---

# Project Overview

Satellite imagery is widely used for environmental monitoring, urban planning, agriculture, and disaster response. In this project, I built a complete machine learning pipeline that classifies satellite images into three land cover categories:

* Forest
* River
* Residential

Instead of using deep learning, this project focuses on understanding the fundamentals of machine learning by extracting meaningful numerical features from images and training a traditional KNN classifier.

---

# Dataset

This project uses a subset of the **EuroSAT RGB** satellite imagery dataset.

The raw dataset is **not included** in this repository to keep the project lightweight.

After downloading the dataset, place it in the following directory:

```text
data/
└── raw/
    └── eurosat_subset/
        ├── Forest/
        ├── River/
        └── Residential/
```

The processed feature dataset generated during preprocessing is included in:

```text
data/processed/satellite_landcover_features.csv
```

---

# Project Workflow

```text
Satellite Images
        │
        ▼
Data Exploration
        │
        ▼
Feature Extraction
(RGB Means, Standard Deviations, Brightness)
        │
        ▼
Processed Dataset
        │
        ▼
Train/Test Split
        │
        ▼
K-Nearest Neighbors (K = 5)
        │
        ▼
Model Evaluation
        │
        ▼
Predict New Satellite Images
```

---

# Feature Engineering

Each **64 × 64 RGB satellite image** was converted into numerical features:

* Mean Red value
* Mean Green value
* Mean Blue value
* Standard deviation of the Red channel
* Standard deviation of the Green channel
* Standard deviation of the Blue channel
* Overall image brightness

These engineered features transformed each image into a numerical representation suitable for machine learning.

---

# Machine Learning Model

## Algorithm

* K-Nearest Neighbors (KNN)

## Training/Test Split

* 80% Training
* 20% Testing

## Hyperparameter Testing

The following values of **K** were evaluated:

* K = 1
* K = 3
* K = 5
* K = 7
* K = 9
* K = 11

The highest accuracy was achieved with **K = 5**.

---

# Results

**Overall Accuracy:** **96.76%**

| Class       | Precision | Recall | F1 Score |
| ----------- | --------: | -----: | -------: |
| Forest      |      0.99 |   1.00 |     0.99 |
| Residential |      0.94 |   0.98 |     0.96 |
| River       |      0.97 |   0.91 |     0.94 |

Feature visualization showed that Forest images formed a distinct cluster, while River images partially overlapped with both Forest and Residential classes due to mixed landscape characteristics.

---

# Technologies Used

* Python
* NumPy
* Pandas
* Pillow (PIL)
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook
* Git
* GitHub

---

# Repository Structure

```text
satellite-landcover-knn/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_analysis.ipynb
│
├── README.md
└── requirements.txt
```

---

# Future Improvements

Potential extensions to this project include:

* Expanding the model to classify additional EuroSAT land cover categories.
* Comparing KNN with Decision Trees, Random Forests, and Support Vector Machines.
* Training a Convolutional Neural Network (CNN) using raw image pixels.
* Exploring FAISS for scalable nearest-neighbor search on large image datasets.
* Deploying the model as an interactive web application.

---

# Key Takeaways

This project demonstrates a complete machine learning workflow, including:

* Data exploration
* Feature engineering
* Model training
* Hyperparameter tuning
* Performance evaluation
* Prediction on unseen data
* Organizing a reproducible machine learning project using Git and GitHub
