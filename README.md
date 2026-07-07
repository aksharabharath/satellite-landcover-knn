# Satellite Land Cover Classification Using KNN

## Overview

This project uses satellite imagery from the EuroSAT dataset to classify land cover types using a machine learning approach.

The goal was to build a beginner-friendly image classification pipeline by extracting numerical image features and training a K-Nearest Neighbors (KNN) classifier.

The model classifies images into three categories:

- Forest
- River
- Residential

---

## Dataset

This project uses the EuroSAT RGB dataset.

Download it from:
https://github.com/phelber/EuroSAT

After downloading, place the images here:

data/
└── raw/
    └── eurosat_subset/
        ├── Forest/
        ├── River/
        └── Residential/

--

## Project Workflow

The project follows a complete machine learning pipeline:

1. Explore satellite image data
2. Extract RGB-based image features
3. Create a processed dataset
4. Train a KNN classifier
5. Evaluate model performance
6. Optimize model parameters
7. Test predictions on new images

---

## Feature Engineering

Instead of using raw images directly, each image was converted into numerical features:

- Mean Red value
- Mean Green value
- Mean Blue value
- Red channel variation
- Green channel variation
- Blue channel variation
- Average brightness

---

## Machine Learning Model

Model:
- K-Nearest Neighbors (KNN)

Best parameter:
- K = 5

Dataset split:
- 80% training
- 20% testing

---

## Results

The final model achieved:

Accuracy: 96.76%

Performance was strongest for Forest classification, while River classification had slightly more overlap due to mixed landscape features.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Pillow

---

## Future Improvements

Possible improvements include:

- Using full pixel-level information instead of summarized RGB features
- Testing more advanced models
- Applying convolutional neural networks (CNNs)
- Using additional satellite bands
