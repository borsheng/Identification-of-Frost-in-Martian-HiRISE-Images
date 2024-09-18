# Identification of Frost in Martian HiRISE Images

## Project Overview

This project focuses on building a classifier to distinguish images of Martian terrain that contain frost using **Convolutional Neural Networks (CNN)** and **Transfer Learning**. The dataset, which contains images from NASA's HiRISE camera on Mars, is used to study Mars' seasonal frost cycle and its role in climate and surface evolution. The project implements two key approaches:
1. Training a CNN from scratch.
2. Using Transfer Learning with pre-trained models (EfficientNetB0, ResNet50, and VGG16).

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Task 1: CNN + MLP Model](#task-1-cnn--mlp-model)
- [Task 2: Transfer Learning](#task-2-transfer-learning)
- [How to Run](#how-to-run)
- [Results](#results)
- [License](#license)

## Dataset

The dataset contains images of Martian terrain with annotations specifying whether the tile contains frost or is part of the background. The dataset consists of 214 subframes and 119,920 image tiles. Each tile has dimensions of **299x299 pixels** and is labeled as either `frost` or `background`. The dataset is divided into **train**, **test**, and **validation** sets.

- **Source**: [Mars HiRISE Frost Dataset](https://dataverse.jpl.nasa.gov/dataset.xhtml?persistentId=doi:10.48577/jpl.QJ9PYA)
- **Classes**: Binary classification - `frost` or `background`.

## Task 1: CNN + MLP Model

### (a) Data Augmentation and Preprocessing
- Empirical regularization techniques such as cropping, random zooming, rotating, flipping, contrast adjustments, and translations were applied to augment the images in the training set. **OpenCV** was used for image augmentation.

### (b) CNN + MLP Model
- A three-layer **CNN** followed by a **dense layer** (MLP) was trained on the dataset.
  - **Layers**: Each CNN layer uses ReLU activations and is followed by batch normalization and dropout.
  - **Regularization**: Dropout rate of 30% and L2 regularization were applied to avoid overfitting.
  - **Optimizer**: The **ADAM** optimizer was used with **cross-entropy loss**.
  - **Training**: The model was trained for at least 20 epochs with early stopping to prevent overfitting. The network parameters with the lowest validation error were saved.
  
### (c) Evaluation Metrics
- The model's performance was evaluated using **Precision**, **Recall**, and **F1 Score**.

## Task 2: Transfer Learning

### (a) Pre-trained Models
- **Transfer learning** was applied using the following pre-trained models:
  - **EfficientNetB0**
  - **ResNet50**
  - **VGG16**
  
  For each model:
  - The final fully connected layer was replaced and trained, while all preceding layers were frozen.
  - The outputs of the penultimate layer of each model were used as feature vectors.

### (b) Data Augmentation and Training
- Similar data augmentation techniques were applied during training.
- **ReLU** activation functions, **softmax** layers, **batch normalization**, and a **dropout rate of 30%** were used, along with the **ADAM** optimizer and cross-entropy loss.
- Models were trained for 10-20 epochs with early stopping, and the network parameters with the lowest validation error were saved.

### (c) Evaluation and Comparison
- The performance of the transfer learning models was compared to the CNN + MLP model using **Precision**, **Recall**, and **F1 Score**.
- **Training and Validation Errors** were plotted across epochs for each model to analyze their learning curves.

## How to Run

### Requirements

- Python 3.x
- Jupyter Notebook
- Required Python libraries (can be installed via `pip`):
  - `keras`
  - `tensorflow`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `opencv-python`
  - `scikit-learn`

### Instructions

1. Clone the repository and navigate to the project folder.
2. Open the Jupyter Notebooks:
   - `dataPreprocessingNotebook.ipynb` for data augmentation and preprocessing steps.
   - `Huang_Bor-Sheng.ipynb` for CNN and Transfer Learning model training.
3. Run the notebook cells to execute the tasks and view the results.

## Results

- **CNN + MLP Model**: The model achieved the following performance metrics:
  - **Precision**: [Insert Precision]
  - **Recall**: [Insert Recall]
  - **F1 Score**: [Insert F1 Score]

- **Transfer Learning**:
  - **EfficientNetB0**:
    - Precision: [Insert Precision]
    - Recall: [Insert Recall]
    - F1 Score: [Insert F1 Score]
  - **ResNet50**:
    - Precision: [Insert Precision]
    - Recall: [Insert Recall]
    - F1 Score: [Insert F1 Score]
  - **VGG16**:
    - Precision: [Insert Precision]
    - Recall: [Insert Recall]
    - F1 Score: [Insert F1 Score]

## License

This project is intended for academic purposes and is based on the DSCI 552 course material.

