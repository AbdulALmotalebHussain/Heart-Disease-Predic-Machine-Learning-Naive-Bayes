


# Heart Disease Prediction Using Naive Bayes

This project involves building a machine learning model to predict heart disease using the Naive Bayes algorithm. The dataset used is the UCI Heart Disease dataset.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Libraries Used](#libraries-used)
- [Data Exploration and Cleaning](#data-exploration-and-cleaning)
- [Model Building and Evaluation](#model-building-and-evaluation)
- [Results](#results)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Introduction

This project aims to predict the presence of heart disease in patients using machine learning techniques. The Naive Bayes classifier is employed for this purpose due to its simplicity and effectiveness in classification problems.

## Dataset

The UCI Heart Disease dataset is used for this project. The dataset contains 14 attributes related to heart disease diagnostics.

## Libraries Used

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Missingno

## Data Exploration and Cleaning

The dataset is first loaded and explored to understand the distributions and relationships between variables. Data cleaning involves handling missing values and encoding categorical variables.

```python
import pandas as pd

# Load the dataset
file_path = '/kaggle/input/heart-disease-data/heart_disease_uci.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe and some basic statistics
data_head = data.head()
data_description = data.describe(include='all')
data_info = data.info()

print(data_head, data_description, data_info)
```

## Model Building and Evaluation

The Naive Bayes classifier is trained on the dataset, and various performance metrics are used to evaluate the model, including accuracy, precision, recall, and F1-score.

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# Split the data into training and testing sets
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Make predictions
y_pred = nb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(report)
```

## Results

The performance of the Naive Bayes classifier is summarized with key metrics. The results indicate the model's effectiveness in predicting heart disease.

## Usage

To run this project, you need to have the necessary libraries installed. You can install them using the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn missingno
```

Load the dataset and execute the code cells in the provided Jupyter notebook.

## Conclusion

This project demonstrates how to use the Naive Bayes algorithm to predict heart disease. The results show that the model performs reasonably well, making it a useful tool for initial diagnostic purposes.
