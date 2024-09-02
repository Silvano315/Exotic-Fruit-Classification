# Exotic-Fruit-Classification

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Preprocessing and EDA](#preprocessing-and-eda)
4. [Machine Learning with KNN](#machine-learning-with-knn)
5. [Results](#results)
6. [Extra: Visualization with Pygwalker](#extra-visualization-with-pygwalker)
7. [Requirements](#requirements)


## Introduction

This repository is the first project of the master's degree in AI Engineering with [Profession AI](https://profession.ai), all the credits for the requests and idea go to this team.

TropicTaste Inc., a leading exotic fruit distributor, is looking to improve the efficiency and accuracy of the fruit classification process. The **goal** is to develop a machine learning model that can predict the type of fruit based on numerical features. The current exotic fruit classification process is manual and error-prone, making it inefficient and resource-intensive. The need for an automated and accurate system is crucial to streamline business operations and maintain high quality standards.

By implementing an automated classification model, TropicTaste Inc. will be able to:
- Improve Operational Efficiency: Automating classification will reduce the time and resources needed, increasing productivity.
- Reduce Human Error: A machine learning model will minimize classification errors, ensuring greater accuracy.
- Optimize Inventory: Accurate classification will allow for better inventory management, ensuring optimal storage conditions for each type of fruit.
- Increase Customer Satisfaction: Correct identification and classification of fruits will help maintain high quality standards, improving customer satisfaction.

Project Requirements:
1. Loading and preprocessing of exotic fruit data.
2. Handling of any missing values, normalization and scaling of data.
3. Development and training of the KNN model.
4. Optimization of parameters to improve predictive accuracy.
5. Use of cross-validation techniques to evaluate the generalization ability of the model.
6. Calculation of performance metrics, such as classification accuracy and error.
7. Create graphs to visualize and compare the model performance.
8. Analyze and interpret the results to identify areas for improvement.

To complete all the project requirements, I have created a single [notebook](Project_Notebook.ipynb) where all operations are performed and results are visualized. Additionally, I have organized the code into a [`src`](src/) folder containing Python files with the methods used in the notebook.


## Dataset

This dataset contains the following variables:
- **Fruit**: The type of fruit. This is the target variable that we want to predict.
- **Weight** (g): The weight of the fruit in grams. Continuous variable.
- **Average diameter** (mm): The average diameter of the fruit in millimeters. Continuous variable.
- **Average length** (mm): The average length of the fruit in millimeters. Continuous variable.
- **Peel hardness** (1-10): The hardness of the fruit's peel on a scale of 1 to 10. Continuous variable.
- **Sweetness** (1-10): The sweetness of the fruit on a scale of 1 to 10. Continuous variable.
- **Acidity** (1-10): The acidity of the fruit on a scale of 1 to 10. Continuous variable.

In folder [`Data`](Data/), you can find both the original dataset and the transformed dataset after preprocessing steps.


## Preprocessing and EDA

In the [`data_engineering.py`](src/preprocessing_and_viz.py) file, I have gathered all methods for managing dataset preparation and Exploratory Data Analysis (EDA), including:

- **Analysis of Statistical Information**: An initial analysis of the dataset's statistics revealed that the feature *Peel hardness* does not have a maximum value of 10, as suggested by the project guidelines, but instead, it reaches a value of 13.72.

- **Preprocessing with Duplicate and Missing Values Check**: I searched for any duplicates and missing values in the dataset, but none were found. The dataset remains with a shape of (500, 6).

- **Data Visualization Before Feature Engineering**: Interactive plots were generated using the `Plotly` package to visualize the data. These visualizations provided crucial insights into the relationships between features. It is highly recommended to explore these plots directly in the notebook to gain an interactive, step-by-step understanding, as each plot is accompanied by descriptive markdown comments.

- **Implementation of Feature Scaling Solutions**: Two feature scaling methods were implemented to transform the continuous features:
  1. **MinMaxScaler**: This method scales the features into a fixed range, which can lead to a loss of significant information.
  2. **StandardScaler**: This method standardizes features by removing the mean and scaling to unit variance. This was chosen as the final approach due to its statistical robustness, especially for distributions that, with an increasing number of data points, are expected to approach a normal distribution.

- **Second Visualization with Transformed Data**: To validate the chosen preprocessing steps, I visualized the transformed dataset using several types of Plotly plots: Distplot with Normal Distribution, Distribution with Histogram and Box Plot, Scatter Plots with Target Column Comparison.

These steps provide a comprehensive understanding of the dataset and ensure that the data is well-prepared for the subsequent machine learning modeling phase.


## Machine Learning with KNN

In the [`models.py`](src/models.py) file, I have implemented methods for training and evaluating KNN models:


## Results

The detailed description of the results, including both the Exploratory Data Analysis (EDA) and model performance evaluations, can be found in the markdown cells of the [notebook](Project_Notebook.ipynb). It is highly recommended to view the results directly within the notebook, as interactive `Plotly` graphs have been utilized for a more dynamic exploration of the data.

### Key Findings

## Extra: Visualization with Pygwalker

## Requirements

To run this project, I used a 3.11.x Python version. You need to installed the packages in the [requirements](requirements.txt):

```bash
pip install -r requirements.txt