import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.constants import RANDOM_SEED


# Function to split a dataframe with train_test_split
def data_split(df, target, test_size = 0.3):

    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED)
    
    return X_train, X_test, y_train, y_test


# Function to train K-Nearest Neighbors (KNN) model
def train_KNN(X_train, y_train, num_neighbors = 3):

    knn = KNeighborsClassifier(n_neighbors = num_neighbors)
    knn.fit(X_train, y_train)

    return knn


#
def knn_optimization(X_train, y_train, params=None, k_cv = 5, save_res = False):

    if params is None:
        params = {
            'n_neighbors' : np.arange(1,20)
        }
    
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, params, 
                               cv=k_cv, scoring='accuracy',
                               return_train_score=True)
    grid_search.fit(X_train, y_train)

    print("="*50)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best model: {grid_search.best_estimator_}")
    print(f"Best accuracy score: {grid_search.best_score_}")
    print("="*50)

    results_df = pd.DataFrame(grid_search.cv_results_)

    if save_res:
        os.makedirs("Results", exist_ok=True)
        results_df.to_csv("Results/results_gridsearch_cv_knn.csv", index=False)
        print("\nResults saved...")

    return grid_search.best_estimator_, results_df


# Function to visualize results from GridSearchCV:
def viz_grid_search_res(results):

    plt.figure(figsize=(10,6))
    plt.errorbar(results['param_n_neighbors'], 
                 results['mean_test_score'], yerr = results['std_test_score'], 
                 label='Mean Test Score', capsize=5, 
                 fmt='o', linestyle = 'solid', linewidth = 1,
                 color = 'navy', ecolor='royalblue')
    plt.errorbar(results['param_n_neighbors'], 
                 results['mean_train_score'], yerr = results['std_train_score'], 
                 label='Mean Train Score', capsize=5,
                 fmt='o', linestyle = 'solid', linewidth = 1,
                 color = 'darkorange', ecolor='salmon')

    plt.xticks(np.arange(1,20,1))
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy Score')
    plt.title('GridSearchCV Results for KNN')
    plt.legend()
    plt.grid()
    plt.show()


# Function to evaluate model from GridSearchCV:
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    print("="*70)
    print(f"Accuracy: {acc:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("="*70)
    print("Confusion Matrix:")
    #percentage_matrix = (confusion_matrix(y_pred, y_test)/confusion_matrix(y_pred, y_test).sum().sum())
    plt.figure(figsize = (8,6))
    sns.heatmap(confusion_matrix(y_pred, y_test), annot=True,
                fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Apple', 'Banana', 'Grape', 'Kiwi', 'Orange'],
                yticklabels=['Apple', 'Banana', 'Grape', 'Kiwi', 'Orange'])
    plt.xlabel('Predicted', labelpad=12)
    plt.ylabel('Actual', labelpad=12)
    plt.title('Confusion Matrix')
    plt.show()