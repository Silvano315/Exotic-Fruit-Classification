import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from typing import Tuple, Dict, Any, Optional, Literal

from pandas import DataFrame, Series, melt
from numpy import arange, mean, std
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from seaborn import set_theme, heatmap
from plotly import graph_objects as go
from plotly.express import violin
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             precision_score, recall_score, f1_score, log_loss)

from src.constants import RANDOM_SEED


def data_split(df: DataFrame, target: str, test_size: float = 0.3) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Splits a DataFrame into training and testing sets.

    Parameters:
    df (DataFrame): The DataFrame to split.
    target (str): The name of the target column.
    test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.3.

    Returns:
    Tuple[DataFrame, DataFrame, DataFrame, DataFrame]: The training features, testing features, training target, and testing target.
    
    Raises:
    ValueError: If the target column is not found in the DataFrame or if the test_size is invalid.
    """
    try:
        X = df.drop(columns=[target])
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED)
        
        return X_train, X_test, y_train, y_test

    except KeyError:
        raise ValueError(f"The target column '{target}' was not found in the DataFrame.")
    except ValueError as e:
        raise ValueError(f"Invalid value for test_size or other parameters: {e}")


def knn_optimization(X_train: DataFrame, y_train: Series, params: Optional[dict] = None, k_cv: int = 5, save_res: bool = False) -> Tuple[KNeighborsClassifier, DataFrame]:
    """
    Trains a K-Nearest Neighbors (KNN) model and optimizes it using GridSearchCV.

    Parameters:
    X_train (DataFrame): The training features.
    y_train (Series): The training target variable.
    params (dict, optional): The hyperparameters for GridSearchCV. Defaults to None, which uses a default range for 'n_neighbors'.
    k_cv (int, optional): The number of cross-validation folds. Defaults to 5.
    save_res (bool, optional): Whether to save the results to a CSV file. Defaults to False.

    Returns:
    Tuple[KNeighborsClassifier, DataFrame]: The best KNN estimator and a DataFrame with GridSearchCV results.
    """
    try:
        if params is None:
            params = {
                'n_neighbors': arange(1, 20)
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

        results_df = DataFrame(grid_search.cv_results_)

        if save_res:
            os.makedirs("Results", exist_ok=True)
            results_df.to_csv("Results/results_gridsearch_cv_knn.csv", index=False)
            print("\nResults saved...")

        return grid_search.best_estimator_, results_df

    except Exception as e:
        print(f"Error during KNN optimization: {e}")
        return None, DataFrame() 
    

def viz_grid_search_res(results: DataFrame) -> Figure:
    """
    Visualizes the results from GridSearchCV for K-Nearest Neighbors (KNN) optimization and returns the figure.

    Parameters:
    results (DataFrame): A DataFrame containing the results from GridSearchCV with columns 
                         for 'param_n_neighbors', 'mean_test_score', 'std_test_score', 
                         'mean_train_score', and 'std_train_score'.
    
    Returns:
    Figure: The matplotlib figure object containing the plot.
    
    Raises:
    ValueError: If the necessary columns are not present in the DataFrame.
    """
    try:
        required_columns = ['param_n_neighbors', 'mean_test_score', 'std_test_score', 
                            'mean_train_score', 'std_train_score']
        if not all(col in results.columns for col in required_columns):
            raise ValueError("Results DataFrame must contain the following columns: "
                             "'param_n_neighbors', 'mean_test_score', 'std_test_score', "
                             "'mean_train_score', and 'std_train_score'.")

        fig = plt.figure(figsize=(10, 6))
        
        plt.errorbar(results['param_n_neighbors'], 
                 results['mean_test_score'], yerr=results['std_test_score'], 
                 label='Mean Test Score', capsize=5, 
                 fmt='o', linestyle='solid', linewidth=1,
                 color='navy', ecolor='royalblue')
        
        plt.errorbar(results['param_n_neighbors'], 
                 results['mean_train_score'], yerr=results['std_train_score'], 
                 label='Mean Train Score', capsize=5,
                 fmt='o', linestyle='solid', linewidth=1,
                 color='darkorange', ecolor='salmon')

        plt.xticks(range(1, 20))
        plt.xlabel('Number of Neighbors')
        plt.ylabel('Accuracy Score')
        plt.title('GridSearchCV Results for KNN')
        plt.legend()
        plt.grid(True)

        return fig

    except ValueError as e:
        print(f"Error: {e}")
        return plt.figure()


def evaluate_model(model, X_test, y_test) -> Tuple[Figure, Figure]:
    """
    Evaluates the given model using accuracy score, classification report, and confusion matrix.

    Parameters:
    model: The trained model to evaluate.
    X_test: The features of the test set.
    y_test: The true labels of the test set.
    
    Returns:
    Tuple[Figure, Figure]: A tuple containing the matplotlib Figure objects for the confusion matrix heatmap.
    """
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print("="*70)
    print(f"Accuracy: {acc:.2f}")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.figure(figsize=(8, 6)), None
    
    try:
        ax = fig.add_subplot(111)
        heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Apple', 'Banana', 'Grape', 'Kiwi', 'Orange'],
                    yticklabels=['Apple', 'Banana', 'Grape', 'Kiwi', 'Orange'], ax=ax)
        plt.xlabel('Predicted', labelpad=12)
        plt.ylabel('Actual', labelpad=12)
        plt.title('Confusion Matrix')
    
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")
    
    return fig, ax


def evaluate_knn_neighbors(df: DataFrame, target: str, min_neighbors: int = 1, max_neighbors: int = 20, cv: int = 5) -> DataFrame:
    """
    Evaluates the K-Nearest Neighbors (KNN) classifier across a range of neighbors using stratified k-fold cross-validation.

    Parameters:
    df (DataFrame): The dataset containing features and the target variable.
    target (str): The name of the target variable column in the DataFrame.
    min_neighbors (int): The minimum number of neighbors to test.
    max_neighbors (int): The maximum number of neighbors to test.
    cv (int): The number of cross-validation folds.

    Returns:
    DataFrame: A DataFrame containing the results of the evaluation, including mean and standard deviation 
               of train and test accuracy and log loss for each number of neighbors.
    """
    X = df.drop(columns=[target])
    y = df[target]
    
    results = {
        'n_neighbors': [],
        'train_accuracy_mean': [],
        'test_accuracy_mean': [],
        'train_log_loss_mean': [],
        'test_log_loss_mean': [],
        'train_accuracy_std': [],
        'test_accuracy_std': [],
        'train_log_loss_std': [],
        'test_log_loss_std': []
    }
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    
    for n_neighbors in range(min_neighbors, max_neighbors + 1):
        train_accuracies = []
        test_accuracies = []
        train_log_losses = []
        test_log_losses = []
        
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn.fit(X_train, y_train)
            y_train_pred = knn.predict(X_train)
            y_train_proba = knn.predict_proba(X_train)
            y_test_pred = knn.predict(X_test)
            y_test_proba = knn.predict_proba(X_test)
            
            train_accuracies.append(accuracy_score(y_train, y_train_pred))
            test_accuracies.append(accuracy_score(y_test, y_test_pred))
            train_log_losses.append(log_loss(y_train, y_train_proba))
            test_log_losses.append(log_loss(y_test, y_test_proba))
        
        results['n_neighbors'].append(n_neighbors)
        results['train_accuracy_mean'].append(mean(train_accuracies))
        results['test_accuracy_mean'].append(mean(test_accuracies))
        results['train_log_loss_mean'].append(mean(train_log_losses))
        results['test_log_loss_mean'].append(mean(test_log_losses))
        results['train_accuracy_std'].append(std(train_accuracies))
        results['test_accuracy_std'].append(std(test_accuracies))
        results['train_log_loss_std'].append(std(train_log_losses))
        results['test_log_loss_std'].append(std(test_log_losses))
        
        print("-" * 100)
        print(f"n_neighbors={n_neighbors}: Train Accuracy={mean(train_accuracies):.4f} "
              f"Test Accuracy={mean(test_accuracies):.4f} "
              f"Train Log Loss={mean(train_log_losses):.4f} "
              f"Test Log Loss={mean(test_log_losses):.4f}")
    
    results_df = DataFrame(results)
    
    return results_df


def plot_cv_results(df: DataFrame) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """
    Visualizes the results of cross-validation to find the best number of neighbors for KNN.
    Plots accuracy and log loss metrics over different numbers of neighbors.

    Parameters:
    df (pd.DataFrame): DataFrame containing cross-validation results with columns for number of neighbors,
                       train and test accuracy, and train and test log loss.

    Returns:
    Tuple[plt.Figure, plt.Axes, plt.Axes]: A tuple containing the matplotlib Figure object and the two Axes objects.
    """
    set_theme(style="whitegrid")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(df['n_neighbors'], df['train_accuracy_mean'], '-o', 
             label=f'Train Accuracy (Mean: {df["train_accuracy_mean"].mean():.2f}, Std: {df["train_accuracy_mean"].std():.2f})', 
             color='blue')
    ax1.plot(df['n_neighbors'], df['test_accuracy_mean'], '-o', 
             label=f'Test Accuracy (Mean: {df["test_accuracy_mean"].mean():.2f}, Std: {df["test_accuracy_mean"].std():.2f})', 
             color='green')
    ax1.set_xlabel('Number of Neighbors')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy over Different Numbers of Neighbors')
    ax1.set_ylim(0.8, 1)  
    ax1.set_xticks(df['n_neighbors']) 
    ax1.legend(loc='best', fontsize=10)

    ax2.plot(df['n_neighbors'], df['train_log_loss_mean'], '-o', 
             label=f'Train Log Loss (Mean: {df["train_log_loss_mean"].mean():.2f}, Std: {df["train_log_loss_mean"].std():.2f})', 
             color='red')
    ax2.plot(df['n_neighbors'], df['test_log_loss_mean'], '-o', 
             label=f'Test Log Loss (Mean: {df["test_log_loss_mean"].mean():.2f}, Std: {df["test_log_loss_mean"].std():.2f})', 
             color='orange')
    ax2.set_xlabel('Number of Neighbors')
    ax2.set_ylabel('Log Loss')
    ax2.set_title('Log Loss over Different Numbers of Neighbors')
    ax2.set_xticks(df['n_neighbors']) 
    ax2.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    return fig, ax1, ax2


def cross_validation_model(df: DataFrame, target: str, n_neighbors: int, cv: int = 5) -> DataFrame:
    """
    Perform cross-validation for a KNN model with a specified number of neighbors.

    Parameters:
    df (pd.DataFrame): DataFrame containing features and the target column.
    target (str): The name of the target column in the DataFrame.
    n_neighbors (int): The number of neighbors to use for the KNN classifier.
    cv (int): The number of folds for cross-validation (default is 5).

    Returns:
    pd.DataFrame: DataFrame with cross-validation metrics for each fold.
    """
    X = df.drop(columns=[target])
    y = df[target]

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)

    metrics: Dict[str, list] = {
        'fold': [],
        'train_accuracy': [], 'test_accuracy': [],
        'train_precision': [], 'test_precision': [],
        'train_recall': [], 'test_recall': [],
        'train_f1': [], 'test_f1': [],
        'train_loss': [], 'test_loss': []
    }

    print("=" * 40)
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        print(f"Training KNN in Fold {fold + 1}...")
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)

        y_train_pred = knn.predict(X_train)
        y_test_pred = knn.predict(X_test)
        y_train_proba = knn.predict_proba(X_train)
        y_test_proba = knn.predict_proba(X_test)

        metrics['fold'].append(fold + 1)
        metrics['train_accuracy'].append(accuracy_score(y_train, y_train_pred))
        metrics['test_accuracy'].append(accuracy_score(y_test, y_test_pred))
        metrics['train_precision'].append(precision_score(y_train, y_train_pred, average='weighted'))
        metrics['test_precision'].append(precision_score(y_test, y_test_pred, average='weighted'))
        metrics['train_recall'].append(recall_score(y_train, y_train_pred, average='weighted'))
        metrics['test_recall'].append(recall_score(y_test, y_test_pred, average='weighted'))
        metrics['train_f1'].append(f1_score(y_train, y_train_pred, average='weighted'))
        metrics['test_f1'].append(f1_score(y_test, y_test_pred, average='weighted'))
        metrics['train_loss'].append(log_loss(y_train, y_train_proba))
        metrics['test_loss'].append(log_loss(y_test, y_test_proba))

    print("=" * 40)
    results_cv = DataFrame(metrics)

    return results_cv

def plot_line_accuracy_log_loss(results_df: DataFrame, metric: Literal['accuracy', 'log_loss'] = 'accuracy') -> Figure:
    """
    Visualize a scatter plot for a chosen metric over the folds of cross-validation.

    Parameters:
    results_df (DataFrame): DataFrame containing cross-validation results with metrics.
    metric (Literal['accuracy', 'log_loss']): The metric to plot, either 'accuracy' or 'log_loss' (default is 'accuracy').

    Returns:
    Figure: A Plotly Figure object with the line plot.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results_df['fold'], 
        y=results_df[f'train_{metric}'],
        mode='lines+markers',
        name=f'Train {metric.capitalize()}',
        line=dict(color='blue'),
        marker=dict(size=10)
    ))

    fig.add_trace(go.Scatter(
        x=results_df['fold'], 
        y=results_df[f'test_{metric}'],
        mode='lines+markers',
        name=f'Test {metric.capitalize()}',
        line=dict(color='green'),
        marker=dict(size=10)
    ))

    fig.update_layout(
        title=f"{metric.capitalize()} over folds",
        xaxis_title='Fold',
        yaxis_title='Score',
        legend_title='Metric',
        template='plotly_white'
    )

    return fig


def plot_bar_metrics(results_df: DataFrame) -> Figure:
    """
    Visualize KNN performance metrics (Accuracy, Precision, Recall, F1 Score) for both train and test sets.

    Parameters:
    results_df (DataFrame): DataFrame containing KNN performance metrics for training and test sets.

    Returns:
    Figure: A Plotly Figure object with the bar plot.
    """
    metrics_summary = {
        'Metric Type': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Train Mean': [
            results_df['train_accuracy'].mean(),
            results_df['train_precision'].mean(),
            results_df['train_recall'].mean(),
            results_df['train_f1'].mean(),
        ],
        'Train Std': [
            results_df['train_accuracy'].std(),
            results_df['train_precision'].std(),
            results_df['train_recall'].std(),
            results_df['train_f1'].std(),
        ],
        'Test Mean': [
            results_df['test_accuracy'].mean(),
            results_df['test_precision'].mean(),
            results_df['test_recall'].mean(),
            results_df['test_f1'].mean(),
        ],
        'Test Std': [
            results_df['test_accuracy'].std(),
            results_df['test_precision'].std(),
            results_df['test_recall'].std(),
            results_df['test_f1'].std(),
        ]
    }
    summary_df = DataFrame(metrics_summary)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=summary_df['Metric Type'],
        y=summary_df['Train Mean'],
        name='Train',
        error_y=dict(type='data', array=summary_df['Train Std'], visible=True),
        marker_color='lightblue'
    ))
    fig.add_trace(go.Bar(
        x=summary_df['Metric Type'],
        y=summary_df['Test Mean'],
        name='Test',
        error_y=dict(type='data', array=summary_df['Test Std'], visible=True),
        marker_color='salmon'
    ))

    fig.update_layout(
        title='Mean and Standard Deviation of Metrics for Training and Test Sets',
        xaxis_title='Metric Type',
        yaxis_title='Score',
        barmode='group',
        template='plotly_white',
        legend_title='Dataset'
    )

    return fig


def plot_violin_metric(results_df: DataFrame, metric_name: str) -> Figure:
    """
    Create a violin plot for the specified metric (Accuracy, Precision, Recall, F1 Score, or Loss) for both train and test sets.

    Parameters:
    results_df (DataFrame): DataFrame containing performance metrics for KNN across different folds.
    metric_name (str): The metric to plot. Must be one of 'accuracy', 'precision', 'recall', 'f1', 'loss'.

    Returns:
    Figure: A Plotly Figure object with the violin plot.
    """
    valid_metrics = ['accuracy', 'precision', 'recall', 'f1', 'loss']
    if metric_name not in valid_metrics:
        raise ValueError(f"Invalid metric name. Choose from {valid_metrics}.")
    
    train_col = f'train_{metric_name}'
    test_col = f'test_{metric_name}'
    melted_df = melt(results_df[['fold', train_col, test_col]], id_vars=['fold'], 
                     value_vars=[train_col, test_col], 
                     var_name='Dataset', value_name='Score')
    melted_df['Dataset'] = melted_df['Dataset'].apply(lambda x: 'Train' if 'train' in x else 'Test')

    fig = violin(melted_df, x='Dataset', y='Score', color='Dataset', 
                 box=True, points='all',
                 title=f'Violin Plot for {metric_name.capitalize()} Metric',
                 template='plotly_white')

    fig.update_layout(
        yaxis_title=f'{metric_name.capitalize()} Score',
        xaxis_title='Dataset',
    )

    return fig



"""
EXTRA PART:
    - Integration of a model optimization function for a choosen model and grid parameters
"""

def model_optimization(X_train: DataFrame, y_train: DataFrame, model: Any, params: dict, 
                       k_cv: int = 5, scoring: str = 'accuracy', save_res: bool = False, 
                       results_filename: str = "results_gridsearch_cv.csv") -> Tuple[Any, DataFrame]:
    """
    Optimize a model using GridSearchCV with the given parameters and data.

    Parameters:
    X_train (DataFrame): The training features.
    y_train (DataFrame): The training target variable.
    model (Any): The model to be optimized.
    params (dict): The parameters for GridSearchCV.
    k_cv (int): The number of cross-validation folds (default is 5).
    scoring (str): The scoring metric for optimization (default is 'accuracy').
    save_res (bool): Whether to save the results to a CSV file (default is False).
    results_filename (str): The filename for saving results (default is 'results_gridsearch_cv.csv').

    Returns:
    Tuple[Any, DataFrame]: The best estimator and a DataFrame of the grid search results.
    """
    
    grid_search = GridSearchCV(model, params, cv=k_cv, scoring=scoring, return_train_score=True)
    grid_search.fit(X_train, y_train)

    print("="*70)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best model: {grid_search.best_estimator_}")
    print(f"Best {scoring} score: {grid_search.best_score_}")
    print("="*70)

    results_df = DataFrame(grid_search.cv_results_)
    if save_res:
        os.makedirs("Results", exist_ok=True)
        filepath = os.path.join("Results", results_filename)
        results_df.to_csv(filepath, index=False)
        print(f"\nResults saved to {filepath}")

    return grid_search.best_estimator_, results_df