import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, log_loss

from src.constants import RANDOM_SEED


# Function to split a dataframe with train_test_split
def data_split(df, target, test_size = 0.3):

    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED)
    
    return X_train, X_test, y_train, y_test


# Function to train K-Nearest Neighbors (KNN) model and optimize it with a GridSearchCV
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


# Function to find the best number of neighbors using a k-folds strafied cross validation
def evaluate_knn_neighbors(df, target, min_neighbors=1, max_neighbors=20, cv=5):
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
        results['train_accuracy_mean'].append(np.mean(train_accuracies))
        results['test_accuracy_mean'].append(np.mean(test_accuracies))
        results['train_log_loss_mean'].append(np.mean(train_log_losses))
        results['test_log_loss_mean'].append(np.mean(test_log_losses))
        results['train_accuracy_std'].append(np.std(train_accuracies))
        results['test_accuracy_std'].append(np.std(test_accuracies))
        results['train_log_loss_std'].append(np.std(train_log_losses))
        results['test_log_loss_std'].append(np.std(test_log_losses))
        
        print("-"*100)
        print(f"n_neighbors={n_neighbors}: Train Accuracy={np.mean(train_accuracies):.4f} Test Accuracy={np.mean(test_accuracies):.4f} "
              f"Train Log Loss={np.mean(train_log_losses):.4f} Test Log Loss={np.mean(test_log_losses):.4f}")
    
    results_df = pd.DataFrame(results)
    
    return results_df


# Function to visualize results from cross-validation to find the best number of neighbors for KNN
def plot_cv_results(df):
    
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(df['n_neighbors'], df['train_accuracy_mean'], '-o', label=f'Train Accuracy (Mean: {df["train_accuracy_mean"].mean():.2f}, Std: {df["train_accuracy_mean"].std():.2f})', color='blue')
    ax1.plot(df['n_neighbors'], df['test_accuracy_mean'], '-o', label=f'Test Accuracy (Mean: {df["test_accuracy_mean"].mean():.2f}, Std: {df["test_accuracy_mean"].std():.2f})', color='green')
    ax1.set_xlabel('Number of Neighbors')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy over Different Numbers of Neighbors')
    ax1.set_ylim(0.8, 1)  
    ax1.set_xticks(df['n_neighbors']) 
    ax1.legend(loc='best', fontsize=10)

    ax2.plot(df['n_neighbors'], df['train_log_loss_mean'], '-o', label=f'Train Log Loss (Mean: {df["train_log_loss_mean"].mean():.2f}, Std: {df["train_log_loss_mean"].std():.2f})', color='red')
    ax2.plot(df['n_neighbors'], df['test_log_loss_mean'], '-o', label=f'Test Log Loss (Mean: {df["test_log_loss_mean"].mean():.2f}, Std: {df["test_log_loss_mean"].std():.2f})', color='orange')
    ax2.set_xlabel('Number of Neighbors')
    ax2.set_ylabel('Log Loss')
    ax2.set_title('Log Loss over Different Numbers of Neighbors')
    ax2.set_xticks(df['n_neighbors']) 
    ax2.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.show()


# Function to performe a cross-validation using the number of neighbors for the best KNN model
def cross_validation_model(df, target, n_neighbors, cv=5):

    X = df.drop(columns=[target])
    y = df[target]

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)

    metrics = {
        'fold': [],
        'train_accuracy': [], 'test_accuracy': [],
        'train_precision': [], 'test_precision': [],
        'train_recall': [], 'test_recall': [],
        'train_f1': [], 'test_f1': [],
        'train_loss': [], 'test_loss': []
    }

    print("="*40)
    for fold, (train_index, test_index) in enumerate(skf.split(X,y)):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        print(f"Training KNN in Fold {fold+1}...")
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

    print("="*40)
    results_cv = pd.DataFrame(metrics)

    return results_cv


# Function to visualize scatter plot for a choosen metric over the folds
def plot_line_accuracy_log_loss(results_df, metric = 'accuracy'):

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
        title = f"{metric.capitalize()} over folds",
        xaxis_title='Fold',
        yaxis_title='Score',
        legend_title='Metric',
        template='plotly_white'
    )

    return fig


# Function to visualize KNN performances for 'Accuracy', 'Precision', 'Recall', 'F1 Score' both train and test
def plot_bar_metrics(results_df):
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
    summary_df = pd.DataFrame(metrics_summary)

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


# Violin Plot for a choosen metric both train and test 
def plot_violin_metric(results_df, metric_name):
    valid_metrics = ['accuracy', 'precision', 'recall', 'f1', 'loss']
    if metric_name not in valid_metrics:
        raise ValueError(f"Invalid metric name. Choose from {valid_metrics}.")
    
    train_col = f'train_{metric_name}'
    test_col = f'test_{metric_name}'
    melted_df = pd.melt(results_df[['fold', train_col, test_col]], id_vars=['fold'], 
                        value_vars=[train_col, test_col], 
                        var_name='Dataset', value_name='Score')
    melted_df['Dataset'] = melted_df['Dataset'].apply(lambda x: 'Train' if 'train' in x else 'Test')

    fig = px.violin(melted_df, x='Dataset', y='Score', color='Dataset', 
                    box=True, points='all',
                    title=f'Violin Plot for {metric_name.capitalize()} Metric',
                    template='plotly_white')

    fig.update_layout(
        yaxis_title=f'{metric_name.capitalize()} Score',
        xaxis_title='Dataset',
    )

    return fig