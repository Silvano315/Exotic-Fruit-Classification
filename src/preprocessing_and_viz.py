import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer

# Function to check if there are duplicates in the dataset
def handle_duplicates(df):
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Number of duplicates: {duplicates}. Shape DF: {df.shape}")
        df_cleaned = df.drop_duplicates()
        print(f"Number of removed duplicates: {duplicates}. New shape DF: {df_cleaned.shape}")
    else:
        print("No duplicates found!")
        df_cleaned = df
    return df_cleaned


# Function to perform EDA with plotly.express: box/violin plot and histogram with or without boxplot and comparison
def plot_feature_distribution(df, feature, comparison = None):
    if comparison:
        fig = px.histogram(df, x=feature, labels={feature:feature},
                           color=comparison,
                           title=f'Distribution of {feature} compared with {comparison}')
    else:
        fig = px.histogram(df, x=feature, labels={feature:feature},
                           marginal='box', #'violin' for violin plot
                           title=f'Distribution of {feature}, with mean {df[feature].mean():.2f} and std {df[feature].std():.2f}')
    fig.show()


# Function to perform EDA with plotly.express: scatter plot with comparison using Fruit target column
def scatter_plot(df, feature_x, feature_y, target):
    fig = px.scatter(df, feature_x, feature_y,
                     color=target,
                     #marginal_x='istogram', marginal_y='rug',
                     title=f'Scatter Plot {feature_x} vs {feature_y} with target column comparison')
    
    fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    fig.show()


# Function to perform EDA with plotly.figure_factory: kde plot with normal distribution, also display histogram and rug plot
def plot_feature_kde(df, feature, bin = 0.2):

    fig = ff.create_distplot([df[feature]], group_labels=[feature], 
                             bin_size=bin, colors=['coral'],
                             curve_type='normal'
                             )
    fig.update_layout(title_text = f'Distplot with Normal Distribution and Rug Plot for {feature}')
    fig.show()


# Function to scale the features of the dataset
def feature_scaling(df, method = 'Standard'):

    X = df.drop(columns='Fruit')
    y = df['Fruit']
    feature_names = X.columns

    if method == 'Standard':
        preprocessing = ColumnTransformer(
        transformers = [
            ('standard_scaling', StandardScaler(), feature_names)
        ],
        remainder = 'passthrough'
        )
    elif method == 'MinMax':
        preprocessing = ColumnTransformer(
        transformers = [
            ('min_max_scaling', MinMaxScaler(), feature_names)
        ],
        remainder = 'passthrough'
        )
    else:
        raise NameError('The only available methods are: MinMax and Standard. Type one of that!')

    X = preprocessing.fit_transform(X)

    X = pd.DataFrame(X, columns=feature_names)
    df_transformed = pd.concat([X, y], axis=1)

    return df_transformed