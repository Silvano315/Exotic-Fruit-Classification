import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px


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


# Function to perform EDA with plotly.express: box/violin plot and histogram with or without boxplot for comparison
def plot_feature_distribution(df, feature, comparison = None):
    if comparison:
        fig = px.histogram(df, x=feature, labels={feature:feature},
                           color=comparison,
                           title=f'Distribution of {feature} compared with {comparison}')
    else:
        fig = px.histogram(df, x=feature, labels={feature:feature},
                           marginal='box', #'violin' for violin plot
                           title=f'Distribution of {feature}')
    fig.show()