from pandas import DataFrame, concat, read_csv
from typing import Optional
import plotly.express as px
from plotly.graph_objs import Figure
import plotly.figure_factory as ff
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer



def read_data(file_path: str, chunksize: Optional[int] = None) -> DataFrame:
    """
    Reads a CSV file into a DataFrame, with optional chunking for large files.

    Parameters:
    file_path (str): Path to the CSV file.
    chunksize (int, optional): Number of rows to read at a time if chunking is needed.

    Returns:
    DataFrame: The loaded DataFrame.
    """
    try:
        if chunksize:
            chunks = read_csv(file_path, chunksize=chunksize)
            return concat(chunks, ignore_index=True)
        else:
            return read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return DataFrame()


def handle_duplicates(df: DataFrame) -> DataFrame:
    """
    Checks and removes duplicates from a DataFrame.

    Parameters:
    df (DataFrame): The DataFrame to check for duplicates.

    Returns:
    DataFrame: The DataFrame with duplicates removed.
    """
    try:
        duplicates = df.duplicated().sum()

        if duplicates > 0:
            print(f"Number of duplicates: {duplicates}. Original DataFrame shape: {df.shape}")
            df_cleaned = df.drop_duplicates()
            print(f"Removed duplicates: {duplicates}. New DataFrame shape: {df_cleaned.shape}")
        else:
            print("No duplicates found!")
            df_cleaned = df

        return df_cleaned

    except Exception as e:
        print(f"Error handling duplicates: {e}")
        return df 


def histogram_for_single_feature(df: DataFrame, feature: str, comparison: str = None) -> Figure:
    """
    Creates a histogram (optionally with a boxplot or violin plot) to display the distribution of a feature.
    If a comparison column is provided, the distribution is split by the comparison feature.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    feature (str): The feature to visualize.
    comparison (str, optional): The feature for comparison (e.g., a categorical variable for color coding). Defaults to None.

    Returns:
    Figure: The Plotly figure object for further customization or display.
    """
    try:
        if comparison:
            fig = px.histogram(df, x=feature, labels={feature: feature},
                               color=comparison,
                               title=f'Distribution of {feature} compared with {comparison}')
        else:
            fig = px.histogram(df, x=feature, labels={feature: feature},
                               marginal='box',  # 'violin' can be used for violin plot
                               title=f'Distribution of {feature}, mean: {df[feature].mean():.2f}, std: {df[feature].std():.2f}')
        
        return fig  

    except Exception as e:
        print(f"Error in plotting feature distribution: {e}")
        return None 


def scatter_plot(df: DataFrame, feature_x: str, feature_y: str, target: str) -> Figure:
    """
    Creates a scatter plot using Plotly to visualize the relationship between two features
    and adds a color comparison based on a target column.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    feature_x (str): The feature to plot on the x-axis.
    feature_y (str): The feature to plot on the y-axis.
    target (str): The target column for color coding in the scatter plot.

    Returns:
    Figure: The Plotly figure object for further customization or display.
    """
    try:
        fig = px.scatter(df, x=feature_x, y=feature_y,
                         color=target,
                         # Optional: marginal_x='histogram', marginal_y='rug',
                         title=f'Scatter Plot {feature_x} vs {feature_y} with target column comparison')

        fig.update_traces(marker=dict(size=12,
                                      line=dict(width=2, color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        
        return fig 

    except Exception as e:
        print(f"Error creating scatter plot: {e}")
        return None 


def kde_for_single_feature(df: DataFrame, feature: str, bin: float = 0.2) -> Figure:
    """
    Creates a KDE plot with a normal distribution curve, including a histogram and rug plot.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    feature (str): The feature to visualize.
    bin (float, optional): The bin size for the histogram. Defaults to 0.2.

    Returns:
    Figure: The Plotly figure object for further customization or display.
    """
    try:
        fig = ff.create_distplot([df[feature]], group_labels=[feature], 
                                 bin_size=bin, colors=['coral'],
                                 curve_type='normal')
        fig.update_layout(title_text=f'Distplot with Normal Distribution and Rug Plot for {feature}')
        
        return fig  

    except Exception as e:
        print(f"Error creating KDE plot: {e}")
        return None


def feature_scaling(df: DataFrame, method: str = 'Standard') -> DataFrame:
    """
    Scales the features of the dataset using the specified method.

    Parameters:
    df (DataFrame): The DataFrame containing the features and target variable.
    method (str, optional): The scaling method to use ('Standard' or 'MinMax'). Defaults to 'Standard'.

    Returns:
    DataFrame: The DataFrame with scaled features and the target variable.
    
    Raises:
    ValueError: If the specified scaling method is not recognized.
    """
    try:
        X = df.drop(columns='Fruit')
        y = df['Fruit']
        feature_names = X.columns

        if method == 'Standard':
            preprocessing = ColumnTransformer(
                transformers=[
                    ('standard_scaling', StandardScaler(), feature_names)
                ],
                remainder='passthrough'
            )
        elif method == 'MinMax':
            preprocessing = ColumnTransformer(
                transformers=[
                    ('min_max_scaling', MinMaxScaler(), feature_names)
                ],
                remainder='passthrough'
            )
        else:
            raise ValueError('The only available methods are: MinMax and Standard. Type one of these!')

        X_scaled = preprocessing.fit_transform(X)
        X_scaled = DataFrame(X_scaled, columns=feature_names)

        df_transformed = concat([X_scaled, y], axis=1)

        return df_transformed

    except Exception as e:
        print(f"Error in feature scaling: {e}")
        return df