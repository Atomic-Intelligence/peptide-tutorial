import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random


def compare_distributions(
    synthetic_csv_path: str,
    original_csv_path: str,
    variable: str | None = None,
    num_bins: int = 30,
):
    """
    Function to plot and compare the distributions of a variable from synthetic and original data loaded from CSV files.
    If no variable is provided, a random one is chosen from the dataset columns.

    Parameters:
        synthetic_csv_path (str): Path to the CSV file containing synthetic data.
        original_csv_path (str): Path to the CSV file containing original data.
        variable (str, optional): The variable/column to compare. If None, a random variable is selected. Default is None.
        num_bins (int, optional): Number of bins for the histogram. Default is 30.

    Raises:
        ValueError: If the variable is not found in both the original and synthetic datasets.

    Example:
        `compare_distributions('synthetic_data.csv', 'original_data.csv', variable='age', num_bins=20)`
    """

    # Load data from CSV files
    synthetic_df = pd.read_csv(synthetic_csv_path)
    original_df = pd.read_csv(original_csv_path)

    # If variable is not provided, select one randomly
    if variable is None:
        variable = random.choice(synthetic_df.columns)

    # Check if the variable exists in both dataframes
    if variable not in synthetic_df.columns or variable not in original_df.columns:
        raise ValueError(f"Variable '{variable}' not found in both dataframes")

    # Get the minimum and maximum values from both datasets for binning
    data_min = min(original_df[variable].min(), synthetic_df[variable].min())
    data_max = max(original_df[variable].max(), synthetic_df[variable].max())

    # Create bin edges based on the combined range
    bin_edges = np.linspace(data_min, data_max, num_bins + 1)

    # Create a figure and axis
    plt.figure(figsize=(10, 6))

    # Plot original data distribution with KDE, same bins for consistency
    sns.histplot(
        original_df[variable],
        color="blue",
        label="Original",
        kde=True,
        stat="percent",
        bins=bin_edges,
        alpha=0.5,
    )

    # Plot synthetic data distribution with KDE, using the same bin edges
    sns.histplot(
        synthetic_df[variable],
        color="orange",
        label="Synthetic",
        kde=True,
        stat="percent",
        bins=bin_edges,
        alpha=0.5,
    )

    # Add labels and title
    plt.title(f"Distribution of '{variable}' - Original vs Synthetic")
    plt.xlabel(variable)
    plt.ylabel("Density")

    # Show legend
    plt.legend()

    # Display the plot
    plt.show()


def compare_correlations(
    synthetic_csv_path: str,
    original_csv_path: str,
    col1: str | None = None,
    col2: str | None = None,
):
    """
    Function to plot and compare the correlation between two variables from synthetic and original data loaded from CSV files.
    If no columns are provided, two random columns are selected. Correlations are plotted on the same graph.

    Parameters:
        synthetic_csv_path (str): Path to the CSV file containing synthetic data.
        original_csv_path (str): Path to the CSV file containing original data.
        col1 (str, optional): The first variable/column for correlation comparison. If None, a random variable is selected. Default is None.
        col2 (str, optional): The second variable/column for correlation comparison. If None, a random variable is selected. Default is None.

    Raises:
        ValueError: If either column is not found in both the original and synthetic datasets.

    Example:
        `compare_correlations('synthetic_data.csv', 'original_data.csv', col1='age', col2='weight')`
    """

    # Load data from CSV files
    synthetic_df = pd.read_csv(synthetic_csv_path)
    original_df = pd.read_csv(original_csv_path)

    # If columns are not provided, select two random ones
    if col1 is None or col2 is None:
        col1, col2 = random.sample(synthetic_df.columns.tolist(), 2)

    # Check if both columns exist in both dataframes
    if col1 not in synthetic_df.columns or col2 not in synthetic_df.columns:
        raise ValueError(
            f"Columns '{col1}' and/or '{col2}' not found in the synthetic dataset"
        )
    if col1 not in original_df.columns or col2 not in original_df.columns:
        raise ValueError(
            f"Columns '{col1}' and/or '{col2}' not found in the original dataset"
        )

    # Compute correlations for both datasets
    synthetic_corr = synthetic_df[[col1, col2]].corr().iloc[0, 1]
    original_corr = original_df[[col1, col2]].corr().iloc[0, 1]

    # Create a figure and axis
    plt.figure(figsize=(10, 6))

    # Plot original data scatter plot
    sns.scatterplot(
        x=col1,
        y=col2,
        data=original_df,
        color="blue",
        label=f"Original (corr = {original_corr:.2f})",
        alpha=0.6,
    )

    # Plot synthetic data scatter plot
    sns.scatterplot(
        x=col1,
        y=col2,
        data=synthetic_df,
        color="orange",
        label=f"Synthetic (corr = {synthetic_corr:.2f})",
    )

    # Add title and labels
    plt.title(f"Correlation of '{col1}' vs '{col2}' - Original vs Synthetic")
    plt.xlabel(col1)
    plt.ylabel(col2)

    # Show legend
    plt.legend()

    # Display the plot
    plt.show()
