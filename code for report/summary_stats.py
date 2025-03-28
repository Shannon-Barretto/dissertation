import numpy as np
import pandas as pd

def compute_summary_statistics(original_data, synthetic_data):
    """
    Compute summary statistics for the original and synthetic data.

    Parameters:
        original_data (dataframe): The original data.
        synthetic_data (dataframe): The synthetic data.

    Returns:
        Relative (%) average difference between the summary statistics of 
        the original and synthetic data.
    """

    total_difference = 0
    n_features = len(original_data.columns)

    # Summary statistics
    # 
    original_data_summary = original_data.describe().drop(
        index=["count", "min", "max"], axis=0)
    synthetic_data_summary = synthetic_data.describe().drop(
        index=["count", "min", "max"], axis=0)

    for column in original_data.columns:
        # Extract the feature summary statistics
        original_col_stat = original_data_summary[column]
        synthetic_col_stat = synthetic_data_summary[column]

        # To compute the relative scale based difference
        data_range = original_col_stat.max() - original_col_stat.min()
        # Compute the relative difference
        relative_difference = np.abs(
            original_col_stat - synthetic_col_stat) / data_range

        # Find the mean relative difference and add to total difference
        total_difference += np.mean(relative_difference)

    # A lower relative difference indicates a better match between the 
    # original and synthetic data
    return total_difference / n_features
