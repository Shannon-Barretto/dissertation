import numpy as np
import pandas as pd

def compute_correlation_matrix_mad(original_data, synthetic_data):
    """
    Compute the correlation matrix mean absolute difference (MAD) between 
    the original and synthetic data.

    Parameters:
        original_data (pd.DataFrame): The original data.
        synthetic_data (pd.DataFrame): The synthetic data.

    Returns:
        float: The correlation matrix MAD between the original and synthetic 
        data.
    """

    # Correlation matrix
    original_correlation_matrix = original_data.corr(method="spearman")
    synthetic_correlation_matrix = synthetic_data.corr(method="spearman")

    # Mean Absolute Difference (MAD) of the correlation matrices
    correlation_matrix_mad = np.mean(np.abs(original_correlation_matrix - 
                                            synthetic_correlation_matrix))

    # A lower MAD indicates a better match between the original and 
    # synthetic data
    return correlation_matrix_mad
