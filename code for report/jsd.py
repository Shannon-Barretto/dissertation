from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
import numpy as np

def compute_js_divergence(original_data, synthetic_data, bins=50):
    """
    Compute the Jensen-Shannon divergence between the original and 
    synthetic data.

    Parameters:
        original_data (dataframe): The original data.
        synthetic_data (dataframe): The synthetic data.
        bins (int): The number of bins to use for the histogram.

    Returns:
        float: The Jensen-Shannon divergence between the original and 
        synthetic data.
    """

    total_divergence = 0
    n_features = len(original_data.columns)

    for column in original_data.columns:
        # Extract the feature values
        original_feature = original_data[column].values
        synthetic_feature = synthetic_data[column].values

        # Compute range for current feature/histogram
        min_value = min(original_feature.min(), synthetic_feature.min())
        max_value = max(original_feature.max(), synthetic_feature.max())
        x_range = np.linspace(min_value, max_value, bins)

        # Estimate the densities
        kde_original = gaussian_kde(original_feature)
        kde_synthetic = gaussian_kde(synthetic_feature)

        # Compute probabilities
        p = kde_original(x_range)
        q = kde_synthetic(x_range)

        # Add epsilon to prevent division by 0
        epsilon = 1e-10

        # To prevent division by 0
        p_sum = np.sum(p)
        q_sum = np.sum(q)
        if p_sum <= epsilon or q_sum <= epsilon:
            print(f"Warning: Unstable density estimation detected "
                f"(p_sum={p_sum}, q_sum={q_sum}). Using fallback value:0.5")
            return 0.5  # Return a moderate fallback value

        # Normalize to ensure they sum to 1
        p = p / p_sum
        q = q / q_sum

        # Add to tatal divergence
        total_divergence += jensenshannon(p, q)

        # Final safety check on the result
        if np.isnan(total_divergence) or np.isinf(total_divergence):
            print(f"Warning: JS divergence calculation unstable, using "
                  f"fallback value:0.5")
            return 0.5

    # A lower JS divergence indicates a better match between the original 
    # and synthetic data
    return total_divergence / n_features
