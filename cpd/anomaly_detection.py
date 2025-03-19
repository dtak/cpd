"""Implementation of the multivariate time series anomaly detection approach.

The approach is described in Barnett et al. (2018) "Relapse prediction in
schizophrenia through digital phenotyping: a pilot study".
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import inv
from statsmodels.tsa.seasonal import seasonal_decompose

logger = logging.getLogger(__name__)


class MultivariateTSAnomalyDetector:
  """Implementation of the multivariate time series anomaly detection approach.

  The approach is described in Barnett et al. (2018) "Relapse prediction in
  schizophrenia through digital phenotyping: a pilot study".
  """

  def __init__(self, alpha=0.05, n_bootstrap=1000, freq=7):
    """Initialize the anomaly detector.

    Parameters:
    -----------
    alpha : float
        Significance level for anomaly detection
    n_bootstrap : int
        Number of bootstrap samples for determining significance threshold
    freq : int
        Frequency for seasonal decomposition (7 for weekly patterns)
    """
    self.alpha = alpha
    self.n_bootstrap = n_bootstrap
    self.freq = freq

  def decompose_series(self, series):
    """Decompose a time series into trend and error components.

    Handles missing values by linear interpolation.

    Parameters:
    -----------
    series : pd.Series
        Time series data with datetime index

    Returns:
    --------
    trend : pd.Series
        Trend component of the series
    error : pd.Series
        Error component of the series
    """
    # Interpolate missing values for decomposition
    interpolated = series.interpolate(method='linear')

    # Decompose with trend and weekly component
    try:
      result = seasonal_decompose(
        interpolated,
        model='additive',
        period=self.freq,
        extrapolate_trend='freq',
      )

      # Error is the residual after removing trend and seasonal components
      error = series - (result.trend + result.seasonal)

      return result.trend, error

    except:
      logger.warning(
        f'Decomposition failed for series {series.name}. Using simpler method.'
      )
      # If decomposition fails (e.g., not enough data), use simpler method
      trend = interpolated.rolling(window=self.freq, min_periods=1).mean()
      error = series - trend
      return trend, error

  def transform_errors_to_zscores(self, errors):
    """Transform errors non-parametrically into Z-scores using rank-based methods.

    Following probability integral transform approach from the paper.

    Parameters:
    -----------
    errors : pd.DataFrame
        DataFrame of error components for each feature

    Returns:
    --------
    zscores : pd.DataFrame
        DataFrame of Z-scores derived from errors
    """
    zscores = pd.DataFrame(index=errors.index, columns=errors.columns)

    for column in errors.columns:
      # Get non-NaN errors for this feature
      valid_errors = errors[column].dropna()

      if len(valid_errors) == 0:
        continue

      # Get ranks (add 1 to make ranks 1-based as in the paper)
      ranks = valid_errors.rank(method='average')

      # Transform to normal distribution using probability integral transform
      # Following formula in the paper: Φ^(-1)((rank(ε_i,j))/(m+1))
      # where m is the number of observations and Φ^(-1) is the inverse CDF of standard normal
      m = len(valid_errors)
      transformed = pd.Series(
        stats.norm.ppf(ranks / (m + 1)), index=valid_errors.index
      )

      zscores.loc[valid_errors.index, column] = transformed

    return zscores

  def hotelling_t2_test(self, zscores):
    """Perform Hotelling's T² test on the transformed Z-scores.

    Parameters:
    -----------
    zscores : pd.DataFrame
        DataFrame of Z-scores

    Returns:
    --------
    t2_stats : pd.Series
        Series of Hotelling's T² statistics for each day
    p_values : pd.Series
        Series of p-values corresponding to the T² statistics
    """
    t2_stats = pd.Series(index=zscores.index)
    p_values = pd.Series(index=zscores.index)

    # Calculate overall covariance matrix (excluding missing values)
    valid_rows = zscores.dropna()
    if len(valid_rows) < 2:  # Need at least 2 rows for covariance
      return t2_stats, p_values

    cov_matrix = valid_rows.cov()

    # Handle potential singular covariance matrix
    try:
      inv_cov = inv(cov_matrix.values)
    except:
      # If inversion fails, use pseudoinverse
      inv_cov = np.linalg.pinv(cov_matrix.values)

    # Get mean vector
    mean_vector = valid_rows.mean().values

    # Calculate T² statistic for each day
    for idx in zscores.index:
      row = zscores.loc[idx].dropna()

      # Skip days with too many missing features
      if len(row) < 2:
        continue

      # Get indices of non-missing features
      valid_cols = row.index

      # Extract relevant parts of covariance matrix and mean vector
      sub_cov_indices = [
        i for i, col in enumerate(cov_matrix.columns) if col in valid_cols
      ]
      sub_inv_cov = inv_cov[np.ix_(sub_cov_indices, sub_cov_indices)]
      sub_mean = mean_vector[sub_cov_indices]

      # Calculate deviation from mean
      deviation = row.values - sub_mean

      # Calculate T² statistic: (x-μ)' Σ^(-1) (x-μ)
      t2 = deviation.T @ sub_inv_cov @ deviation
      t2_stats.loc[idx] = t2

      # Calculate p-value
      p = 1 - stats.chi2.cdf(t2, len(valid_cols))
      p_values.loc[idx] = p

    return t2_stats, p_values

  def bootstrap_significance_threshold(self, errors):
    """Bootstrap the error components to establish a significance threshold.

    Parameters:
    -----------
    errors : pd.DataFrame
        DataFrame of error components for each feature

    Returns:
    --------
    threshold : float
        Significance threshold based on bootstrap distribution
    """
    max_t2_stats = []

    for _ in range(self.n_bootstrap):
      # Bootstrap sample (resample errors assuming stationarity)
      bootstrap_errors = pd.DataFrame(
        index=errors.index, columns=errors.columns
      )

      for col in errors.columns:
        valid_errors = errors[col].dropna()
        if len(valid_errors) == 0:
          continue

        # Resample with replacement
        resampled = np.random.choice(
          valid_errors.values, size=len(valid_errors), replace=True
        )
        bootstrap_errors.loc[valid_errors.index, col] = resampled

      # Transform and test the bootstrap sample
      bootstrap_zscores = self.transform_errors_to_zscores(bootstrap_errors)
      t2_stats, _ = self.hotelling_t2_test(bootstrap_zscores)

      # Store the maximum T² statistic
      if not t2_stats.empty and not np.all(np.isnan(t2_stats)):
        max_t2_stats.append(t2_stats.max())

    # Get threshold as the alpha-quantile of the bootstrap distribution
    if max_t2_stats:
      threshold = np.quantile(max_t2_stats, 1 - self.alpha)
    else:
      threshold = np.nan

    return threshold

  def detect_anomalies(self, df: np.ndarray | pd.DataFrame):
    """Detect anomalies in a multivariate time series.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index and feature columns

    Returns:
    --------
    results : dict
        Dictionary containing anomaly detection results
        - 'anomalies': Boolean Series indicating anomalous days
        - 't2_stats': T² statistics for each day
        - 'p_values': p-values for each day
        - 'threshold': significance threshold
    """
    if isinstance(df, np.ndarray):
      df = pd.DataFrame(
        df, columns=[f'feature_{i}' for i in range(df.shape[1])]
      )

    # Initialize DataFrames for trend and error components
    trends = pd.DataFrame(index=df.index, columns=df.columns)
    errors = pd.DataFrame(index=df.index, columns=df.columns)

    # Decompose each feature
    for column in df.columns:
      series = df[column]
      if series.count() < self.freq:  # Skip features with too few observations
        continue

      trend, error = self.decompose_series(series)
      trends[column] = trend
      errors[column] = error

    # Transform errors to Z-scores
    zscores = self.transform_errors_to_zscores(errors)

    # Calculate T² statistics and p-values
    t2_stats, p_values = self.hotelling_t2_test(zscores)

    # Determine significance threshold via bootstrapping
    threshold = self.bootstrap_significance_threshold(errors)

    # Identify anomalies based on threshold
    anomalies = t2_stats > threshold

    return {
      'anomalies': anomalies,
      't2_stats': t2_stats,
      'p_values': p_values,
      'threshold': threshold,
      'zscores': zscores,
      'errors': errors,
      'trends': trends,
    }

  def plot_results(self, df, results, figsize=(12, 10)):
    """Plot the anomaly detection results.

    Parameters:
    -----------
    df : pd.DataFrame
        Original data
    results : dict
        Results from detect_anomalies method
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Plot original data
    df.plot(ax=axes[0], alpha=0.7, marker='o', linestyle='-', markersize=4)
    axes[0].set_title('Original Time Series Data')
    axes[0].legend(loc='upper right')

    # Plot T² statistics
    results['t2_stats'].plot(
      ax=axes[1], color='blue', marker='o', linestyle='-', markersize=4
    )
    axes[1].axhline(
      results['threshold'],
      color='red',
      linestyle='--',
      label=f'Threshold (α={self.alpha})',
    )
    axes[1].set_title("Hotelling's T² Statistics")
    axes[1].set_ylabel('T²')
    axes[1].legend()

    # Highlight anomalies
    anomaly_days = results['anomalies'][results['anomalies']].index
    if len(anomaly_days) > 0:
      for idx in anomaly_days:
        for ax in axes[:2]:
          ax.axvline(x=idx, color='red', alpha=0.3)

    # Plot p-values
    results['p_values'].plot(
      ax=axes[2], color='green', marker='o', linestyle='-', markersize=4
    )
    axes[2].axhline(
      self.alpha, color='red', linestyle='--', label=f'α={self.alpha}'
    )
    axes[2].set_title('P-values')
    axes[2].set_ylabel('p-value')
    axes[2].set_yscale('log')
    axes[2].legend()

    plt.tight_layout()
    return fig


# Example usage
def synthetic_example():
  """Generate synthetic data similar to the paper's description.

  Also demonstrates the anomaly detection method.
  """
  n_days = 180
  n_weeks = n_days // 7
  # Create a date range for 3 months of daily data
  index = pd.date_range(start='2023-01-01', periods=n_days, freq='D')

  # Generate synthetic data for multiple features
  np.random.seed(42)

  # Normal behavioral patterns with weekly seasonality
  mobility_base = np.sin(np.linspace(0, 12 * np.pi, n_days)) * 2 + 10
  mobility_weekly = np.tile([0, 0.5, 0, 0, -0.5, -1, 1], n_weeks + 1)[:n_days]
  mobility = mobility_base + mobility_weekly + np.random.normal(0, 0.5, n_days)

  social_base = np.cos(np.linspace(0, 6 * np.pi, n_days)) * 3 + 15
  social_weekly = np.tile([1, 0.5, 0, -0.5, -1, 2, 1.5], n_weeks + 1)[:n_days]
  social = social_base + social_weekly + np.random.normal(0, 0.8, n_days)

  clinical_base = np.ones(n_days) * 5
  clinical_weekly = np.tile([0, 0, 0.2, 0.5, 0.2, 0, 0], n_weeks + 1)[:n_days]
  clinical = clinical_base + clinical_weekly + np.random.normal(0, 0.3, n_days)

  # Create a dataframe
  df = pd.DataFrame(
    {'mobility': mobility, 'social': social, 'clinical': clinical}, index=index
  )

  # Add some missing values (as mentioned in the paper)
  mask = np.random.random(df.shape) > 0.9
  df[mask] = np.nan

  # Induce anomalies (simulating pre-relapse behavior) around day 70-75
  relapse_window = slice('2023-03-10', '2023-03-15')
  df.loc[relapse_window, 'mobility'] += 3
  df.loc[relapse_window, 'social'] -= 4
  df.loc[relapse_window, 'clinical'] += 2

  # Create and apply the detector
  detector = MultivariateTSAnomalyDetector(alpha=0.05, n_bootstrap=500, freq=7)
  results = detector.detect_anomalies(df)

  # Plot the results
  fig = detector.plot_results(df, results)

  # Print the detected anomalies
  anomaly_days = results['anomalies'][results['anomalies']].index
  print(f'Detected {len(anomaly_days)} anomalous days:')
  for day in anomaly_days:
    print(
      f'- {day.date()} (T² = {results["t2_stats"][day]:.2f}, p = {results["p_values"][day]:.4f})'
    )

  return df, results, fig


# Run the example if this script is executed
if __name__ == '__main__':
  df, results, fig = synthetic_example()
  plt.show()
