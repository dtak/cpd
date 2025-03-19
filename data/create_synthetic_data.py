import numpy as np


def create_synthetic_multivariate_time_series(
  T: int = 200,
  D: int = 5,
  change_points: list[int] = [0, 50, 100, 150, 200],
) -> tuple[np.ndarray, list[int]]:
  """Creates a synthetic multivariate time series dataset with shape (N, T, D).

  Each time series is of length T and dimension D.
  The series is divided into segments by the change_points, with segments alternating
  between:
    - Random observations drawn from a standard normal distribution.
    - A linear trend with random slope and intercept (plus a small noise term).

  Parameters:
    T: Total number of time steps.
    D: Number of features (dimensions) per time step.
    change_points: List of time indices defining segment boundaries.

  Returns:
    Array of shape (T, D) containing the synthetic multivariate time series.
    List of change points.
  """
  data = np.zeros((T, D))
  num_segments = len(change_points) - 1

  # For each sample create segments with alternating patterns:
  for seg in range(num_segments):
    start = change_points[seg]
    end = change_points[seg + 1]
    seg_length = end - start

    if seg % 2 == 0:
      # Random segment: observations drawn from a normal distribution.
      seg_data = np.random.randn(seg_length, D)
    else:
      # Linear segment: each feature follows a linear trend.
      slopes = np.random.uniform(-1, 1, size=D)
      intercepts = np.random.uniform(-5, 5, size=D)
      t_seg = np.arange(seg_length).reshape(-1, 1)  # time vector for segment
      seg_data = intercepts + slopes * t_seg
      # Add a little noise
      seg_data += 0.1 * np.random.randn(seg_length, D)
    data[start:end, :] = seg_data

  data = (data - np.mean(data, axis=0, keepdims=True)) / np.std(
    data, axis=0, keepdims=True
  )

  return data, change_points
