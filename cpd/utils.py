import json
import os

import numpy as np

TCPD_DATASETS_DIR = (
  '/Users/abhisheksharma-mbpro/Documents/cpd-moht/TCPD/datasets/'  # noqa: E501
)
TCPD_ANNOTATIONS_PATH = (
  '/Users/abhisheksharma-mbpro/Documents/cpd-moht/TCPD/annotations.json'  # noqa: E501
)


def get_tcpd_dataset_paths(
  tcpd_dir: str = TCPD_DATASETS_DIR,
) -> dict[str, str]:
  """Get the paths to all the datasets in the TCPD directory."""
  dataset_paths = {}
  for root, dirs, files in os.walk(tcpd_dir):
    json_files = [file for file in files if file.endswith('.json')]
    if json_files:
      dataset_paths[os.path.basename(root)] = os.path.join(root, json_files[0])

  return dataset_paths


def get_tcpd_annotations(
  tcpd_annotations_path: str = TCPD_ANNOTATIONS_PATH,
) -> dict[str, dict[str, list[int]]]:
  """Get the annotations for all the datasets in the TCPD directory."""
  annotations: dict[str, dict[str, list[int]]] = json.load(
    open(tcpd_annotations_path)
  )

  return annotations


def load_dataset(filename: str) -> tuple[dict, np.ndarray]:
  """Load a CPDBench dataset.

  Source: https://github.com/alan-turing-institute/TCPDBench/blob/master/execs/python/cpdbench_utils.py
  """
  with open(filename, 'r') as fp:
    data = json.load(fp)

  if data['time']['index'] != list(range(0, data['n_obs'])):
    raise NotImplementedError(
      'Time series with non-consecutive time axis are not yet supported.'
    )

  mat = np.zeros((data['n_obs'], data['n_dim']))
  for j, series in enumerate(data['series']):
    mat[:, j] = series['raw']

  # We normalize to avoid numerical errors.
  mat = (mat - np.nanmean(mat, axis=0)) / np.sqrt(
    np.nanvar(mat, axis=0, ddof=1)
  )

  return data, mat
