import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, brier_score_loss
from relplot.metrics import smECE as smece
from . import misc

def ece(y_true: np.array, y_pred: np.array, n_bins: int = 10) -> float:
    """
    Calculate the Expected Calibration Error: for each bin, the absolute difference between
    the mean fraction of positives and the average predicted probability is taken. The ECE is
    the weighed mean of these differences.

    Parameters
    ----------
    y_true: np.ndarray
        The true labels.
    y_pred: np.ndarray
        The predicted probabilities
    n_bins: int
        The number of bins to use.
    Returns
    -------
    ece: float
        The expected calibration error.
    """
    n = len(y_pred)
    bins = np.arange(0.0, 1.0, 1.0 / n_bins)
    bins_per_prediction = np.digitize(y_pred, bins)

    df = pd.DataFrame({"y_pred": y_pred, "y": y_true, "pred_bins": bins_per_prediction})

    grouped_by_bins = df.groupby("pred_bins")
    # calculate the mean y and predicted probabilities per bin
    binned = grouped_by_bins.mean()

    # calculate the number of items per bin
    binned_counts = grouped_by_bins["y"].count()

    # calculate the proportion of data per bin
    binned["weight"] = binned_counts / n

    weighed_diff = abs(binned["y_pred"] - binned["y"]) * binned["weight"]
    return weighed_diff.sum()


def confidence_metrics(df, correct_col, conf_col, n_bins=10):
  conf = df[conf_col].copy()
  conf = misc.norm_series(conf)
  print(conf_col)
  print(conf.describe())
  results = {}
  results['auc'] = roc_auc_score(df[correct_col],  conf)
  results['brier'] = brier_score_loss(df[correct_col],  conf)
  results['ece'] = ece(df[correct_col], conf, n_bins=10)
  results['smece'] = smece(df[correct_col], conf)
  return results
