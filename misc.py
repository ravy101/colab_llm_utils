import numpy as np


def norm_series(series, invert = False):
    normed = (series - series.min())/(series.max() - series.min())
    if invert:
      normed = (1 - normed)
    return normed

def clip_series(series, u_quantile = None, l_quantile = None):
    new_series = series.copy()
    if u_quantile:
        uq = np.quantile(series, u_quantile)
        new_series[new_series > u_quantile] = uq
    
    if l_quantile:
        lq = np.quantile(series, l_quantile)
        new_series[new_series < l_quantile] = lq
    return new_series

def cap_interp_curve(x, y, x_lim):
    int_y = np.interp(x_lim, x, y)
    x[-1] = x_lim
    y[-1] = int_y
    return x, y

def dist_mh(arr1, arr2):
    return np.sum(np.abs(arr1 - arr2))

def sim_cosine(arr1, arr2):
    dprod = np.dot(arr1, arr2)
    if dprod == 0:
        result = 0
    else:
        result = dprod / (np.linalg.norm(arr1) * np.linalg.norm(arr2))
    return result



