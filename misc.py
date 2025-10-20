import numpy as np

def norm_series(series, invert = False):
    normed = (series - series.min())/(series.max() - series.min())
    if invert:
      normed = (1 - normed)
    return normed

def cap_interp_curve(x, y, x_lim):
    int_y = np.interp(x_lim, x, y)
    x[-1] = x_lim
    y[-1] = int_y
    return x, y
