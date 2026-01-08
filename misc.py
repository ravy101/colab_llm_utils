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
        new_series[new_series > uq] = uq
    
    if l_quantile:
        lq = np.quantile(series, l_quantile)
        new_series[new_series < lq] = lq
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
    norm1 = np.linalg.norm(arr1)
    norm2 =  np.linalg.norm(arr2)
    if dprod == 0 or norm1 == 0 or norm2 == 0:
        result = 0
    else:
        result = dprod/(norm1*norm2)
    return result

def dist_transform(dist, a = .5, b=1.2):
    res = a*(dist+1)**3 - b*dist - b
    res = min(res, 1)
    res = max(res, -1)
    return res

def dist_transform2(dist, a = .5, b=1.4):
    if dist < 0:
      res = .75
    elif dist < .4:
      res =  0
    elif dist < .75:
      res = .5
    else:
      res = .75
    return res

def gaussian_valley(sim, mu=0.25, sigma=0.25, low=-0., high=.750):
    return low + (high - low) * (1 - np.exp(-((sim - mu)**2) / (2 * sigma**2)))

def generalized_gaussian_valley(sim, mu=0.25, sigma=0.5, p=2,
                                low=-0.2, high=.8):
    z = np.abs((sim - mu) / sigma)
    return low + (high - low) * (1 - np.exp(-(z ** p)))

