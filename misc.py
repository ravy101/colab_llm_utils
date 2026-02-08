import numpy as np
from . import likelihood

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

def extra_cols(df):
  df['random'] = np.random.rand(len(df))
  df['output_len'] = [len(l) for l in df['logit_outs']]
  df['word_len'] = [len(r[0].split()) for r in df['responses']]
  df['word_len_large'] = [len(r[0].split()) for r in df['responses_large']]
  df['words_per_token'] = df['word_len'] / df['output_len']
  df['log_chow_av'] = [likelihood.log_chow_av(p) for p in df['top_probas']]
  df['cvar'] = [likelihood.chow_cvar_uncertainty(p) for p in df['top_probas']]
  df['quant_5'] = [likelihood.chow_quantile(p, alpha=.5) for p in df['top_probas']]
  df['normed_quantile'] = norm_series(df['quant_5'], invert = True)
  df['normed_av'] = norm_series(df['chow_av'], invert = True)
  df['normed_sum'] = norm_series(df['chow_sum'], invert = True)


def cuda_duignostics():
    print("===== PYTHON =====")
    import sys, platform
    print("Python:", sys.version)
    print("Executable:", sys.executable)
    print("Platform:", platform.platform())

    print("\n===== CUDA / GPU =====")
    import subprocess
    try:
        print(subprocess.check_output(["nvidia-smi"], text=True))
    except Exception as e:
        print("nvidia-smi failed:", e)

    print("\n===== TORCH =====")
    import torch
    print("torch:", torch.__version__)
    print("torch cuda version:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        free, total = torch.cuda.mem_get_info()
        print(f"VRAM free / total: {free/1e9:.2f} GB / {total/1e9:.2f} GB")

    print("\n===== TRANSFORMERS =====")
    import transformers
    print("transformers:", transformers.__version__)
    print("transformers path:", transformers.__file__)

    print("\n===== ACCELERATE =====")
    import accelerate
    print("accelerate:", accelerate.__version__)
    print("accelerate path:", accelerate.__file__)

    print("\n===== BITSANDBYTES =====")
    import bitsandbytes
    print("bitsandbytes:", bitsandbytes.__version__)
    try:
        import bitsandbytes as bnb
        print("bitsandbytes CUDA test:")
        subprocess.run(["python", "-m", "bitsandbytes"], check=False)
    except Exception as e:
        print("bitsandbytes import error:", e)

    print("\n===== NUMPY =====")
    import numpy
    print("numpy:", numpy.__version__)
    print("numpy path:", numpy.__file__)

    print("\n===== PIP FREEZE (relevant packages) =====")
    import pkg_resources
    pkgs = sorted(
        p for p in pkg_resources.working_set
        if any(k in p.project_name.lower() for k in [
            "torch", "transformers", "accelerate", "bitsandbytes",
            "numpy", "scipy", "flash", "xformers"
        ])
    )
    for p in pkgs:
        print(p)

    print("\n===== CUDA ALLOC CONF =====")
    import os
    print("PYTORCH_CUDA_ALLOC_CONF =", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))


    print("BnB CONFIG")
    print(model_config['bnb_config'])
    print("\n===== DONE =====")