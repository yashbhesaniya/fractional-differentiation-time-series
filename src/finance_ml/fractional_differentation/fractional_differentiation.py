
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot import *

def _get_weight_ffd(d, thres, lim):
    w, k = [1.], 1
    ctr = 0
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
        ctr += 1
        if ctr == lim - 1:
            break
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def frac_diff_ffd(x, d, thres=1e-5, disable_warning=False):
    if np.max(x) > 10.0 and not disable_warning:
        print('WARNING: have you applied log before calling this function? If yes, discard this warning.')
    w = _get_weight_ffd(d, thres, len(x))
    width = len(w) - 1
    output = []
    output.extend([0] * width)
    for i in range(width, len(x)):
        output.append(np.dot(w.T, x[i - width:i + 1])[0])
    return np.array(output) 



def main():
    df = pd.read_parquet('/home/yash/Desktop/New Folder/ML-in-Finance/data/cryptos/BTCUSD_2020-04-07_2022-04-06.parquet', engine = 'pyarrow')
    fractional_returns = frac_diff_ffd(df['OPEN'].apply(np.log).values, d=0.4)
    df['Fractional differentiation FFD'] = fractional_returns
    df = df[['OPEN', 'Fractional differentiation FFD']]
    print(df.tail())
    plot_multi(df)
    plt.show()