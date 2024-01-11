import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from fracdiff2 import frac_diff_ffd
from utils import plot_multi

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
    df = pd.read_csv('../doc/sp500.csv', index_col=0, parse_dates=True)
    df = df['1993':]
    fractional_returns = frac_diff_ffd(df['Close'].apply(np.log).values, d=0.4)
    df['Fractional differentiation FFD'] = fractional_returns
    df['SP500'] = df['Close']
    df = df[['SP500', 'Fractional differentiation FFD']]
    print(df.tail())
    # burn the first days, where the weights are not defined.
    plot_multi(df[1500:])
    plt.show()


if __name__ == '__main__':
    main()
