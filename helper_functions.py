import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from statsmodels.stats.proportion import proportion_confint



def cal_plot_data_prep(y_pred, y_test):
    # create cuts
    cuts = np.quantile(y_pred, q = [0.25, 0.5, 0.75])
    cuts = np.insert(cuts, obj = 0, values = 0)
    cuts = np.append(cuts, 1)
    
    # get indices of cuts
    cut_idx = []
    bin_cuts = []
    for i in range(len(cuts) - 1):
        cut_idx.append(np.where((y_pred >= cuts[i]) & (y_pred < cuts[i+1]))[0]) 
        bin_cuts.append(str(np.round(cuts[i:i+2], 4)))
        
    pred_prob = []
    obs_prop = []
    obs_prop_up = []
    obs_prop_lo = []
    obs_cases = []

    # calc confint
    for idx in cut_idx:
        pred_prob.append(np.mean(y_pred[idx]))
        obs_prop.append(np.mean(y_test[idx]))
        obs_cases.append(np.sum(y_test[idx]))
        confint = proportion_confint(count = np.sum(y_test[idx]), 
                                 nobs = len(y_test[idx]), 
                                 alpha = 0.05, 
                                 method = "wilson")
        obs_prop_lo.append(confint[0])
        obs_prop_up.append(confint[1])
        
    # save as df
    cal_plot_data = pd.DataFrame(
        {"predicted_probability_mean": pred_prob,
         "predicted_probability_middle": np.convolve(cuts,[.5,.5],"valid"),
         "observed_proportion": obs_prop,
         "observed_proportion_lower": obs_prop_lo,
         "observed_proportion_upper": obs_prop_up,
         "observed_cases": obs_cases,
         "bin_cuts": bin_cuts
        }
    )
    
    return cal_plot_data


def cal_plot(dat, x_vals, y_vals, lwr_vals, upr_vals, alpha = 1, show = True, col = "C0"):
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    sns.lineplot(
        data=dat, x=x_vals, y=y_vals, 
        marker="o",
        alpha = alpha,
        color = col
    )

    plt.plot([0, 1], [0, 1], c = "grey", linewidth = 2)
    for i in range(dat.shape[0]):
        plt.axvline(dat[x_vals].tolist()[i], 
                    dat[lwr_vals].tolist()[i], 
                    dat[upr_vals].tolist()[i],
                    alpha = alpha,
                    color = col)
    if show:
        plt.show()
        
def inverse_sigmoid(y):
    return np.log(y/(1-y))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))