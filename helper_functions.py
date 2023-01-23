import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from matplotlib import pyplot as plt
from statsmodels.stats.proportion import proportion_confint
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from scipy import special



def bin_class_report(X_test,y_test, model):
    """
    modified classification report for binary output
    """
    NLL, Acc, AUC0 = model.evaluate( x=X_test, y=y_test, verbose=0)
    y_pred = model.predict(X_test)
    # cm , AUC
    cm = confusion_matrix(np.round(y_test), np.round(y_pred))
    AUC =  metrics.roc_auc_score(np.round(y_test), np.round(y_pred))
    #acc
    nobs = sum(sum(cm))
    count = sum([cm[0,0], cm[1,1]])
    Acc = count/nobs
    acc_ci_low, acc_ci_upp = proportion_confint(count , nobs,  alpha=0.05, method='wilson')
    #sens 
    sens = cm[1,1]/(cm[1,1]+cm[1,0])
    nobs = sum([cm[1,0],cm[1,1]])
    count = sum([cm[1,1]])
    sens_ci_low, sens_ci_upp = proportion_confint(count , nobs,  alpha=0.05, method='wilson')
    #spec 
    spec = cm[0,0]/(cm[0,1]+cm[0,0])
    nobs = sum([cm[0,1],cm[0,0]])
    count = sum([cm[0,0]])
    spec_ci_low, spec_ci_upp = proportion_confint(count , nobs,  alpha=0.05, method='wilson')
    
    print("\nPerformance on Test Set : ")
    print("\nAccuracy    [95% Conf.] :", np.around(Acc,4),np.around([acc_ci_low, acc_ci_upp],4))
    print("Sensitivity [95% Conf.] :", np.around(sens,4), np.around([sens_ci_low, sens_ci_upp],4))
    print("Specificity [95% Conf.] :", np.around(spec,4), np.around([spec_ci_low, spec_ci_upp],4))
    print("\nArea under Curve (AUC) Binary :", np.around(AUC,4))
    print("Area under Curve (AUC) Probability :", np.around(AUC0,4))
    print("Negative Log-Likelihood :", np.around(NLL, 4))
#     print(metrics.classification_report(y_test.argmax(axis=1), y_pred.argmax(axis =1)))

def calc_metrics(y, p):
#     NLL = np.mean(-special.xlogy(y, p) - special.xlogy(1-y, 1-p))
    NLL = tf.keras.losses.binary_crossentropy(y, p)
    AUC =  metrics.roc_auc_score(y, p)
    AUCB =  metrics.roc_auc_score(y, np.round(p))
    
    cm = confusion_matrix(np.round(y), np.round(p))
    nobs = sum(sum(cm))
    count = sum([cm[0,0], cm[1,1]])
    ACC = count/nobs
    
    print("\nPerformance on Test Set : ")
    print("Accuracy:", np.around(ACC,4))
    print("Area under Curve (AUC) Binary :", np.around(AUCB,4))
    print("Area under Curve (AUC) Probability :", np.around(AUC,4))
    print("Negative Log-Likelihood :", np.around(NLL, 4))
    print("Confusion Matrix : \n", cm)


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