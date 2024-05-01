# same but with a loop for all the frequencies
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind


def auc_ppv_score(topo_features_surv, topo_features_nsurv, plot=False):
    aucs = []
    ppvs = []
    for i in range(40):
        topo_feature_freq_i = np.concatenate((-topo_features_surv[:, i].flatten(), -topo_features_nsurv[:,i].flatten()))
        labels = np.concatenate((np.ones_like(topo_features_surv[:, i].flatten()), np.zeros_like(topo_features_nsurv[:,i].flatten())))
        fpr, tpr, thresholds = roc_curve(labels, topo_feature_freq_i)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        positive_predictions = np.sum(topo_feature_freq_i >= optimal_threshold)
        if positive_predictions > 0:
            ppv = np.sum((topo_feature_freq_i >= optimal_threshold) & (labels == 1)) / positive_predictions
        else:
            ppv = 0
        ppvs.append(ppv)
    #print(f"Max AUC: {max(aucs)} at frequency {np.argmax(aucs)+1}")
    #print(f"Max PPV: {max(ppvs)} at frequency {np.argmax(ppvs)+1}")

    if plot:
        # plot the AUCs
        plt.figure(figsize=(8, 6))
        plt.plot(range(40), aucs, color='darkorange', lw=2)
        plt.xlabel('Frequency')
        plt.ylabel('AUC')
        plt.title('AUC for each frequency')
        plt.show()

        # plot the PPVs
        plt.figure(figsize=(8, 6))
        plt.plot(range(40), ppvs, color='darkorange', lw=2)
        plt.xlabel('Frequency')
        plt.ylabel('PPV')
        plt.title('PPV for each frequency')
        plt.show()

    return aucs, ppvs


def t_test_topo_features(topo_features_surv, topo_features_nsurv, best_features=False):
    t_test_results = []
    for i in range(40):
        _, p = ttest_ind(topo_features_surv[:, i], topo_features_nsurv[:, i], equal_var=False)
        t_test_results.append(p)
    if best_features:
        return np.min(t_test_results), np.argmin(t_test_results)
    else:
        return t_test_results


def mannwhitneyu_topo_features(topo_features_surv, topo_features_nsurv, best_features=False):
    u_test_results = []
    for i in range(40):
        _, p = mannwhitneyu(topo_features_surv[:, i], topo_features_nsurv[:, i])
        u_test_results.append(p)
    if best_features:
        return np.min(u_test_results), np.argmin(u_test_results)
    else:
        return u_test_results
