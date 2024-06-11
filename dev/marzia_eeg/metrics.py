# same but with a loop for all the frequencies
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind


def auc_ppv_score(topo_features_surv, topo_features_nsurv, method='tpr-fpr', freq_list = np.arange(1,41), plot=False):
    aucs = []
    ppvs = []
    sensitivities = []
    specificities = []
    npvs = []
    accuracies = []

    n_it_freq = len(freq_list) if freq_list is not None else 1 # 1 if only testibng one freq or one band
    for i in range(n_it_freq):
        # if topo feat is 2D then we flatten it else we keep it as it is
        if topo_features_surv.ndim == 2:
            topo_feature_freq_i_surv = topo_features_surv[:, i].flatten()
            topo_feature_freq_i_nsurv = topo_features_nsurv[:, i].flatten()
        else:
            topo_feature_freq_i_surv = topo_features_surv
            topo_feature_freq_i_nsurv = topo_features_nsurv
        topo_feature_freq_i = np.concatenate((-topo_feature_freq_i_surv, -topo_feature_freq_i_nsurv))
        labels = np.concatenate((np.ones_like(topo_feature_freq_i_surv), np.zeros_like(topo_feature_freq_i_nsurv)))
        fpr, tpr, thresholds = roc_curve(labels, topo_feature_freq_i)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        precision, recall, thresholds_pr = precision_recall_curve(labels, topo_feature_freq_i)

        if method == 'tpr-fpr':
            optimal_idx = np.argmax(tpr - fpr)
        elif method == 'f1':
            f1_scores = [f1_score(labels, topo_feature_freq_i >= thresh) for thresh in thresholds_pr]
            optimal_idx = np.argmax(f1_scores)
            thresholds = thresholds_pr
        elif method == 'auprc':
            min_recall = 0.80
            valid_indices = np.where(recall[1:] >= min_recall)[0]
            if len(valid_indices) > 0:
                high_recall_precisions = precision[valid_indices]
                optimal_idx = np.argmax(high_recall_precisions)
                thresholds = thresholds_pr[valid_indices]
            else:
                optimal_idx = 0
        else:
            raise ValueError("Method must be 'tpr-fpr', 'f1', or 'auprc'")

        optimal_threshold = thresholds[optimal_idx]
        predictions = topo_feature_freq_i >= optimal_threshold
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        ppvs.append(ppv)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        npvs.append(npv)
        accuracies.append(accuracy)

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

    return aucs, ppvs, npvs, sensitivities, specificities, accuracies


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
