import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

def plot_cdf(data, votes, bin_edges, ax, xlabel=None, color=None):
    '''
    This function is useful to understand whether the input feature "data" has
    information that is correlated to the
    satisfaction of the users.
    Generally speaking, the meaning of the CDFs is: if there is a gap between
    the distributions of the data conditioned to the satisfaction class of the
    corresponding users, it means that the information in the data is correlated
    to users satisfaction and thus can be learnt by a supervised classifier.
    Generally speaking, looking at data distributions is the first step to
    decide whether some data may be useful or not for ML problems.
    :param data: data to be plot (one-dimensional array)
    :param votes: satisfaction labels, int; if already binary, set threshold to None
    :param bin_edges: array of type  np.linspace(min(data), max(data), num_bins+1)
    :param ax: axis of type plt.subplots(figsize=(a,b));
    :param xlabel: label to gice to x axis
    :param color: axis color
    :return:
    '''

    if xlabel is None:
        xlabel = 'your data'
    if color is None:
        color = 'black'

    yt = votes.copy()

    neg, _ = np.histogram(data[yt == +1], bins=bin_edges)  # count number of evidences per bin
    pos, _ = np.histogram(data[yt == 0], bins=bin_edges)

    sumpos = sum(pos)
    sumneg = sum(neg)
    pos = pos.astype(float) / sumpos  # normalize to total number of evidences
    neg = neg.astype(float) / sumneg

    xrange = bin_edges[1:] - bin_edges[:1]

    title = 'CDF'
    ax.plot(xrange, np.cumsum(pos))
    ax.plot(xrange, np.cumsum(neg))
    ax.xaxis.label.set_color(color)
    ax.yaxis.label.set_color(color)
    ax.tick_params(axis='x', colors=color)
    ax.tick_params(axis='y', colors=color)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_title(title, color=color)
    ax.legend(['High QoE', 'Low QoE'])
    return


# USEFUL FUNCTIONS FOR PREDICTION

def hyperparameter_tuning(train_sample, train_target, names, classifiers, parameters_grid,
                          n_splits_in=None, ref_metric=None):
    '''
    This function applies a cross validation strategy to select, for each of the
    classifiers provided in input, the best hyper-parameters (hp) values out of
    a pool of candidate values (Grid Search Procedure).
    The function saves on a file the best hp values, for the input Training Fold.
    Finally, it returns the prediction performance on the input Validation Fold.
    (ref: https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/)

    :param train_sample: training samples set
    :param train_target: training users satisfaction labels
    :param test_sample: test samples set
    :param test_target: test users satisfaction labels
    :param names: involved classifiers names
    :param classifiers: involved classifiers scikitlearn functions
    :param n_splits_in: number of k fold splits for validation (our results were derived with 10 folds, which is the default value)
    :param ref_metric: optimization metric (sklearn.metrics); default roc_auc
    :return: prediction performance on the test set (AUC)
    '''

    if ref_metric is None:
        ref_metric = 'roc_auc'
    if n_splits_in is None:
        n_splits_in = 2

    best_hp = pd.DataFrame(index=names, columns=['BestHP_Values'])
    print('Choose Best hyper-parameters through Cross Validation')
    text_file = open('Best_hyper-parameters (HP Tuning).txt', "w")  # If this filename already exists in folder,
    # results will be appended to older file. Delete older version to fill a new txt file.
    text_file.write("############\n")
    for name, clf in zip(names, classifiers):
        text_file.write("{}:\n".format(name))
        print("############")
        print(' Classifier {} - Processing'.format(name))
        grid = parameters_grid[names.index(name)]  # take hyper-parameters candidate values grid
        estimator = model_selection.GridSearchCV(clf, grid, scoring=ref_metric, refit=True,
                                                 cv=n_splits_in).fit(train_sample, train_target)  # Grid Search
        bp = estimator.best_params_
        print(' Best Parameters Values: {}'.format(bp))
        print(list(bp.values()))
        best_hp.at[name, 'BestHP_Values'] = list(bp.values())
        text_file.write("{}:\n".format(estimator.best_params_))
        text_file.write("############\n")
        print("############")
    text_file.write("******************\n")
    text_file.close()

    return best_hp


def direct_prediction(train_sample, train_target, test_sample, test_target, names, classifiers):
    '''
    This function takes in input a group of classifiers with already fixed HP values, train them on the input data
    train_sample --> train_target and finally performs prediction on the input test_sample-->test_target.

    Note that each classifier outputs the probability that a given test user belongs to the
    class of Dissatisfied Users. By thresholding such probability, one can effectively assign to the test user
    either the Satisfied ('0') or the Dissatisfied ('1') label. Computing the FPR and TPR of the classifier for
    different threshold values, it is possible to draw a ROC Curve.
    Finally, the performance in terms of Area Under the ROC Curve are returned as output.
    (ref: https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)


    :param train_sample: training samples set
    :param train_target: training users satisfaction labels
    :param test_sample: test samples set
    :param test_target: test users satisfaction labels
    :param names: names of the considered classifiers
    :param classifiers: the scikit methods corresponding to the considered classifiers
    :return: prediction performance (AUC) on the test set
    '''

    perf = pd.DataFrame(index=names, columns=['AUC'])
    prediction_proba = np.empty((len(names), len(test_sample)))

    plt.figure(figsize=(20, 5))
    color = ['b', 'r', 'g', 'c', 'k', 'm']  # choose a color for each classifier
    color = color[:len(names)]
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.3)  # ROC Curve of a dummy
    # Classifier
    for name, clf in zip(names, classifiers):
        print(' Classifier {} - Fit & Predict'.format(name))
        estimator = clf.fit(train_sample, train_target)  # fit the classifier on training set

        prediction_proba[names.index(name), :] = estimator.predict_proba(test_sample)[:, 1]  # generate, for each test
        # user, the probability that the user is not satisfied

        fpr, tpr, decision_thresholds = metrics.roc_curve(test_target, prediction_proba[names.index(name), :]
                                                          , pos_label=1)

        perf.at[name, 'AUC'] = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color[names.index(name)], label=r'ROC %s (AUC = %0.3f)' % (name, perf.loc[name,
        'AUC']), lw=2, alpha=.8)
        perf.at[name, 'AUC'] = metrics.auc(fpr, tpr)

    plt.plot(0, 1, '*', color='k', label=r'Optimum: FPR = 0, TPR = 1', lw=2, alpha=.8, markersize=15)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xticks(color='black')
    plt.yticks(color='black')
    plt.grid(1)
    plt.xlabel('False Positive Rate', color='black', fontsize=14)
    plt.ylabel('True Positive Rate', color='black', fontsize=14)
    plt.legend(loc="lower right")
    plt.show()
    # plt.savefig('ROC.png', bbox_inches='tight') #uncomment to save the plot
    return perf