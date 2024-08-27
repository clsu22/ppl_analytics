# author: Haoyu Su
# date: 2020-06-21

"""
This script is for helper functions, all functions are called directly by
ml_pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, f1_score, accuracy_score, \
    recall_score, roc_auc_score
import shap

shap.initjs()


def show_weights(feat_names, weights, num):
    """
    Show weights of Logistic Regression result

    Parameters
    ----------
    feat_names: list
        A list of feature names
    weights: list
        A list of coefficients of features
    num: int
        The row number

    Returns
    -------
    df: pandas.DataFrame
        A table containing sorted features' coefficient

    """
    # feat_names #= X_train_sel_cols
    # weights #= lr.coef_.flatten()

    # Sort the coefficients in descending order
    inds = np.argsort(weights)

    # pick the first 20 as most informative features for negative reviews
    negative_words = [feat_names[index] for index in inds[:num]]

    # pick the last 20 features as most informative features for positive
    # reviews
    positive_words = [feat_names[index] for index in inds[-num:][::-1]]

    neg_words_weights = [(weights[index]) for index in inds[:num]]
    pos_words_weights = [(weights[index]) for index in inds[-num:][::-1]]

    df = pd.DataFrame(
        {'Neg feats': negative_words, 'Neg weights': neg_words_weights,
         'Pos feats': positive_words, 'Pos weights': pos_words_weights})
    return df


def show_cv_score(model, X, y):
    """
    Calculate cross-validation scores

    Parameters
    ----------
    model: sklearn.model
        A ML classifier
    X: pandas.DataFrame
        A table containing all features in training dataset
    y: pandas.Series
        A series of target values in training dataset

    Returns
    -------
    score_result: dict
        A dictionary containing cross-validation scores
    """
    score_result = {"accuracy": [], "recall": [], "precision": [],
                    "roc_auc": [], "f1_score": []}

    score_result["accuracy"].append(
        round(np.mean(cross_val_score(model, X, y, scoring="accuracy", cv=5)),
              3))
    score_result["recall"].append(
        round(np.mean(cross_val_score(model, X, y, scoring="recall", cv=5)),
              3))
    score_result["precision"].append(
        round(np.mean(cross_val_score(model, X, y, scoring="precision", cv=5)),
              3))
    score_result["roc_auc"].append(
        round(np.mean(cross_val_score(model, X, y, scoring="roc_auc", cv=5)),
              3))
    score_result["f1_score"].append(
        round(np.mean(cross_val_score(model, X, y, scoring="f1", cv=5)), 3))
    score_result = pd.DataFrame(score_result, index=["cross-validation"])

    return score_result


def format_cv_score(result_dict):
    """
    Transform cross_validate result from dict to dataframe

    Parameters
    ----------
    result_dict: dict
        A dict return be cross_validate function

    Returns
    -------
    score_result: pandas.DataFrame
        A table containing all cross-validation scores
    """
    score_result = {"accuracy": [], "recall": [], "precision": [],
                    "roc_auc_score": [], "f1_score": []}

    score_result["accuracy"].append(
        round(np.mean(result_dict['test_accuracy']), 3))
    score_result["recall"].append(
        round(np.mean(result_dict['test_recall']), 3))
    score_result["precision"].append(
        round(np.mean(result_dict['test_precision']), 3))
    score_result["roc_auc_score"].append(
        round(np.mean(result_dict['test_roc_auc']), 3))
    score_result["f1_score"].append(round(np.mean(result_dict['test_f1']), 3))

    score_result = pd.DataFrame(score_result, index=["cross-validation"])
    return score_result


def show_test_score(model, X_test, y_test):
    """
    Show test scores for model

    Parameters
    ----------
    model: sklearn.model
        A ML classifier
    X_test: pandas.DataFrame
        A table containing all features in testing dataset
    y_test: pandas.Series
        A series of target values in testing dataset

    Returns
    -------
    score_result: pandas.DataFrame
        A table containing all test scores
    """
    y_pred_test = model.predict(X_test)
    score_result = {"accuracy": [], "recall": [], "precision": [],
                    "roc_auc_score": [], "f1_score": []}
    score_result["accuracy"].append(
        round(accuracy_score(y_test, y_pred_test), 3))
    score_result["recall"].append(round(recall_score(y_test, y_pred_test), 3))
    score_result["precision"].append(
        round(precision_score(y_test, y_pred_test), 3))
    score_result["roc_auc_score"].append(
        round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 3))
    score_result["f1_score"].append(round(f1_score(y_test, y_pred_test), 3))

    score_result = pd.DataFrame(score_result, index=["test"])
    return score_result


def make_barplot(metric, dummy_result, lr_result, svm_result, rf_result,
                 xgb_result, lgbm_result, mlp_result, output):
    """

    Parameters
    ----------
    metric: str
        Options: 'accuracy', 'f1', 'recall', 'precision', 'roc_auc'
    dummy_result: dict
        Dummy result from cross_validate
    lr_result: dict
        Logistic Regression result from cross_validate
    svm_result: dict
        SVM result from cross_validate
    rf_result: dict
        Random Forest result from cross_validate
    xgb_result: dict
        XGBoosting result from cross_validate
    lgbm_result: dict
        LGBM result from cross_validate
    mlp_result: dict
        MLP result from cross_validate
    output: str
        Name of folder to be used to save result

    Returns
    -------
        None, bar plots saved in folder
    """
    dummy = dummy_result['test_' + metric]
    lr = lr_result['test_' + metric]
    svm = svm_result['test_' + metric]
    rf = rf_result['test_' + metric]
    xgb = xgb_result['test_' + metric]
    lgbm = lgbm_result['test_' + metric]
    mlp = mlp_result['test_' + metric]

    dummy_mean, dummy_std = np.mean(dummy), np.std(dummy)
    lr_mean, lr_std = np.mean(lr), np.std(lr)
    svm_mean, svm_std = np.mean(svm), np.std(svm)
    rf_mean, rf_std = np.mean(rf), np.std(rf)
    xgb_mean, xgb_std = np.mean(xgb), np.std(xgb)
    lgbm_mean, lgbm_std = np.mean(lgbm), np.std(lgbm)
    mlp_mean, mlp_std = np.mean(mlp), np.std(mlp)

    labels = ['Dummy', 'Logistic Regression', 'SVM', 'Random Forest',
              "XGBoost", "LGBM", "MLP"]
    x_pos = np.arange(len(labels))
    CTEs = [dummy_mean, lr_mean, svm_mean, rf_mean, xgb_mean, lgbm_mean,
            mlp_mean]
    error = [dummy_std, lr_std, svm_std, rf_std, xgb_std, lgbm_std, mlp_std]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x_pos, CTEs,
           yerr=error,
           align='center',
           alpha=0.5,
           ecolor='black',
           capsize=10)
    ax.set_ylabel(metric)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title(
        metric + ' score of different ML classifiers (Cross-validation)')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig(f"./{output}/result_bar_plot_with_error_bars.png")


def show_probs(employee_code, y_obs, y_pred, probs, dataset="train"):
    """
    show probabilities of all observations

    Parameters
    ----------
    employee_code: pandas.Series
        A series containing all employee codes
    y_obs: numpy.array
        An array containing all observed target values
    y_pred: numpy.array
        An array containing all predicted target values
    probs: numpy.array
        An array containing all probabilitis
    dataset: str
        option: "train", "test", default is "train"

    Returns
    -------
    df_prob: pandas.DataFrame
        A table containing probabilities

    """
    if dataset == "train":
        df1 = pd.DataFrame(
            {"employee_code": employee_code, "y_obs_train": y_obs,
             "y_pred_train": y_pred}).reset_index(drop=True)
        df2 = pd.DataFrame(columns=["non-high performer", "high performer"],
                           data=probs)
        df_prob = pd.concat([df1, df2], axis=1)
        df_prob["False_positive"] = df_prob.apply(lambda
                                                      row: 1 if
        row.y_obs_train == 0 and row.y_pred_train == 1 else 0,
                                                  axis=1)
        df_prob["False_negative"] = df_prob.apply(lambda
                                                      row: 1 if
        row.y_obs_train == 1 and row.y_pred_train == 0 else 0,
                                                  axis=1)
    elif dataset == "test":
        df1 = pd.DataFrame(
            {"employee_code": employee_code, "y_obs_test": y_obs,
             "y_pred_test": y_pred}).reset_index(drop=True)
        df2 = pd.DataFrame(columns=["non-high performer", "high performer"],
                           data=probs)
        df_prob = pd.concat([df1, df2], axis=1)
        df_prob["False_positive"] = df_prob.apply(lambda
                                                      row: 1 if
        row.y_obs_test == 0 and row.y_pred_test == 1 else 0,
                                                  axis=1)
        df_prob["False_negative"] = df_prob.apply(lambda
                                                      row: 1 if
        row.y_obs_test == 1 and row.y_pred_test == 0 else 0,
                                                  axis=1)
    return df_prob


def draw_shap_summary(finalist, X, output):
    """
    Draw SHAP summary plot

    Parameters
    ----------
    finalist: sklearn.model
        The finalist model
    X: pandas.DataFrame
        A table containing all features
    output: str
        Name of folder to be used to save result

    Returns
    -------
    shap_summary_plot:
     A SHAP summary plot

    """
    explainer = shap.LinearExplainer(finalist, X,
                                     feature_dependence="independent")
    shap_values = explainer.shap_values(X)
    X_test_array = X

    shap_summary_plot = shap.summary_plot(shap_values, X_test_array,
                                          show=False)
    return shap_summary_plot


if __name__ == "__main__":
    main()
