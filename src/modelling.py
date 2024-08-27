# author: Haoyu Su
# date: 2020-06-08

"""
This script is used for dummy model and logistic regression model

Usage: src/modelling.py --input <input> --output
<output>

Options:
--input <input>  Name of csv file containing generated features, must be
within the /data directory.
--output <output>  Name of directory to be saved in, no slashes necessary,
'results' folder recommended.

"""

import pandas as pd
import re
import datetime
import numpy as np
from docopt import docopt
import matplotlib.pyplot as plt
from itertools import compress
import string
import warnings

warnings.simplefilter('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score, \
    plot_confusion_matrix, classification_report

# from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import normalize, scale, Normalizer, \
    StandardScaler, \
    OneHotEncoder

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer

import matplotlib.pyplot as plt

pd.set_option("display.max_colwidth", 200)
from sklearn.feature_selection import RFECV

from sklearn.metrics import make_scorer, precision_score, f1_score, \
    accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import roc_auc_score, roc_curve

import shap

shap.initjs()

opt = docopt(__doc__)


def show_scores(model, X_train, y_train, X_test, y_test, show=True):
    """
    Calculate the scores
    """
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    score_result = {"accuracy": [], "recall": [], "precision": [],
                    "roc_auc_score": [], "f1_score": []}
    score_result["accuracy"].append(
        round(accuracy_score(y_train, y_pred_train), 2))
    score_result["accuracy"].append(
        round(accuracy_score(y_test, y_pred_test), 2))
    score_result["recall"].append(
        round(recall_score(y_train, y_pred_train), 2))
    score_result["recall"].append(round(recall_score(y_test, y_pred_test), 2))
    score_result["precision"].append(
        round(precision_score(y_train, y_pred_train), 2))
    score_result["precision"].append(
        round(precision_score(y_test, y_pred_test), 2))
    score_result["roc_auc_score"].append(
        round(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]), 2))
    score_result["roc_auc_score"].append(
        round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 2))
    score_result["f1_score"].append(round(f1_score(y_train, y_pred_train), 2))
    score_result["f1_score"].append(round(f1_score(y_test, y_pred_test), 2))
    if show:
        score_result = pd.DataFrame(score_result, index=["train", "valid"])
        return score_result
    else:
        return


def show_weights(feat_names, weights, num):
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


def show_probs(employee_code, y_obs, y_pred, probs, dataset="train"):
    """
    Show probabilities
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
    elif dataset == "valid":
        df1 = pd.DataFrame(
            {"employee_code": employee_code, "y_obs_valid": y_obs,
             "y_pred_valid": y_pred}).reset_index(drop=True)
        df2 = pd.DataFrame(columns=["non-high performer", "high performer"],
                           data=probs)
        df_prob = pd.concat([df1, df2], axis=1)
        df_prob["False_positive"] = df_prob.apply(lambda
                                                      row: 1 if
        row.y_obs_valid == 0 and row.y_pred_valid == 1 else 0,
                                                  axis=1)
        df_prob["False_negative"] = df_prob.apply(lambda
                                                      row: 1 if
        row.y_obs_valid == 1 and row.y_pred_valid == 0 else 0,
                                                  axis=1)
    return df_prob


def main(input, output):
    # Read in data
    df = pd.read_csv(f"./data/{input}")

    # Split data
    X_train = df.drop(columns=["hp_class"])
    y_train = df["hp_class"]
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                          train_size=0.8,
                                                          random_state=1234)

    numeric_features = ['referral_flag',
                        "job_hopper",  # avg_work
                        'competitor_experience',
                        'accounting_concentration',
                        'arts_concentration', 'business_concentration',
                        'computer_systems_concentration',
                        'engineering_concentration',
                        'finance_concentration', 'general_concentration',
                        'human_resource_concentration',
                        'interactive_arts_and_technology_concentration',
                        'marketing_concentration',
                        'other_concentration',  # education concentration
                        'administrative', 'assistant manager', 'blue collar',
                        'cashier', 'cook',
                        'customer service representative', 'driver',
                        'education',
                        'financial services', 'fitness/sports', 'manager',
                        'other',
                        'sales associate', 'technicians', 'telemarketers',
                        'sales_exp', 'leader_ship_exp', 'customer_serv_exp',
                        'raw_date_chall_readability',  # communication level
                        'food_service_industry_exp',
                        'apparel_industry_exp',
                        'supercenter_convenience_industry_exp',
                        'automotive_sales_industry_exp',
                        'blue_collar_industry_exp',
                        'consumer_electronics',
                        'trilingual_flag', 'goal_record',
                        'sales_customer_base_exp', 'volunteer_exp',
                        'problem_solver',
                        'sports_mention', 'communication_skills',
                        'team_player',
                        'leadership_mention']

    categorical_features = ['rehired_', 'highest_degree',
                            'country_highest_degree',
                            'background_highest_degree']

    # Preprocess
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant',
                                  fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # dummy baseline
    print('Fitting dummy model...')
    dummy = DummyClassifier()

    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', dummy)])

    clf.fit(X_train, y_train)

    dummy_result = show_scores(clf, X_train, y_train, X_valid, y_valid)
    dummy_result.to_csv(f"./{output}/dummy_result.csv")
    print("Finished dummy model, output: dummy_result.csv")
    print("-------\n\n")

    # Logistic regression
    print('Fitting Logistic regression...')
    clf_featr_sele = LogisticRegression(solver='liblinear', penalty="l1")
    rfecv = RFECV(estimator=clf_featr_sele,
                  step=1,
                  cv=5,
                  scoring='f1'
                  )

    weights = np.linspace(0.1, 0.5, 9)
    lg_param_grid = {"C": [0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 10],
                     'class_weight': [{0: x, 1: 1.0 - x} for x in weights],
                     'l1_ratio': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                  0.9, 1.0],
                     'penalty': ["l1", "l2"]
                     }
    CV_rfc = GridSearchCV(clf_featr_sele,
                          lg_param_grid,
                          cv=5, scoring='f1')

    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('feature_sele', rfecv),
                          ('clf_cv', CV_rfc)])

    clf.fit(X_train, y_train);

    # Plot confusion matrix plot for training dataset
    disp = plot_confusion_matrix(clf, X_train, y_train,
                                 display_labels=['Non-high performer',
                                                 'high performer'],
                                 cmap=plt.cm.Blues,
                                 values_format='d')
    disp.ax_.set_title('Confusion matrix for high performer training data set')
    plt.tight_layout()
    plt.savefig(f'./{output}/lr_confusion_matrix_plot_train.png')

    # Plot confusion matrix plot for validation dataset
    disp = plot_confusion_matrix(clf, X_valid, y_valid,
                                 display_labels=['Non-high performer',
                                                 'high performer'],
                                 cmap=plt.cm.Blues,
                                 values_format='d')
    disp.ax_.set_title(
        'Confusion matrix for high performer validation data set')
    plt.tight_layout()
    plt.savefig(f'./{output}/lr_confusion_matrix_plot_valid.png')

    # Plot a roc curve
    plt.rc('font', size=18)
    plt.rc('axes', titlesize=16, labelsize=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('figure', titlesize=18)

    fpr, tpr, thresholds = roc_curve(y_valid, clf.predict_proba(X_valid)[:, 1])
    fig, ax = plt.subplots(nrows=1)
    plt.plot(fpr, tpr);
    plt.title('ROC report')
    plt.plot((0, 1), (0, 1), '--k');
    plt.xlabel('False positive rate');
    plt.ylabel('True positive rate');
    plt.tight_layout()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.tight_layout()
    plt.savefig(f'./{output}/roc_curve_validation.png')

    # show metrics
    lr_result = show_scores(clf, X_train, y_train, X_valid, y_valid)
    lr_result.to_csv(f"./{output}/lr_result.csv")

    # Take out the training dataset after preprocessing
    X_train_trans = pd.DataFrame(preprocessor.fit_transform(X_train),
                                 index=X_train.index,
                                 columns=(numeric_features +
                                          list(
                                              preprocessor.named_transformers_[
                                                  'cat']['onehot']
                                                  .get_feature_names(
                                                  categorical_features))))
    X_valid_trans = pd.DataFrame(preprocessor.fit(X_train).transform(X_valid),
                                 index=X_valid.index,
                                 columns=(numeric_features +
                                          list(
                                              preprocessor.named_transformers_[
                                                  'cat']['onehot']
                                                  .get_feature_names(
                                                  categorical_features))))

    X_train_sel = X_train_trans.to_numpy()[:, rfecv.support_]
    X_valid_sel = X_valid_trans.to_numpy()[:, rfecv.support_]
    X_train_sel_cols = list(compress(X_train_trans.columns, rfecv.support_))

    # Refit the best model
    lr = LogisticRegression(solver='liblinear', C=CV_rfc.best_params_["C"],
                            class_weight=CV_rfc.best_params_["class_weight"],
                            l1_ratio=CV_rfc.best_params_["l1_ratio"],
                            penalty=CV_rfc.best_params_["penalty"])
    lr.fit(X_train_sel, y_train);

    print("Summary:")
    print(
        "- before feature selection, after preprocessing, there are %d "
        "features" % (
            X_train_trans.shape[1]))
    print(
        "- after feature selection, after preprocessing, there are %d "
        "features" % (
            len(X_train_sel_cols)))

    # Show feature weights
    feat_names = X_train_sel_cols
    weights = lr.coef_.flatten()
    top20_feat_weights = show_weights(feat_names, weights, 10)
    top20_feat_weights.to_csv(f"./{output}/lr_top20_feat_weights.csv")

    # Show probabilities for training set
    lr_probs_train = show_probs(X_train.employee_code, y_train,
                                lr.predict(X_train_sel),
                                lr.predict_proba(X_train_sel))
    lr_probs_train.to_csv(f"./{output}/lr_probs_train.csv", index=False)

    # Show probabilities for validation set
    lr_probs_valid = show_probs(X_valid.employee_code, y_valid,
                                lr.predict(X_valid_sel),
                                lr.predict_proba(X_valid_sel), dataset="valid")
    lr_probs_valid.to_csv(f"./{output}/lr_probs_valid.csv", index=False)

    # Show false negative
    false_negative_valid = lr_probs_valid[lr_probs_valid.False_negative == 1]
    false_negative_valid.to_csv(f"./{output}/false_negative_valid.csv",
                                index=False)

    plt.savefig(f"./{output}/false_negative_valid.png")

    ## Explain the Linear model
    plt.rc('font', size=18)
    plt.rc('axes', titlesize=16, labelsize=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('figure', titlesize=18)

    explainer = shap.LinearExplainer(lr, X_train_sel,
                                     feature_dependence="independent")
    shap_values = explainer.shap_values(X_valid_sel)
    X_valid_array = X_valid_sel  # we need to pass a dense version for the plotting functions

    fig = shap.summary_plot(shap_values, X_valid_array,
                      feature_names=X_train_sel_cols, show=False)
    plt.title('SHAP Summary plot')
    plt.tight_layout()
    plt.savefig(f"./{output}/SHAP_summary_plot.png")

    plt.rc('font', size=18)
    plt.rc('axes', titlesize=16, labelsize=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('figure', titlesize=18)

    ind1 = false_negative_valid.index[0]
    fig = shap.force_plot(
        explainer.expected_value, shap_values[ind1, :], X_valid_array[ind1, :],
        feature_names=X_train_sel_cols)
    plt.savefig(f"./{output}/SHAP_false_negatives.png")


if __name__ == "__main__":
    main(input=opt["--input"], output=opt["--output"])
