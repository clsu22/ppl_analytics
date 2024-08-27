# author: Haoyu Su
# date: 2020-06-21

"""
This script is used for machine learning pipeline

Usage: src/model_pipeline/ml_pipeline.py --train=<train> --test=<test> --output_model=<output_model> --output_results=<output_results> --output_img=<output_img>

Options:
--train=<train>  Name of training dataset, must be
within the /data directory.
--test=<test> Name of testing dataset, must be within the /data directory.
--output_model=<output_model>  Name of directory model will be saved in, no slashes necessary,
'model' folder recommended.
--output_results=<output_results> Name of directory resulting .csv files will be saved. 'model/model_results' is the recommened
--output_img=<output_img> Name of the directory the .png will be saved. img/model_results_img is recommended
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt
import joblib
from itertools import compress
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 200)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate, GridSearchCV, \
    RandomizedSearchCV
from sklearn.metrics import plot_confusion_matrix, roc_curve
from sklearn.dummy import DummyClassifier

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

# for testing
import os
import regex as re

from help_functions import show_weights, show_cv_score, format_cv_score, \
    show_test_score, make_barplot, show_probs, draw_shap_summary

opt = docopt(__doc__)


# for testing purposes
# train="../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/"
# test="../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/"
# output_model="model/"
# output_results="model/model_results/"
# output_img="img/model_result_img/"

def test_function():
    '''
    Tests the input data and specified paths.
    '''
    train = re.match("([A-Za-z]+[.]{1}[A-Za-z]+)", opt["--train"])
    output_model = re.match("([A-Za-z]+[.]{1}[A-Za-z]+)", opt["--output_model"])
    output_results = re.match("([A-Za-z]+[.]{1}[A-Za-z]+)", opt["--output_results"])
    output_img = re.match("([A-Za-z]+[.]{1}[A-Za-z]+)", opt["--output_img"])

    assert train == None, "you can not have extensions in path, only directories."
    assert output_model == None, "you can not have extensions in path, only directories."
    assert output_results == None, "you can not have extensions in path, only directories."
    assert output_img == None, "you can not have extensions in path, only directories."
    try:
        os.listdir(opt["--train"])
        os.listdir(opt["--output_model"])
        os.listdir(opt["--output_results"])
        os.listdir(opt["--output_img"])
    except Exception as e:
        print(e)


# test function runs here
test_function()


def load_data(train, test):
    """
    Load training and testing dataset

    Parameters
    ----------
    train: str
        Name of training dataset, default folder path is /data
    test: str
        Name of testing dataset, defalt folder path is /data

    Returns
    -------
    train: pandas.DataFrame
        A table containing training data
    test: pandas.DataFrame
        A table containing testing data
    """
    train = pd.read_csv(train + "manual_clean_training_dataset.csv")
    train.rename(columns={"rehired_": "rehired_flag"}, inplace=True)
    train.rehired_flag = train.rehired_flag.apply(
        lambda x: 1 if x == "Yes" else 0)

    test = pd.read_csv(test + "manual_clean_testing_dataset.csv")
    test.rename(columns={"rehired_": "rehired_flag"}, inplace=True)
    test.rehired_flag = train.rehired_flag.apply(
        lambda x: 1 if x == "Yes" else 0)

    print("Data Summary\n"
          "- There are {0} observations in the training dataset\n"
          "- There are {1} observations in the testing dataset\n"
          "===================================================\n"
          .format(train.shape[0], test.shape[0]))
    return train, test


def feature_grouping(output_model, output_results, output_img):
    """
    Group features into categories

    Parameters
    ----------
    output_model: str
        Name of model output folder to be used to save result
    output_results: str
        Name of model results output folder to be used to save result
    output_results: str
        Name of model image output folder to be used to save result

    Returns
    -------
    feat_groups_dict: dict
        A feature dictionary containing all groups of features
    categorical: list
        A list of all categorical feature names
    numerical:
        A list of all numerical feature names
    """
    # Build features dict
    feat_groups_dict = dict()
    feat_groups_dict["academic_background"] = ['accounting_concentration',
                                               'arts_concentration',
                                               'business_concentration',
                                               'computer_systems_concentration',
                                               'engineering_concentration',
                                               'finance_concentration',
                                               'general_concentration',
                                               'human_resource_concentration',
                                               'interactive_arts_and_technology_concentration',
                                               'marketing_concentration',
                                               'other_concentration',
                                               'background_highest_degree',
                                               'business_flag']
    feat_groups_dict["educational_level"] = ['highest_degree',
                                             'country_highest_degree',
                                             'flag_hd_bachelor_plus',
                                             'flag_hd_highschool']
    feat_groups_dict["internal_glentel_profile"] = ['rehired_flag',
                                                    'referral_flag']
    feat_groups_dict["job_counts"] = ['telco_electro_jobs', 'no_jobs',
                                      'no_job_categorical',
                                      'telco_electro_perc_group']
    feat_groups_dict["job_tenure_general"] = ['job_hopper',
                                              'average_tenure_per_job',
                                              'shortest_tenure',
                                              'total_experience_months',
                                              'longest_tenure']
    feat_groups_dict["job_tenure_industry"] = ['sales_exp_months',
                                               'customer_serv_exp_months',
                                               'leader_ship_exp_months']
    feat_groups_dict["knowledge_and_skills"] = ['no_lang_spoken',
                                                'trilingual_flag',
                                                'goal_record',
                                                'sales_customer_base_exp',
                                                'volunteer_exp',
                                                'problem_solver',
                                                'sports_mention',
                                                'communication_skills',
                                                'team_player',
                                                'leadership_mention']
    feat_groups_dict["readability"] = ['raw_dale_chall_readability',
                                       'clean_Flesch-Kincaid_readability',
                                       'read_score_categorical']
    feat_groups_dict["work_experience_industry"] = ['competitor_experience',
                                                    'Clothing_and_Footwear_industry_exp',
                                                    'Consumer_electronics_industry_exp',
                                                    'Food_Service_industry_exp',
                                                    'Food-Convenience-Pharmacy_industry_exp',
                                                    'Other_industry_exp',
                                                    'Sport_Travel_Enterntain_Hotel_industry_exp',
                                                    # 'Telecommunications_industry_exp'
                                                    ]
    feat_groups_dict["work_experience_industry_recency"] = [
        'recency_type_telco_electro_exp', 'telco_electro_recency']
    feat_groups_dict["work_experience_position"] = ['administrative_jobtitle',
                                                    'assistant_manager_jobtitle',
                                                    'blue_collar_jobtitle',
                                                    'cashier_jobtitle',
                                                    'cook_jobtitle',
                                                    'customer_service_representative_jobtitle',
                                                    'driver_jobtitle',
                                                    'education_jobtitle',
                                                    'financial_services_jobtitle',
                                                    'fitness_sports_jobtitle',
                                                    'manager_jobtitle',
                                                    'other_jobtitle',
                                                    'sales_associate_jobtitle',
                                                    'technicians_jobtitle',
                                                    'telemarketers_jobtitle']
    feat_groups_df = pd.DataFrame.from_dict(feat_groups_dict, orient="index").T
    feat_groups_df.to_csv(output_results + "result_features_grouping.csv")
    feat_groups = list(feat_groups_dict.keys())

    # Split numerical and categorical features
    features = []
    for values in feat_groups_dict.values():
        features.append(values)
    all_feat = sum(features, [])
    numerical = sum(features, [])

    categorical = ['background_highest_degree', 'highest_degree',
                   'country_highest_degree',
                   'recency_type_telco_electro_exp',
                   'telco_electro_perc_group', 'no_job_categorical',
                   'read_score_categorical']
    for x in categorical:
        numerical.remove(x)
    print(
        "Features Summary:\n- total groups: {3}\n  - all features: {0}\n    "
        "- numerical: {1}\n    - categorical: {2}\n"
        "===================================================\n".format(
            len(all_feat),
            len(numerical),
            len(categorical),
            len(feat_groups)))

    return feat_groups_dict, categorical, numerical


def feature_selection(X, y, X_test, y_test, output_model, output_results, output_img, numeric_features,
                      categorical_features):
    """
    Feature selection by LassoCV and RFECV

    Parameters
    ----------
    X: pandas.DataFrame
        A table containing all features in training dataset
    y: pandas.Series
        A series of target values in training dataset
    X_test: pandas.DataFrame
        A table containing all features in testing dataset
    y_test: pandas.Series
        A series of target values in testing dataset
    output_model: str
        Name of model output folder to be used to save result
    output_results: str
        Name of model results output folder to be used to save result
    output_results: str
        Name of model image output folder to be used to save result
    numeric_features: list
        A list of all numeric feature names
    categorical_features: list
        A list of all categorical feature names

    Returns
    -------
    X_trans_sel_lassocv: pandas.DataFrame
        A table containing features selected by LassoCV from training dataset
    X_test_trans_sel_lassocv
        A table containing features selected by LassoCV from testing dataset

    """
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

    # LassoCV
    selector = SelectFromModel(estimator=LassoCV())
    clf_lassocv = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('selector', selector),
                                  ])
    X_trans = pd.DataFrame(preprocessor.fit_transform(X),
                           index=X.index,
                           columns=(numeric_features +
                                    list(preprocessor.named_transformers_[
                                        'cat']['onehot']
                                        .get_feature_names(
                                        categorical_features))))

    X_sel_lassocv = clf_lassocv.fit_transform(X, y)
    X_sel_cols_lassocv = list(
        compress(X_trans.columns, selector.get_support()))
    X_trans_sel_lassocv = pd.DataFrame(X_sel_lassocv,
                                       columns=X_sel_cols_lassocv)
    print(
        "- There are {0} features after preprocessing\n- L1 selected {1} "
        "features"
            .format(X_trans.shape[1], X_sel_lassocv.shape[1]))
    feat_names = X_trans.columns
    weights = selector.estimator_.coef_.flatten()

    weights_lassocv_df = show_weights(feat_names, weights, 10)
    weights_lassocv_df.to_csv(output_results + "result_lassocv.csv")
    print("finished LassoCV")

    # RFECV
    clf_featr_sele = LogisticRegression(solver="liblinear", penalty="l1")
    rfecv = RFECV(estimator=clf_featr_sele,
                  cv=5,
                  scoring='f1'
                  )
    clf_rfecv = Pipeline(steps=[('preprocessor', preprocessor),
                                ('feature_sele', rfecv),
                                ])
    clf_rfecv.fit(X, y)
    X_sel_rfecv = X_trans.to_numpy()[:, rfecv.support_]
    X_sel_cols_rfecv = list(compress(X_trans.columns, rfecv.support_))
    X_trans_sel_rfecv = pd.DataFrame(X_sel_rfecv, columns=X_sel_cols_rfecv)
    print(
        "- There are {0} features after preprocessing\n- RFECV selected {1} "
        "features"
            .format(X_trans.shape[1], X_sel_rfecv.shape[1]))
    feat_names = X_trans.columns
    weights = rfecv.estimator_.coef_.flatten()
    weights_rfecv_df = show_weights(feat_names, weights, 10)
    weights_rfecv_df.to_csv(output_results + "result_rfecv.csv")
    print("finished RFECV\n"
          "===================================================\n")

    # Transform testing dataset by LassoCV selector
    X_test_sel_lassocv = clf_lassocv.transform(X_test)
    X_test_trans_sel_lassocv = pd.DataFrame(X_test_sel_lassocv,
                                            columns=X_sel_cols_lassocv)

    df_trans_sel = pd.concat([X.employee_code, X_trans_sel_lassocv, y], axis=1)
    df_test_trans_sel = pd.concat([
        X_test.employee_code, X_test_trans_sel_lassocv, y_test], axis=1)
    df_trans_sel.to_csv(output_results + "result_training_trans_sel.csv",
                        index=False)
    df_test_trans_sel.to_csv(output_results + "result_testing_trans_sel.csv",
                             index=False)

    return X_trans_sel_lassocv, X_test_trans_sel_lassocv


def model_selection(X_trans_sel, y, X_test_trans_sel, y_test, output_model, output_results, output_img):
    """
    Do hyperparamter tuning for ML classifiers and finalist model selection

    Parameters
    ----------
    X_trans_sel: pandas.DataFrame
        A table containing selected features from training dataset
    y: pandas.Series
        A series of target values in training dataset
    X_test_trans_sel: pandas.DataFrame
        A table containing selected features from testing dataset
    y_test: pandas.Series
        A series of target values in testing dataset
    output_model: str
        Name of model output folder to be used to save result
    output_results: str
        Name of model results output folder to be used to save result
    output_results: str
        Name of model image output folder to be used to save result


    Returns
    -------
    finalist: Model
        The finalist model

    """
    print("Start hyperparameter tuning...\n")
    # Dummy

    dummy_result = cross_validate(DummyClassifier(strategy="most_frequent"),
                                  X_trans_sel, y, cv=5,
                                  scoring=(
                                      'accuracy', 'f1', 'recall', 'precision',
                                      'roc_auc'),
                                  return_train_score=False)
    dummy_result_df = format_cv_score(dummy_result)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_trans_sel, y)

    # Logistic Regression
    print("Start Logistic Regression")
    weights = np.linspace(0.1, 0.5, 9)
    lr_param_grid = {"C": [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 10, 100],
                     'class_weight': [{0: x, 1: 1.0 - x} for x in weights],
                     }
    lr_param_grid['class_weight'].append("balanced")
    lr_gridsearchCV = GridSearchCV(estimator=LogisticRegression(),
                                   param_grid=lr_param_grid,
                                   cv=5,
                                   scoring='f1',
                                   n_jobs=2
                                   ).fit(X_trans_sel, y)
    lr_result = cross_validate(lr_gridsearchCV, X_trans_sel, y, cv=5,
                               scoring=(
                                   'accuracy', 'f1', 'recall', 'precision',
                                   'roc_auc'),
                               return_train_score=False)
    lr_result_df = format_cv_score(lr_result)
    lr_result_df.to_csv(output_results + "result_lr.csv")
    print("finished Logistic regression\n")

    # SVM RBF
    print("Start SVM")
    weights = np.linspace(0.1, 0.5, 9)
    svm_param_grid = {"C": [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 10, 100],
                      'class_weight': [{0: x, 1: 1.0 - x} for x in weights],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
                      }
    svm_param_grid['class_weight'].append("balanced")
    svm_gridsearchCV = GridSearchCV(estimator=SVC(probability=True),
                                    param_grid=svm_param_grid,
                                    cv=5,
                                    scoring='f1',
                                    n_jobs=2
                                    ).fit(X_trans_sel, y)
    svm_result = cross_validate(svm_gridsearchCV, X_trans_sel, y, cv=5,
                                scoring=(
                                    'accuracy', 'f1', 'recall', 'precision',
                                    'roc_auc'),
                                return_train_score=False)
    svm_result_df = format_cv_score(svm_result)
    print("finished SVM\n")

    # Random Forest
    print("Start Random Forest")
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    rf_param_grid = {'n_estimators': n_estimators,
                     'max_features': max_features,
                     'max_depth': max_depth,
                     'min_samples_split': min_samples_split,
                     'min_samples_leaf': min_samples_leaf,
                     'bootstrap': bootstrap,
                     'class_weight': [{0: x, 1: 1.0 - x} for x in weights]
                     }

    rf_param_grid["class_weight"].append("balanced")
    rf_randomsearchCV = RandomizedSearchCV(estimator=RandomForestClassifier(),
                                           param_distributions=rf_param_grid,
                                           n_iter=100,
                                           cv=5,
                                           random_state=1234,
                                           scoring='f1'
                                           ).fit(X_trans_sel, y)
    rf_result = cross_validate(rf_randomsearchCV, X_trans_sel, y, cv=5,
                               scoring=(
                                   'accuracy', 'f1', 'recall', 'precision',
                                   'roc_auc'),
                               return_train_score=False)
    rf_result_df = format_cv_score(rf_result)
    print("finished Random Forest\n")

    # XGBoost
    print("Start XGBoost")
    xgb_param_grid = {"learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                      "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
                      "min_child_weight": [1, 3, 5, 7],
                      "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
                      "colsample_bytree": [0.3, 0.4, 0.5, 0.7],
                      'class_weight': [{0: x, 1: 1.0 - x} for x in weights]
                      }
    xgb_randomsearchCV = RandomizedSearchCV(estimator=XGBClassifier(silent=True, verbosity=0),
                                            param_distributions=xgb_param_grid,
                                            n_iter=200,
                                            cv=5,
                                            random_state=1234,
                                            scoring='f1'
                                            ).fit(X_trans_sel, y)
    xgb_result = cross_validate(xgb_randomsearchCV, X_trans_sel, y, cv=5,
                                scoring=(
                                    'accuracy', 'f1', 'recall', 'precision',
                                    'roc_auc'),
                                return_train_score=False)
    xgb_result_df = format_cv_score(xgb_result)
    print("finished XGBoost\n")

    # LGBM
    print("Start LGBM")
    lgbm_param_grid = {'num_leaves': sp_randint(6, 50),
                       'min_child_samples': sp_randint(100, 500),
                       'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1,
                                            1e2, 1e3, 1e4],
                       'subsample': sp_uniform(loc=0.2, scale=0.8),
                       'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
                       'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                       'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
                       'class_weight': [{0: x, 1: 1.0 - x} for x in weights]
                       }
    lgbm_param_grid['class_weight'].append("balanced")
    lgbm_randomsearchCV = RandomizedSearchCV(estimator=LGBMClassifier(),
                                             param_distributions=lgbm_param_grid,
                                             n_iter=300,
                                             cv=5,
                                             random_state=1234,
                                             scoring='f1'
                                             ).fit(X_trans_sel, y)

    lgbm_result = cross_validate(lgbm_randomsearchCV, X_trans_sel, y, cv=5,
                                 scoring=(
                                     'accuracy', 'f1', 'recall', 'precision',
                                     'roc_auc'),
                                 return_train_score=False)
    lgbm_result_df = format_cv_score(lgbm_result)
    print("finished LGBM\n")

    # MLP
    print("Start MLP")
    mlp_param_grid = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }
    mlp_gridsearchCV = GridSearchCV(estimator=MLPClassifier(),
                                    param_grid=mlp_param_grid,
                                    cv=5,
                                    scoring='f1'
                                    ).fit(X_trans_sel, y)
    mlp_result = cross_validate(mlp_gridsearchCV, X_trans_sel, y, cv=5,
                                scoring=(
                                    'accuracy', 'f1', 'recall', 'precision',
                                    'roc_auc'),
                                return_train_score=False)
    mlp_result_df = format_cv_score(mlp_result)
    print("finished MLP\n")

    # Show cross-validation metrics
    model_name = pd.DataFrame({"model": ['Dummy', 'Logistic Regression', "SVM",
                                         "Random Forest", "XGBoost", "LGBM",
                                         "Multi-layer Perceptron"]})
    cv_result = (pd.concat([model_name,
                            pd.concat(
                                [dummy_result_df, lr_result_df, svm_result_df,
                                 rf_result_df, xgb_result_df, lgbm_result_df,
                                 mlp_result_df], axis=0)
                           .reset_index()], axis=1)
                 .drop(columns=["index"])).set_index("model")

    arrays = [['cross-validation', 'cross-validation', 'cross-validation',
               'cross-validation', 'cross-validation'],
              ['accuracy', 'recall', 'precision', 'roc_auc', 'f1']]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples, names=['', ''])

    cv_result.columns = index
    print("Finished hyperparameter tuning, saved cross-validation result\n")

    # Show test metrics
    print("Start prediction in testing dataset")
    dummy_test_df = show_test_score(dummy, X_test_trans_sel, y_test)
    lr_test_df = show_test_score(lr_gridsearchCV, X_test_trans_sel,
                                 y_test)
    svm_test_df = show_test_score(svm_gridsearchCV, X_test_trans_sel,
                                  y_test)
    rf_test_df = show_test_score(rf_randomsearchCV, X_test_trans_sel,
                                 y_test)
    xgb_test_df = show_test_score(xgb_randomsearchCV,
                                  X_test_trans_sel, y_test)
    lgbm_test_df = show_test_score(lgbm_randomsearchCV,
                                   X_test_trans_sel, y_test)
    mlp_test_df = show_test_score(mlp_gridsearchCV, X_test_trans_sel,
                                  y_test)

    test_result = (pd.concat([model_name,
                              pd.concat(
                                  [dummy_test_df, lr_test_df, svm_test_df,
                                   rf_test_df, xgb_test_df, lgbm_test_df,
                                   mlp_test_df], axis=0)
                             .reset_index()], axis=1)
                   .drop(columns=["index"])).set_index("model")

    test_arrays = [['test', 'test', 'test', 'test', 'test'],
                   ['accuracy', 'recall', 'precision', 'roc_auc', 'f1']]
    test_tuples = list(zip(*test_arrays))
    test_index = pd.MultiIndex.from_tuples(test_tuples, names=['', ''])

    test_result.columns = test_index
    lr_test_df.to_csv(output_results + "result_lr_test.csv")
    print("Finished prediction, saved test result\n")

    cv_test_result = pd.concat([cv_result, test_result], axis=1)
    cv_test_result.to_csv(output_results + "result_cv-test-merged.csv")
    print("Finished model selection, saved cv and test result\n")

    # Plot bar charts and error bar of avg F1 cv scores
    metric = "f1"
    make_barplot(metric, dummy_result, lr_result, svm_result, rf_result,
                 xgb_result, lgbm_result, mlp_result, output_model, output_results, output_img)
    print("saved bar plots")

    # Save unfitted and fitted finalist model
    joblib.dump(lr_gridsearchCV.best_estimator_,
                output_model + "result_unfitted_finalist.sav")
    finalist = lr_gridsearchCV.best_estimator_.fit(X_trans_sel, y)
    joblib.dump(finalist, output_model + "result_fitted_finalist.sav")
    print("Finished model selection, saved finalist model\n"
          "===================================================\n")

    return finalist


def finalist_analysis(finalist, X, X_test, X_trans_sel, y, X_test_trans_sel,
                      y_test, output_model, output_results, output_img):
    """
    Report weight tables, probabilities, draw confusion matrix plot,
    ROC plot and SHAP summary plot

    Parameters
    ----------
    finalist: sklearn.model
        The finalist model
    X: pandas.DataFrame
        A table containing all features in training dataset
    X_test: pandas.DataFrame
        A table containing all features in testing dataset
    X_trans_sel: pandas.DataFrame
        A table containing selected features from training dataset
    y: pandas.Series
        A series of target values in training dataset
    X_test_trans_sel: pandas.Series
        A table containing selected features from testing dataset
    y_test: pandas.Series
        A series of target values in testing dataset
    output_model: str
        Name of model output folder to be used to save result
    output_results: str
        Name of model results output folder to be used to save result
    output_results: str
        Name of model image output folder to be used to save result


    Returns
    -------
    None, all plot saved in folder

    """
    print("Start finalist analysis")
    # Weights
    feat_names = X_trans_sel.columns
    weights = finalist.coef_.flatten()
    weights_finalist_df = show_weights(feat_names, weights, 7)
    weights_finalist_df["Neg weights"] = weights_finalist_df[
        "Neg weights"].apply(lambda x: x if x < 0 else "")
    weights_finalist_df["Neg feats"] = weights_finalist_df["Neg feats"].apply(
        lambda x: "" if x in weights_finalist_df["Pos feats"].to_list() else x)
    weights_finalist_df.to_csv(output_results + "result_finalist_weights.csv",
                               index=False)

    # Confusion matrix plot
    disp = plot_confusion_matrix(finalist, X_test_trans_sel, y_test,
                                 display_labels=['Non-high performer',
                                                 'high performer'],
                                 cmap=plt.cm.Blues,
                                 values_format='d')
    disp.ax_.set_title(
        'Confusion matrix for high performer in testing data set')
    plt.tight_layout()
    plt.savefig(output_img + 'result_confusion_matrix_plot_test.png')

    # ROC curve plot
    plt.rc('font', size=18)
    plt.rc('axes', titlesize=16, labelsize=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('figure', titlesize=18)

    fpr, tpr, thresholds = roc_curve(y_test, finalist.predict_proba(
        X_test_trans_sel)[:, 1])
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
    plt.savefig(output_img + 'result_roc_curve_test.png')

    # Probabilities for each observations
    probs_test_df = show_probs(X_test.employee_code, y_test,
                               finalist.predict(X_test_trans_sel),
                               finalist.predict_proba(X_test_trans_sel),
                               dataset="test")
    probs_train_df = show_probs(X.employee_code, y,
                                finalist.predict(X_trans_sel),
                                finalist.predict_proba(X_trans_sel),
                                dataset="train")
    probs_train_df.to_csv(output_results + 'result_finalist_probs_train.csv')
    probs_test_df.to_csv(output_results + 'result_finalist_probs_test.csv')

    # Draw SHAP Summary plot for testing dataset
    plt.figure()
    shap_plot = draw_shap_summary(finalist, X_test_trans_sel, output_model, output_results, output_img)
    plt.title('SHAP Summary plot')
    plt.tight_layout()
    plt.savefig(output_img + "result_SHAP_summary_plot_test.png")
    print("Finished finalist analysis")


def main(train, test, output_model, output_results, output_img):
    # Load data
    train, test = load_data(train, test)

    # Feature grouping
    feat_groups_dict, categorical, numerical = feature_grouping(output_model, output_results, output_img)

    # Separate features and targets
    df = train.copy()
    numeric_features = numerical
    categorical_features = categorical
    X = df.drop(columns=["hp_class"])
    y = df["hp_class"]

    df_test = test.copy()
    X_test = df_test.drop(columns=["hp_class"])
    y_test = df_test["hp_class"]

    # Feature selection
    X_trans_sel_lassocv, X_test_trans_sel_lassocv = feature_selection(
        X,
        y,
        X_test,
        y_test,
        output_model, output_results, output_img,
        numeric_features,
        categorical_features)

    # Model selection
    finalist = model_selection(X_trans_sel_lassocv, y,
                               X_test_trans_sel_lassocv,
                               y_test, output_model, output_results, output_img)

    # finalist analysis
    finalist_analysis(finalist, X, X_test, X_trans_sel_lassocv, y,
                      X_test_trans_sel_lassocv, y_test, output_model, output_results, output_img)

    print("\n\nAll done")


# main(train, test, output_model, output_results, output_img)

if __name__ == "__main__":
    main(train=opt["--train"], test=opt["--test"], output_model=opt["--output_model"],
         output_results=opt["--output_results"], output_img=opt["--output_img"])
