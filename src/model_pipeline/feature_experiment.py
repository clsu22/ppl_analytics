# author: Haoyu Su
# date: 2020-06-21

"""
This script is used for feature experiment

Usage: src/feature_experiment.py --train=<train> --test=<test> --finalist=<finalist> --output=<output>

Options:
--train=<train>  Name of training dataset, must be
within the /data directory.
--test=<test> Name of testing dataset, must be within the /data directory.
--finalist=<finalist> Name of unfitted finalist model, must be within /result
directory
--output=<output>  Name of directory to be saved in, no slashes necessary,
'results' folder recommended.

"""
import pandas as pd
import numpy as np
from docopt import docopt
import joblib
from sklearn.feature_extraction.text import CountVectorizer, \
    TfidfVectorizer
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LassoCV
from IPython.display import clear_output
from help_functions import show_weights, show_cv_score, format_cv_score, \
    show_test_score, make_barplot

import os
import regex as re

opt = docopt(__doc__)




#for testing
# train = "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/"
# test = "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/"
# finalist = "model/"
# output = "model/model_results/"

def test_function():
    '''
    Tests the input data and specified paths.
    '''
    train = re.match("([A-Za-z]+[.]{1}[A-Za-z]+)", opt["--train"])
    finalist = re.match("([A-Za-z]+[.]{1}[A-Za-z]+)", opt["--finalist"])
    output = re.match("([A-Za-z]+[.]{1}[A-Za-z]+)", opt["--output"])

    assert train == None, "you can not have extensions in path, only directories."
    assert finalist == None, "you can not have extensions in path, only directories."
    assert output == None, "you can not have extensions in path, only directories."
    try:
        os.listdir(opt["--train"])
        os.listdir(opt["--finalist"])
        os.listdir(opt["--output"])
    except Exception as e:
        print(e)

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
    train = pd.read_csv(train+"manual_clean_training_dataset.csv")
    train.rename(columns={"rehired_": "rehired_flag"}, inplace=True)
    train.rehired_flag = train.rehired_flag.apply(
        lambda x: 1 if x == "Yes" else 0)

    test = pd.read_csv(test+"manual_clean_testing_dataset.csv")
    test.rename(columns={"rehired_": "rehired_flag"}, inplace=True)
    test.rehired_flag = train.rehired_flag.apply(
        lambda x: 1 if x == "Yes" else 0)

    print("Data Summary\n"
          "- There are {0} observations in the training dataset\n"
          "- There are {1} observations in the testing dataset\n"
          "===================================================\n"
          .format(train.shape[0], test.shape[0]))
    return train, test


def feature_grouping(output):
    """
    Group features into categories

    Parameters
    ----------
    output: str
        Name of folder to be used to save result

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

    return feat_groups_dict, categorical, numerical, feat_groups


def main(train, test, finalist, output):
    # Load finalist model
    unfitted_finalist = joblib.load(finalist+"result_unfitted_finalist.sav")
    lr_result_df = pd.read_csv(output+"result_lr.csv", index_col=0)
    lr_test_df = pd.read_csv(output+"result_lr_test.csv", index_col=0)

    # Load data
    train, test = load_data(train, test)
    feat_groups_dict, categorical, numerical, feat_groups = feature_grouping(
        output)
    df = train.copy()

    X = df.drop(columns=["hp_class"])
    y = df["hp_class"]

    df_test = test.copy()
    X_test = df_test.drop(columns=["hp_class"])
    y_test = df_test["hp_class"]

    # Baseline model
    print("Start baseline models...")
    # Bag-of-Word
    X_wc = train["clean_text"]
    y_wc = train["hp_class"]

    X_test_wc = test["clean_text"]
    y_test_wc = test["hp_class"]

    vec = CountVectorizer(max_features=5000, ngram_range=(2, 2))
    X_counts = vec.fit_transform(X_wc)
    X_test_counts = vec.transform(X_test_wc)

    weights = np.linspace(0.1, 0.5, 9)
    lr_param_grid = {"C": [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 10, 100],
                     'class_weight': [{0: x, 1: 1.0 - x} for x in weights],
                     }
    lr_param_grid['class_weight'].append("balanced")

    wc_lr_gridsearchCV = GridSearchCV(estimator=LogisticRegression(),
                                      param_grid=lr_param_grid,
                                      cv=5,
                                      scoring='f1',
                                      n_jobs=2
                                      ).fit(X_counts, y_wc)

    wc_lr_result = cross_validate(wc_lr_gridsearchCV, X_counts, y_wc, cv=5,
                                  scoring=(
                                      'accuracy', 'f1', 'recall', 'precision',
                                      'roc_auc'),
                                  return_train_score=False)
    wc_lr_result_df = format_cv_score(wc_lr_result)
    wc_lr_test_result = show_test_score(wc_lr_gridsearchCV, X_test_counts,
                                        y_test_wc)

    # TF-IDF
    X_tfidf = train["clean_text"]
    y_tfidf = train["hp_class"]

    X_test_tfidf = test["clean_text"]
    y_test_tfidf = test["hp_class"]

    vec2 = TfidfVectorizer(max_features=5000, ngram_range=(2, 2))
    X_tfidf = vec2.fit_transform(X_tfidf)
    X_test_tfidf = vec2.transform(X_test_tfidf)

    tfidf_lr_gridsearchCV = GridSearchCV(estimator=LogisticRegression(),
                                         param_grid=lr_param_grid,
                                         cv=5,
                                         scoring='f1',
                                         n_jobs=2
                                         ).fit(X_tfidf, y_tfidf)
    tfidf_lr_result = cross_validate(tfidf_lr_gridsearchCV, X_tfidf, y_tfidf,
                                     cv=5,
                                     scoring=(
                                         'accuracy', 'f1', 'recall',
                                         'precision',
                                         'roc_auc'),
                                     return_train_score=False)
    tfidf_lr_result_df = format_cv_score(tfidf_lr_result)
    tfidf_lr_test_result = show_test_score(tfidf_lr_gridsearchCV,
                                           X_test_counts, y_test_wc)
    print("Finished baseline "
          "models\n""===================================================\n")

    # Feature experiment
    print("Start feature experiment")
    feat_expr_result = dict()
    feat_expr_test_result = dict()
    for i in range(len(feat_groups)):
        clear_output(wait=True)
        cat_feat_cur = categorical.copy()
        num_feat_cur = numerical.copy()
        for j in range(len(feat_groups_dict[feat_groups[i]])):
            if feat_groups_dict[feat_groups[i]][j] in categorical:
                cat_feat_cur.remove(feat_groups_dict[feat_groups[i]][j])
            else:
                num_feat_cur.remove(feat_groups_dict[feat_groups[i]][j])

        # run logitstic regression with the best hyperparameter recursively
        selector2 = SelectFromModel(estimator=LassoCV())

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant',
                                      fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor2 = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_feat_cur),
                ('cat', categorical_transformer, cat_feat_cur)
            ])

        X_trans_sel_expr = selector2.fit_transform(
            preprocessor2.fit_transform(X), y)
        X_test_sel_expr = selector2.transform(preprocessor2.transform(X_test))

        lr_gridsearchCV = GridSearchCV(estimator=LogisticRegression(),
                                       param_grid=lr_param_grid,
                                       cv=5,
                                       scoring='f1',
                                       n_jobs=2
                                       )
        lr_result_expr = cross_validate(lr_gridsearchCV, X_trans_sel_expr, y,
                                        cv=5,
                                        scoring=('accuracy', 'f1', 'recall',
                                                 'precision', 'roc_auc'),
                                        return_train_score=False)
        lr_result_df_expr = format_cv_score(lr_result_expr)

        lr_decided = unfitted_finalist.fit(X_trans_sel_expr, y)

        lr_test_result_expr = show_test_score(lr_decided, X_test_sel_expr,
                                              y_test)

        feat_expr_result[feat_groups[i]] = lr_result_df_expr
        feat_expr_test_result[feat_groups[i]] = lr_test_result_expr

        print("Current progress:", i + 1, "/", len(feat_groups),
              np.round((i + 1) / len(feat_groups) * 100, 2), "%")

    # cross_validation
    arrays = [['cross-validation', 'cross-validation', 'cross-validation',
               'cross-validation', 'cross-validation'],
              ['accuracy', 'recall', 'precision', 'roc_auc', 'f1']]
    tuples = list(zip(*arrays))
    columns = pd.MultiIndex.from_tuples(tuples, names=['', ''])
    feat_index = ["all features"] + ["- " + s for s in
                                     list(feat_expr_result.keys())]
    lr_df = pd.concat(
        [lr_result_df, pd.concat(list(feat_expr_result.values()))])
    lr_df.index = feat_index

    lr_df2 = pd.concat([wc_lr_result_df, tfidf_lr_result_df, lr_df])
    feat_index2 = ["Baseline 1: bag-of-word with lr",
                   "Baseline 2: tf-idf with lr", "all features"] + \
                  ["- " + s for s in list(feat_expr_result.keys())]
    lr_df2.index = feat_index2
    lr_df2.columns = columns

    # test
    test_arrays = [['test', 'test', 'test', 'test', 'test'],
                   ['accuracy', 'recall', 'precision', 'roc_auc', 'f1']]
    test_tuples = list(zip(*test_arrays))
    test_columns = pd.MultiIndex.from_tuples(test_tuples, names=['', ''])
    feat_index = ["all features"] + ["- " + s for s in
                                     list(feat_expr_result.keys())]
    lr_df_test = pd.concat(
        [lr_test_df, pd.concat(list(feat_expr_test_result.values()))])
    lr_df.index = feat_index

    lr_df2_test = pd.concat(
        [wc_lr_test_result, tfidf_lr_test_result, lr_df_test])
    feat_index2 = ["Baseline 1: bag-of-word with lr",
                   "Baseline 2: tf-idf with lr", "all features"] + \
                  ["- " + s for s in list(feat_expr_result.keys())]
    lr_df2_test.index = feat_index2
    lr_df2_test.columns = test_columns

    feat_expr_final_table_lr = pd.concat([lr_df2, lr_df2_test], axis=1)
    feat_expr_final_table_lr.to_csv(output+"result_feat_experiment.csv")
    print("Finished feature experiment")


# main(train, test, finalist, output)

if __name__ == "__main__":
    main(train=opt["--train"], test=opt["--test"], finalist=opt["--finalist"],
         output=opt["--output"])
