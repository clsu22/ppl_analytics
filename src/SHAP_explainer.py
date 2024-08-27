# author: Haoyu Su
# date: 2020-06-21

"""
This script is for SHAP explainer for finalist model
"""

import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

shap.initjs()


def draw_shap_summary(finalist, X):
    explainer = shap.LinearExplainer(finalist, X,
                                     feature_dependence="independent")
    shap_values = explainer.shap_values(X)
    X_test_array = X

    shap_summary_plot = shap.summary_plot(shap_values, X_test_array,
                                          show=False)
    plt.title('SHAP Summary plot')
    plt.tight_layout()
    plt.savefig(f"./result/result_SHAP_summary_plot_test.png")


def main():
    finalist = joblib.load(f"./result/result_fitted_finalist.sav")
    trans_sel_train = pd.read_csv(f"./result/result_training_trans_sel.csv")
    trans_sel_test = pd.read_csv(f"./result/result_testing_trans_sel.csv")

    X_trans_sel_train = trans_sel_train.drop(columns=["employee_code",
                                                      "hp_class"])
    y_train = trans_sel_train["hp_class"]

    X_trans_sel_test = trans_sel_train.drop(columns=["employee_code",
                                                     "hp_class"])
    y_test = trans_sel_test["hp_class"]



    draw_shap_summary(finalist, X_trans_sel_test)


if __name__ == "__main__":
    main()
