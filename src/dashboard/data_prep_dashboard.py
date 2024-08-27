# author: Manuel Maldonado
# date: 2020-06-05

'''This prepares the date for the dashboard visualization. 

Usage: data_prep_dashboard.py --file_path_1=<file_path_1> --file_outpath=<file_outpath=>

Example:
    python data_prep_dashboard.py --file_path_1=data/manual_clean_training_dataset.csv --file_outpath=data/

Options:
--file_path_1=<file_path_1>  Path (excluding filenames) to csv containing all features of the training dataset.
--file_outpath=<file_outpath>  Path for exporting data needed by the dashboard.
'''

import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import normalize, scale, Normalizer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
from docopt import docopt
opt = docopt(__doc__)


print("Start Dashboard Data Prep")

def test_function():
    '''
    Tests the input data and specified paths.
    '''
    file_path_check_1 = re.match("([A-Za-z]+[.]{1}[A-Za-z]+)", opt["--file_path_1"])
    out_path_check = re.match("([A-Za-z]+[.]{1}[A-Za-z]+)", opt["--file_outpath"])
    assert file_path_check_1 == None, "you can not have extensions in path, only directories."
    assert out_path_check == None, "you can not have extensions in path, only directories."
    try:
        os.listdir(opt["--file_path_1"])
        os.listdir(opt["--file_outpath"])
    except Exception as e:
        print(e)

# test function runs here
test_function()

opt = docopt(__doc__)
def main(file_path_1,file_outpath):

    ### Helper functions

    def cat_shortest_tenure(ten):
        if ten == 0:
            label = "not identified"
        elif ten <= 6:
            label = "1-6"
        elif ten <= 12:
            label = "7-11"
        elif ten > 12:
            label = ">=12"
        return label

    ### End of helper functions

    finalist_feats = ['employee_code','hp_class','flag_hd_highschool','fitness_sports_jobtitle','competitor_experience','sales_customer_base_exp','trilingual_flag','finance_concentration','shortest_tenure','cashier_jobtitle','communication_skills','recency_type_telco_electro_exp']
    numeric_features =['hp_class','flag_hd_highschool','fitness_sports_jobtitle','competitor_experience','sales_customer_base_exp','trilingual_flag','finance_concentration','shortest_tenure','cashier_jobtitle','communication_skills']
    features_keep = ['recency_type_telco_electro_exp']
    categorical_features = ['recency_type_telco_electro_exp']

    # read in data

    df= pd.read_csv(file_path_1+"manual_clean_training_dataset.csv")

    ## Limiting to only finalist feats
    df = df[finalist_feats]

    ### Transforming data

    df = df.set_index("employee_code")

    ### Transforming Pipeline

    numeric_transformer = Pipeline(steps=[
                                      ('imputer', SimpleImputer(strategy='most_frequent'))#,
                                      #('scaler', StandardScaler())
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

    ### Transforming_data
    data_transformed = pd.DataFrame(preprocessor.fit_transform(df),
                           index=df.index,
                           columns=(numeric_features +
                                      list(preprocessor.named_transformers_['cat']['onehot']
                                           .get_feature_names(categorical_features))))

    data_rebuilt = data_transformed.join(df[features_keep])
    data_rebuilt['cat_tenure'] = data_rebuilt['shortest_tenure'].apply(lambda x:cat_shortest_tenure(x) )


    #### Creating Dataframe with logistic regression coefficients
    coef_lg = pd.DataFrame({"Feature":['flag_hd_highschool','fitness_sports_jobtitle','competitor_experience','sales_customer_base_exp','trilingual_flag','finance_concentration','shortest_tenure','cashier_jobtitle','communication_skills'],
                 "Coefficient":[-0.2337,-0.2004,0.323218,0.2995,0.2517,0.1903,0.16219,0.1575,0.1308]})
    
    
    ### Exporting Result
    data_rebuilt.to_csv(file_outpath+'data_dashboard_01.csv',index = False)
    coef_lg.to_csv(file_outpath+'data_dashboard_02.csv',index = False)

if __name__ == "__main__":
    main(opt["--file_path_1"], opt["--file_outpath"])

print("Finish Dashboard Data Prep")