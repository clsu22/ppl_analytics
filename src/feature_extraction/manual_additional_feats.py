# author: Manuel Maldonado
# date: 2020-06-05

'''This script generates the following features: "total_experience_months","longest_tenure","shortest_tenure","average_tenure_per_job","job_hopper","no_jobs",

Usage: additional_feats.py --file_path=<file_path> --file_outpath=<file_outpath> --mode=<mode>

Example:
    python scripts/additional_feats.py --file_path=data/ --file_outpath=data/

Options:
--file_path=<file_path>  Path (excluding filenames) to csv file containing features.
--file_outpath=<file_outpath>  Path for dataframe with created features.
--mode=<mode> Set to train and test
'''

import pandas as pd
import numpy as np
import re
import os
from docopt import docopt
opt = docopt(__doc__)

#for testing
# file_path= '../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_raw_data/'
# file_outpath= '../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/'

print("Start addition Feature Extraction Process")

def test_function():
    '''
    Tests the input data and specified paths.
    '''
    file_path_check_1 = re.match("([A-Za-z]+[.]{1}[A-Za-z]+)", opt["--file_path"])
    out_path_check = re.match("([A-Za-z]+[.]{1}[A-Za-z]+)", opt["--file_outpath"])
    assert file_path_check_1 == None, "you can not have extensions in path, only directories."
    assert out_path_check == None, "you can not have extensions in path, only directories."
    try:
        os.listdir(opt["--file_path"])
        os.listdir(opt["--file_outpath"])
    except Exception as e:
        print(e)

# test function runs here
test_function()

opt = docopt(__doc__)

def main(file_path, file_outpath, mode):

    ####Helper Functions
       
    def perc_telco_elec_exp_grouping(exp_perc):
        if exp_perc == 0:
            label = "inexistant"
        elif exp_perc <= 0.20:
            label="low"
        elif exp_perc > 0.20:
            label ="significant"
        else:
            label = "inexistant"
        return label
    
    def readability_grouping(read_score):
        if read_score <= 91.5:
            label = "low"
        elif read_score > 91.5:
            label ="high"
        else:
            label = "TBD"
        return label



    def industry_experience_feats_gen(df):
        myDict = {'Clothing & Footwear':0,
                    'Consumer electronics':0,
                    'Food Service':0,
                    'Food-Convenience-Pharmacy':0,
                    'Other':0,
                    'Sport_Travel_Enterntain_Hotel':0,
                    'Telecommunications':0,
                    'unknown':0}
        electro_telco_counter = 0
        most_recent = 0
        for i in range(7):
            col_name = "industry_"+str(i+1)
            industry_label = df[col_name]
            if industry_label is not np.nan:
                myDict[industry_label] = 1
            if industry_label in ['Consumer electronics','Telecommunications'] and most_recent == 0:
                most_recent = i+1
            if industry_label in ['Consumer electronics','Telecommunications']:
                electro_telco_counter += 1
        if most_recent == 1:
            recency_freq = "high"
        elif most_recent >1:
            recency_freq = "low"
        elif most_recent ==0:
            recency_freq = "inexistant"        
        myDict['exp_number_of_diff_industries'] = sum(myDict.values())
        myDict['most_recent_industry_exp'] = df['industry_1']
        myDict['telco_electro_recency'] = most_recent
        myDict['telco_electro_jobs'] = electro_telco_counter
        myDict['recency_type_telco_electro_exp'] = recency_freq
        return myDict

    def job_hop_def(df):
        myDict = dict()
        no_jobs = df[['work1_company','work2_company','work3_company','work4_company','work5_company','work6_company','work7_company']].count()
        myDict['no_jobs'] = no_jobs
        return myDict

    
#### END OF HELPER FUNCTIONS

    # read in data
    # train test mode
    if mode == "train":
        data_set_01 = pd.read_csv(file_path + 'manual_job_category.csv')
        data_set_02 = pd.read_csv(file_path + 'auto_communication_level.csv', index_col=0)
    elif mode == 'test':
        data_set_01 = pd.read_csv(file_path + 'manual_job_category_test.csv')
        data_set_02 = pd.read_csv(file_path + 'auto_communication_level_test.csv', index_col=0)
    else:
        print("Please pick a test or train mode")

    full_data_set = pd.merge(data_set_01,data_set_02, on = "employee_code", how = "inner")
    full_data_set = full_data_set.drop_duplicates(subset ="employee_code")


    ## Creating Features
    full_data_set['temp_ind_feat'] = full_data_set.apply(lambda x:industry_experience_feats_gen(x),axis=1)
    full_data_set['telco_electro_recency'] = full_data_set.temp_ind_feat.apply(lambda x: x['telco_electro_recency'])
    full_data_set['recency_type_telco_electro_exp'] = full_data_set.temp_ind_feat.apply(lambda x: x['recency_type_telco_electro_exp'])
    full_data_set['no_jobs'] = full_data_set.apply(lambda x: job_hop_def(x)['no_jobs'],axis=1)
    full_data_set['telco_electro_jobs'] = full_data_set.temp_ind_feat.apply(lambda x: x['telco_electro_jobs'])
    full_data_set['telco_electro_perc'] = full_data_set['telco_electro_jobs']/full_data_set['no_jobs']
    full_data_set['telco_electro_perc_group'] = full_data_set.telco_electro_perc.apply(lambda x:perc_telco_elec_exp_grouping(x))
    full_data_set['read_score_categorical'] = full_data_set['clean_Flesch-Kincaid_readability'].apply(lambda x:readability_grouping(x))
    
    ### Exporting Result
    full_data_set = full_data_set[['employee_code','telco_electro_jobs','telco_electro_recency','recency_type_telco_electro_exp','telco_electro_perc_group','read_score_categorical']]

    # train test mode
    if mode == "train":
        full_data_set.to_csv(file_outpath + 'manual_additional_feats.csv', index=False)
    elif mode == 'test':
        full_data_set.to_csv(file_outpath + 'manual_additional_feats_test.csv', index=False)
    else:
        print("Please pick a test or train mode")

if __name__ == "__main__":
    main(opt["--file_path"], opt["--file_outpath"], opt["--mode"])


print("Finish addition Feature Extraction Process")