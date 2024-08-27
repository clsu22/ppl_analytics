# author: Manuel Maldonado
# date: 2020-06-05

'''This script generates the following features: "total_experience_months","longest_tenure","shortest_tenure","average_tenure_per_job","job_hopper","no_jobs",

Usage: manual_tenure.py --file_path=<file_path> --file_outpath=<file_outpath> --mode=<mode>

Example:
    python scr/manual_tenure.py --file_path=data/ --file_outpath=data/

Options:
--file_path=<file_path>  Path (excluding filenames) to csv file extracted durations per job.
--file_outpath=<file_outpath>  Path for dataframe with created features.
--mode=<mode> Set to train and test
'''

import pandas as pd
import re
import os

from docopt import docopt
opt = docopt(__doc__)

print("Start Manual Tenure Feature Extraction")

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

# For testing
# file_path = '../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/'
# file_outpath = '../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/'


def main(file_path, file_outpath, mode):
    ####Helper Functions
    def job_hop_def(df):
        myDict = dict()
        no_jobs = df[
            ['work1_company', 'work2_company', 'work3_company', 'work4_company', 'work5_company', 'work6_company',
             'work7_company']].count()
        if df['avg'] < 12 and no_jobs > 2:
            job_hop_flag = 1
        else:
            job_hop_flag = 0
        myDict['no_jobs'] = no_jobs
        myDict['job_hopper_def2'] = job_hop_flag
        return myDict

    def no_job_grouping(njbs):
        if njbs == 0:
            label = "no_previous_jobs"
        elif njbs < 3:
            label = "few"
        elif njbs >= 3:
            label = "multiple"
        else:
            label = "no_previous_jobs"
        return label

    #### END OF HELPER FUNCTIONS

    # read in data
    if mode == "train":
        data = pd.read_csv(file_path + 'manual_work_exp.csv')
    elif mode == 'test':
        data = pd.read_csv(file_path + 'manual_work_exp_test.csv')
    else:
        print("Please pick a test or train mode")

    ## Creating Features
    data['avg'] = data[
        ['work1_length', 'work2_length', 'work3_length', 'work4_length', 'work5_length', 'work6_length','work7_length']].mean(axis=1)
    data['no_jobs'] = data.apply(lambda x: job_hop_def(x)['no_jobs'], axis=1)
    data['job_hopper'] = data.apply(lambda x: job_hop_def(x)['job_hopper_def2'], axis=1)
    data['total_experience_months'] = data[
        ['work1_length', 'work2_length', 'work3_length', 'work4_length', 'work5_length', 'work6_length','work7_length']].sum(axis=1)
    data['longest_tenure'] = data[
        ['work1_length', 'work2_length', 'work3_length', 'work4_length', 'work5_length', 'work6_length','work7_length']].max(axis=1)
    data['shortest_tenure'] = data[
        ['work1_length', 'work2_length', 'work3_length', 'work4_length', 'work5_length', 'work6_length','work7_length']].min(axis=1)
    data.rename(columns={"avg": "average_tenure_per_job"}, inplace=True)
    data['no_job_categorical'] = data.no_jobs.apply(lambda x: no_job_grouping(x))
    data = data[
        ['employee_code', 'total_experience_months', 'longest_tenure', 'shortest_tenure', 'average_tenure_per_job',
         'job_hopper', 'no_jobs', 'no_job_categorical']]
    ### Exporting Result
    if mode == "train":
        data.to_csv(file_outpath + 'manual_tenure.csv', index=False)
    elif mode == 'test':
        data.to_csv(file_outpath + 'manual_tenure_test.csv', index=False)
    else:
        print("Please pick a test or train mode")



if __name__ == "__main__":
    main(opt["--file_path"], opt["--file_outpath"], opt["--mode"])

# main(file_path, file_outpath)

print("Finish Manual Tenure Feature Extraction")
