# author: Manuel Maldonado
# date: 2020-06-05

'''This script consolidates the tabular data, applies data exclusions, creates the target feature and splits the final sample size into train and test data set

Usage: tabular_data_consolidator.py --file_path_1=<file_path_1> --file_path_2=<file_path_2> --clean_path=<clean_path>

Example:
    python scripts/tabular_data_consolidator.py --file_path_1=data/ --file_path_2=data/ --clean_path=data/

Options:
--file_path_1=<file_path_1>  Path (excluding filenames) to tabular dataset
--file_path_2=<file_path_2>  Path (excluding filenames) to resume text extraction dataset
--clean_path=<clean_path>  Path for dataframe with created features.
'''

import pandas as pd
import numpy as np
import altair as alt
import janitor
import re
from datetime import datetime, timedelta
import calendar
from sklearn.model_selection import train_test_split
import os
from docopt import docopt

opt = docopt(__doc__)

def test_function():
    '''
    Tests the input data and specified paths.
    '''
    file_path_check_1 = re.match("([A-Za-z]+[.]{1}[A-Za-z]+)", opt["--file_path_1"])
    file_path_check_2 = re.match("([A-Za-z]+[.]{1}[A-Za-z]+)", opt["--file_path_1"])
    out_path_check = re.match("([A-Za-z]+[.]{1}[A-Za-z]+)", opt["--clean_path"])
    assert file_path_check_1 == None, "you can not have extensions in path, only directories."
    assert file_path_check_2 == None, "you can not have extensions in path, only directories."
    assert out_path_check == None, "you can not have extensions in path, only directories."
    try:
        os.listdir(opt["--file_path_1"])
        os.listdir(opt["--clean_path"])
    except Exception as e:
        print(e)

# test function runs here
test_function()

opt = docopt(__doc__)

def main(file_path_1,file_path_2,clean_path):

    ####Helper Functions
    
    def str_cols_to_date (df,col_list,frmt='%A, %B %d, %Y'):
        df[col_list] = df[col_list].applymap(lambda x:datetime.strptime(x,frmt))
        return df
    
    def exclusions(df):
        if (df['tenure_at_termination'] < timedelta(days=90)) and (df['tenure_at_termination'] >= timedelta(days=0)):
            return "01-Low Tenure - Termination <= 90 days"
        elif (df['tenure_at_end_perf_window'] < timedelta(days=90)) and (df['tenure_at_end_perf_window'] >= timedelta(days=0)):
            return "02-Low Tenure - Recent Hiring <= 90 days of perf"
        elif (df['resume_found']!= 1):
            return "03-No Resume"
        elif (df['language']!= "English"):
            return "04-Resume not in English"
        elif (df['perf_found']!= 1):
            return "05-No Perf Information"
        else:
            return "06-Pass"
 
#### END OF HELPER FUNCTIONS

    ## read in data
    #Weekly Status Active + Leave Data
    employee_status_weekly_snapshot = pd.read_excel(file_path_1+'Active + Leave (Weekly) - 2020_04_28.xlsx').clean_names()
    employee_status_weekly_snapshot = str_cols_to_date(employee_status_weekly_snapshot,['recent_hire_date_if_applicable_','report_date_week_ending_'])
    employee_snapshot_valid_reports = employee_status_weekly_snapshot[employee_status_weekly_snapshot['report_date_week_ending_'] >= employee_status_weekly_snapshot.groupby('employee_code').recent_hire_date_if_applicable_.transform('max')]
        ##t3 contains first (oldest) status report available since most recent hire
    employee_status_attr_t3 = employee_snapshot_valid_reports[employee_snapshot_valid_reports.groupby('employee_code').report_date_week_ending_.transform('min')==employee_snapshot_valid_reports['report_date_week_ending_']]
    employee_status_attr_t3.rename(columns={'recent_hire_date_if_applicable_':"max_hire_date"},inplace=True)

    #Terminations Data
    terminations_df = pd.read_excel(file_path_1+'Terminations - 2020.04.28.xlsx').clean_names()
    terminations_df_attr = str_cols_to_date(terminations_df,['termination_date'])[['employee_code','termination_date','termination_reason','termination_type']]
        ## Obtaing snapshot of latest termination in case of multiple
    terminations_df_attr = terminations_df_attr[terminations_df_attr.groupby('employee_code').termination_date.transform('max')==terminations_df_attr['termination_date']]
    
    #Adding Performance Data (employee_plateau)
    perf_plateau_df = pd.read_excel(file_path_1+'Employee Plateau Levels by Month.xlsx').clean_names()
    perf_plateau_df['perf_month'] = (perf_plateau_df['year'].astype(str) + '-' +perf_plateau_df['month'].astype(str).str[0:3])
    perf_plateau_df = str_cols_to_date(perf_plateau_df,['perf_month'],'%Y-%b')
        ## We then change the date to reflect the end of month date
    perf_plateau_df['perf_month']=perf_plateau_df.perf_month.apply(lambda x: x.replace(day =calendar.monthrange(x.year,x.month)[1]))
        ## Creating binary flag for reaching mininum compensation plateau
    perf_plateau_df['high_perf_flag'] = perf_plateau_df.high_performer.map(lambda x: 1 if x =="Yes" else 0)
    perf_plateau_df.rename(columns={'username':'employee_code'},inplace=True)
        ## Add max hire date, just to analyze performance from latest hire date onwards
    perf_plateau_df = pd.merge(perf_plateau_df,employee_status_attr_t3[['employee_code','max_hire_date']], on= 'employee_code',how = 'inner')
        ## Just including performance from the latest hired period. Filtering
    perf_plateau_df = perf_plateau_df[perf_plateau_df.perf_month >perf_plateau_df.max_hire_date]
    perf_plateau_df = perf_plateau_df.drop_duplicates()
        ### Logic for ignoring first two months
    perf_plateau_df = perf_plateau_df.sort_values(by=["employee_code","perf_month"])
    perf_plateau_df["MOB"] = perf_plateau_df.groupby(['employee_code']).cumcount()+1
    perf_plateau_df = perf_plateau_df[perf_plateau_df.MOB >2]
        ## Creating counts for performance classification
    perf_plateau_df = perf_plateau_df[['employee_code','high_perf_flag']].groupby('employee_code').agg({'high_perf_flag': ['sum','count']})
    perf_plateau_df.columns = perf_plateau_df.columns.droplevel(0)
    perf_plateau_df = perf_plateau_df.reset_index()
    perf_plateau_df.rename(columns={'sum':'months_high_perf','count':'months_with_perf'},inplace=True)
    perf_plateau_df['hp_perc'] = perf_plateau_df['months_high_perf']/perf_plateau_df['months_with_perf']
    perf_plateau_df['hp_class'] = perf_plateau_df['hp_perc'].map(lambda x: 1 if x>= 0.75 else 0)
    perf_plateau_df['perf_found'] = 1

    ##Rehire data
    new_hires = pd.read_excel(file_path_1+'New Hires - 2020.04.28.xlsx').clean_names()
    new_hires_df_attr = new_hires[['employee_code','job_title_description','seniority_date_greater_than_hire_or_rehire_','rehired_']]
    new_hires_df_attr = str_cols_to_date(new_hires_df_attr,['seniority_date_greater_than_hire_or_rehire_'])
        #### extract snapshot of latest hire
    new_hires_df_attr = new_hires_df_attr[new_hires_df_attr.groupby('employee_code').seniority_date_greater_than_hire_or_rehire_.transform('max')==new_hires_df_attr['seniority_date_greater_than_hire_or_rehire_']]
    new_hires_df_attr = new_hires_df_attr.drop_duplicates()
        ## extracting re_hires
    rehires_df = new_hires_df_attr[new_hires_df_attr.rehired_ =="Yes"][['employee_code','rehired_']]

    ##Referral Data
    referral_df = pd.read_excel(file_path_1+'Referrals - 2020.04.28.xlsx').clean_names()
    referral_df['referral_flag'] = 1

    #Resume Employee Data
    resume_employee = pd.read_csv(file_path_2+'05182020_cleaned_english_resumes_V1.0.csv',index_col =0).clean_names()
    resume_employee['resume_found'] = 1

    ##Merging Data
    full_df = pd.merge(employee_status_attr_t3,terminations_df_attr,on='employee_code',how='left')
    full_df['tenure_at_termination'] = full_df['termination_date'] - full_df['max_hire_date']
    full_df['end_perf_window'] = datetime(2020, 2, 29)

    full_df['tenure_at_end_perf_window'] = full_df['end_perf_window'] - full_df['max_hire_date']


        ##Adding resume data
    full_df = pd.merge(full_df,resume_employee[['employee_code','employee_name','language','resume_found']],on='employee_code',how='left')

        ##Adding performance data
    full_df = pd.merge(full_df,perf_plateau_df,on='employee_code',how='left')

        ##Adding rehire flag
    full_df = pd.merge(full_df,rehires_df,on="employee_code",how ="left")

        ## Adding refferral flag
    full_df = pd.merge(full_df,referral_df[['employee_code','referral_flag']],on="employee_code",how ="left")



    ## Generating Exclusion Code

    full_df['exclusion_code'] = full_df.apply(exclusions, axis=1)
    final_sample_full= full_df[full_df['exclusion_code']=='06-Pass']

    ## Splitting into train and test dataset

    train_dataset = final_sample_full.sample(frac=0.80,random_state=1234)
    test_dataset = final_sample_full[~final_sample_full.index.isin(list(train_dataset.index))]

    ### Exporting Result

    full_df.to_csv(clean_path+'consolidated_tabular_data_df.csv',index=False)
    train_dataset.to_csv(clean_path+'train_dataset.csv')
    test_dataset.to_csv(clean_path+'test_dataset.csv')


if __name__ == "__main__":
    main(opt["--file_path_1"],opt["--file_path_2"], opt["--clean_path"])