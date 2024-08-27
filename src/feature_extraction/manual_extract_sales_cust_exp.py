# author: Thomas Pin
# date: 2020-06-06

'''This script generates the following features: sales_exp_months customer_serv_exp_months leader_ship_exp_months

Usage: manual_extract_sales_cust_exp.py --file_path=<file_path> --file_outpath=<file_outpath> --mode=<mode>

Example:
    python src/feature_extraction/manual_extract_sales_cust_exp.py --file_path=../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/01_resume_scan_data/manual_extraction_template.xlsx --file_outpath=../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_feature/education_concentration.csv

Options:
--file_path=<file_path>  Path (excluding filenames) to the csv file.
--file_outpath=<file_outpath>  Path for dataframe with created features.
--mode=<mode>
'''

import pandas as pd
import numpy as np
from docopt import docopt
opt = docopt(__doc__)

print("Start Customer Service and Sales Experience Feature Exaction")

# For testing
# file_path = '../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/'
# file_outpath = '../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/'


def main(file_path, file_outpath, mode):

    if mode == 'train':
        df_work_title = pd.read_csv(file_path + 'manual_jobtitle.csv')
        df_work_exp = pd.read_csv(file_path + "manual_work_exp.csv")
    elif mode == 'test':
        df_work_title = pd.read_csv(file_path + 'manual_jobtitle_test.csv')
        df_work_exp = pd.read_csv(file_path + "manual_work_exp_test.csv")
    else:
        print("Please pick a test or train mode")


    df_work_title.columns = df_work_title.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(',
                                                                                                            '').str.replace(
        ')', '')


    df_work_title = df_work_title.drop(
        ['unnamed:_0', 'employee_name', 'administrative_jobtitle', 'assistant_manager_jobtitle',
         'blue_collar_jobtitle', 'cashier_jobtitle', 'cook_jobtitle',
         'customer_service_representative_jobtitle', 'driver_jobtitle',
         'education_jobtitle', 'financial_services_jobtitle',
         'fitness_sports_jobtitle', 'manager_jobtitle', 'no_work_title',
         'other_jobtitle', 'sales_associate_jobtitle', 'server_jobtitle',
         'technicians_jobtitle', 'telemarketers_jobtitle'], axis=1)


    df_work_exp.columns = df_work_exp.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(',
                                                                                                        '').str.replace(
        ')',
        '')
    df_work_exp = df_work_exp.drop(['work1_company', 'work2_company', 'work3_company', 'work4_company',
                                    'work5_company', 'work6_company', 'work7_company'], axis=1)

    df_work_exp_time = pd.merge(df_work_exp, df_work_title, on='employee_code')

    df_work_exp_time = df_work_exp_time[
        ['employee_code', 'work1_length', 'work1_title_label', 'work2_length', 'work2_title_label', 'work3_length',
         'work3_title_label', 'work4_length', 'work4_title_label', 'work5_length', 'work5_title_label', 'work6_length',
         'work6_title_label', 'work7_title_label', 'work7_length']]
    df_work_exp_time = df_work_exp_time.fillna(0)

    for i in range(1, 8):
        df_work_exp_time['work' + str(i) + '_length'] = np.abs(df_work_exp_time['work' + str(i) + '_length'])

    sales_exp = np.zeros(df_work_exp_time.shape[0])
    customer_ser_exp = np.zeros(df_work_exp_time.shape[0])
    leader_ship_exp = np.zeros(df_work_exp_time.shape[0])

    for i in range(0, df_work_exp_time.shape[0]):
        for j in range(1, 8):
            if df_work_exp_time['work' + str(j) + '_title_label'][i] == 'sales associate':
                sales_exp[i] += df_work_exp_time['work' + str(j) + '_length'][i]
            if any(x in df_work_exp_time['work' + str(j) + '_title_label'][i] for x in ['service', 'server']):
                customer_ser_exp[i] += df_work_exp_time['work' + str(j) + '_length'][i]
            if any(x in df_work_exp_time['work' + str(j) + '_title_label'][i] for x in ['manager']):
                leader_ship_exp[i] += df_work_exp_time['work' + str(j) + '_length'][i]

    df_work_exp_time['sales_exp_months'] = sales_exp
    df_work_exp_time['customer_serv_exp_months'] = customer_ser_exp
    df_work_exp_time['leader_ship_exp_months'] = leader_ship_exp

    df_work_exp_time = df_work_exp_time.drop(['work1_length', 'work1_title_label', 'work2_length',
                                              'work2_title_label', 'work3_length', 'work3_title_label',
                                              'work4_length', 'work4_title_label', 'work5_length',
                                              'work5_title_label', 'work6_length', 'work6_title_label'], axis=1)

    if mode == 'train':
        df_work_exp_time.to_csv(file_outpath + 'manual_sales_custom_exp.csv')
    elif mode == 'test':
        df_work_exp_time.to_csv(file_outpath + 'manual_sales_custom_exp_test.csv')
    else:
        print("Please pick a test or train mode")

if __name__ == '__main__':
    main(opt["--file_path"], opt["--file_outpath"], opt["--mode"])

# main(file_path, file_outpath)

print("Finish Customer Service and Sales Experience Feature Exaction")
