# author: Haoyu Su
# date: 2020-06-08

'''This script generates the following features: sales_exp_months customer_serv_exp_months leader_ship_exp_months

Usage: build_dataset.py --file_path_features=<file_path_features> --file_path_processed=<file_path_processed> --file_outpath=<file_outpath> --mode=<mode>

Example:
    python src/feature_extraction/manual_extract_sales_cust_exp.py --file_path=../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/01_resume_scan_data/manual_extraction_template.xlsx --file_outpath=../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_feature/education_concentration.csv

Options:
--file_path_features=<file_path_features>  Path to features folders (excluding filenames) to the csv file.
--file_path_processed=<file_path_processed> Path to processed folders (excluding filenames) to the csv file.
--file_outpath=<file_outpath>  outPath for dataframe with created features.
--mode=<mode> Set to train and test
'''

print("Start Building Dataset")

import pandas as pd
import warnings
from sklearn.preprocessing import MultiLabelBinarizer
from docopt import docopt
opt = docopt(__doc__)

warnings.simplefilter('ignore')

# For testing
# file_path_features = '../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/'
# file_path_processed = '../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/'
# file_outpath = '../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/'


def main(file_path_features, file_path_processed, file_outpath, mode):
    # train test mode
    if mode == "train":
        train_original = pd.read_csv(file_path_processed + "train_dataset.csv")
        resume_original = pd.read_csv(file_path_processed + "english_clean_resumes.csv")
        avg_work = pd.read_csv(file_path_features + "manual_work_exp.csv")
        competitor = pd.read_csv(file_path_features + "manual_competitor_experience.csv")
        education_concentration = pd.read_csv(file_path_features + "manual_education_concentration.csv")
        work_title = pd.read_csv(file_path_features + "manual_jobtitle.csv")
        degree = pd.read_csv(file_path_features + "manual_higher_degree.csv")
        sales_cus_exp = pd.read_csv(file_path_features + "manual_sales_custom_exp.csv")
        consolidate_dataset = pd.read_csv(file_path_features + "auto_knowledge_skills.csv")
        communication_level = pd.read_csv(file_path_features + "auto_communication_level.csv")
        industry_exp = pd.read_csv(file_path_features + "manual_job_category.csv")
        tenure = pd.read_csv(file_path_features + "manual_tenure.csv")
        additional = pd.read_csv(file_path_features + "manual_additional_feats.csv")
    elif mode == 'test':
        train_original = pd.read_csv(file_path_processed + "test_dataset.csv")
        resume_original = pd.read_csv(file_path_processed + "english_clean_resumes.csv")
        avg_work = pd.read_csv(file_path_features + "manual_work_exp_test.csv")
        competitor = pd.read_csv(file_path_features + "manual_competitor_experience_test.csv")
        education_concentration = pd.read_csv(file_path_features + "manual_education_concentration_test.csv")
        work_title = pd.read_csv(file_path_features + "manual_jobtitle_test.csv")
        degree = pd.read_csv(file_path_features + "manual_higher_degree_test.csv")
        sales_cus_exp = pd.read_csv(file_path_features + "manual_sales_custom_exp_test.csv")
        consolidate_dataset = pd.read_csv(file_path_features + "auto_knowledge_skills_test.csv")
        communication_level = pd.read_csv(file_path_features + "auto_communication_level_test.csv")
        industry_exp = pd.read_csv(file_path_features + "manual_job_category_test.csv")
        tenure = pd.read_csv(file_path_features + "manual_tenure_test.csv")
        additional = pd.read_csv(file_path_features + "manual_additional_feats_test.csv")
    else:
        print("Please pick a test or train mode")



    education_concentration.drop_duplicates("employee_code", inplace=True)
    work_title.drop_duplicates("employee_code", inplace=True)
    degree.drop_duplicates("employee_code", inplace=True)
    sales_cus_exp.drop_duplicates("employee_code", inplace=True)
    consolidate_dataset.drop_duplicates("employee_code", inplace=True)
    communication_level.drop_duplicates("employee_code", inplace=True)
    industry_exp.drop_duplicates("employee_code", inplace=True)

    consolidate_dataset = consolidate_dataset[['employee_code', 'no_lang_spoken', 'trilingual_flag', 'goal_record',
                                               'sales_customer_base_exp', 'volunteer_exp', 'problem_solver',
                                               'sports_mention', 'communication_skills', 'team_player',
                                               'leadership_mention']]

    train = pd.merge(train_original, resume_original, how="left", on="employee_code").merge(
        avg_work, how="left", on="employee_code").merge(
        competitor, how="left", on="employee_code").merge(
        education_concentration, how="left", on="employee_code").merge(
        work_title, how="left", on="employee_code").merge(
        degree, how="left", on="employee_code").merge(
        sales_cus_exp, how="left", on="employee_code").merge(
        communication_level, how="left", on="employee_code").merge(
        industry_exp, how="left", on="employee_code").merge(
        consolidate_dataset, how="left", on="employee_code").merge(
        tenure, how="left", on='employee_code').merge(
        additional, how="left", on='employee_code')

    # give some more meaningful column names
    train = train.rename(
        columns={'interactive_arts_and technology_concentration': 'interactive_arts_and_technology_concentration',
                 'job_hopper_x': 'job_hopper',
                 'raw_date_chall_readability': 'raw_dale_chall_readability',
                 'clean_date_chall_readability': 'clean_dale_chall_readability'})

    train = train[["employee_code", 'rehired_', 'referral_flag',  # train_dataset
                   'job_hopper',  # avg_work
                   'competitor_experience',
                   'Freedom_competitor_exp', 'Koodo_competitor_exp',
                   'Shaw_competitor_exp',
                   'Telus_competitor_exp', 'Bell_competitor_exp',
                   'Rogers_competitor_exp',
                   'The Mobile Shop_competitor_exp', 'Best Buy_competitor_exp',
                   'Videotron_competitor_exp', 'Wow[!]* Mobile_competitor_exp',
                   'The Source_competitor_exp', 'Walmart_competitor_exp',
                   'Virgin Mobile_competitor_exp', 'Osl_competitor_exp',
                   # competitor_experience
                   'accounting_concentration',
                   'arts_concentration', 'business_concentration',
                   'computer_systems_concentration', 'engineering_concentration',
                   'finance_concentration', 'general_concentration',
                   'human_resource_concentration',
                   'interactive_arts_and_technology_concentration',
                   'marketing_concentration', 'not_specified_concentration',
                   'other_concentration',  # education concentration
                   'administrative_jobtitle', 'assistant_manager_jobtitle', 'blue_collar_jobtitle', 'cashier_jobtitle',
                   'cook_jobtitle',
                   'customer_service_representative_jobtitle', 'driver_jobtitle', 'education_jobtitle',
                   'financial_services_jobtitle', 'fitness_sports_jobtitle', 'manager_jobtitle', 'no_work_title',
                   'other_jobtitle',
                   'sales_associate_jobtitle', 'technicians_jobtitle', 'telemarketers_jobtitle',  # work_title
                   'highest_degree', 'background_highest_degree',
                   'country_highest_degree', 'flag_hd_bachelor_plus', 'flag_hd_highschool', 'business_flag',
                   # degree
                   'sales_exp_months', 'customer_serv_exp_months', 'leader_ship_exp_months',
                   # sales_cus_exp
                   'raw_dale_chall_readability', 'raw_Flesch-Kincaid_readability', 'raw_Gunning_FOG_readability',
                   'raw_automate_readability', 'clean_Flesch-Kincaid_readability', 'clean_Gunning_FOG_readability',
                   'clean_automate_readability',
                   'clean_dale_chall_readability',  # communication level
                   'telco_electro_jobs', 'telco_electro_recency', 'recency_type_telco_electro_exp',
                   'telco_electro_perc_group', 'read_score_categorical',  # additional
                   'no_lang_spoken', 'trilingual_flag', 'goal_record',
                   'sales_customer_base_exp', 'volunteer_exp', 'problem_solver',
                   'sports_mention', 'communication_skills', 'team_player',
                   'leadership_mention',  # others features in consolidate dataset
                   'total_experience_months', 'longest_tenure', 'shortest_tenure', 'average_tenure_per_job',
                   'no_jobs', 'no_job_categorical', # tenure
                   'clean_text',  # resume
                   'hp_class']]

    train.referral_flag = train.referral_flag.fillna(0)
    train.rehired_ = train.rehired_.fillna("No")

    other_countries = ['dubai', 'pakistan', 'uk', 'china',
                       'england', 'syria', 'indore', 'usa', 'philippine',
                       'vietnam',
                       'brazil', 'turkey', 'united kingdom', 'punjab', 'taiwan',
                       'u']

    train.country_highest_degree = train.country_highest_degree.apply(
        lambda x: "others" if x in other_countries else x)

    # regroup background_highest_degree
    other_backgrounds = ['audio technician', 'audio technician', 'kinesiology',
                         'blue collar', 'economics', 'sociology',
                         'kinesiology', 'physic', 'statistic', 'hospitality',
                         'criminology', 'english', 'dental', 'human resource',
                         'healthcare', 'communication', 'education']
    train.background_highest_degree = train.background_highest_degree.apply(
        lambda x: "others" if x in other_backgrounds else x)

    # new_industry_cat = pd.read_csv("../data/job_category_features_2020_06_04_V3.csv")
    if mode == "train":
        new_industry_cat = pd.read_csv(file_path_features + "manual_job_category.csv")
    elif mode == 'test':
        new_industry_cat = pd.read_csv(file_path_features + "manual_job_category_test.csv")
    else:
        print("Please pick a test or train mode")


    new_industry_cat = new_industry_cat.fillna("")
    new_industry_cat.drop_duplicates("employee_code", inplace=True)

    new_industry_cat["industry_expr"] = new_industry_cat.apply(
        lambda x: list([x['industry_1'],
                        x['industry_2'],
                        x['industry_3'],
                        x['industry_4'],
                        x['industry_5'],
                        x['industry_6'],
                        x['industry_7']]), axis=1)

    df = new_industry_cat[["industry_expr"]]

    mlb = MultiLabelBinarizer()

    res = pd.DataFrame(mlb.fit_transform(df.industry_expr),
                       columns=mlb.classes_ + "_industry_exp",
                       index=df.index)

    new_industry_cat_2 = pd.concat([new_industry_cat, res], axis=1)

    new_industry_cat_2.drop(['work1_company','_industry_exp', 'work2_company', 'work3_company',
       'work4_company', 'work5_company', 'work6_company', 'work7_company',
       'industry_1', 'industry_2', 'industry_3', 'industry_4', 'industry_5',
       'industry_6', 'industry_7', 'industry_expr'],axis = 1,inplace = True)

    train.drop_duplicates("employee_code", inplace=True)

    train_new = pd.merge(train, new_industry_cat_2, on = "employee_code",how = "left")


    #last name change
    train_new = train_new.rename(columns={
        'Food Service_industry_exp':'Food_Service_industry_exp',
        'The Mobile Shop_competitor_exp': 'The_Mobile_Shop_competitor_exp',
        'Best Buy_competitor_exp': 'Best_Buy_competitor_exp',
        'Consumer electronics_industry_exp':'Consumer_electronics_industry_exp',
        'Virgin Mobile_competitor_exp': 'Virgin_Mobile_competitor_exp',
        'The Source_competitor_exp': 'The_Source_competitor_exp',
        'Wow[!]* Mobile_competitor_exp': 'Wow_Mobile_competitor_exp',
        'Clothing & Footwear_industry_exp': 'Clothing_and_Footwear_industry_exp'})

    # train test mode
    if mode == "train":
        train_new.to_csv(file_outpath + "manual_clean_training_dataset.csv", index=False)
    elif mode == 'test':
        train_new.to_csv(file_outpath + "manual_clean_testing_dataset.csv", index=False)
    else:
        print("Please pick a test or train mode")

# main(file_path_features, file_path_processed, file_outpath)

if __name__ == "__main__":
    main(opt["--file_path_features"], opt["--file_path_processed"], opt["--file_outpath"], opt["--mode"])

print("Finish Building Dataset")
