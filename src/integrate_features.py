# author: Haoyu Su
# date: 2020-06-08

import pandas as pd
import warnings
from sklearn.preprocessing import MultiLabelBinarizer
warnings.simplefilter('ignore')


train_original = pd.read_csv("../data/train_dataset.csv")
test_original = pd.read_csv("../data/test_dataset.csv")
resume_original = pd.read_csv(
    "../data/05182020_cleaned_english_resumes_V1.0.csv")
avg_work = pd.read_csv("../result/avg_work_exp.csv")
competitor = pd.read_csv("../result/competitor_experience.csv")
education_concentration = pd.read_csv("../data/education_concentration.csv")
work_title = pd.read_csv("../data/work_title.csv")
degree = pd.read_csv("../data/features_edu_01_training_df.csv")
sales_cus_exp = pd.read_csv("../data/sales_custom_exp.csv")
word_counts = pd.read_csv("../data/word_counts.csv")
consolidate_dataset = pd.read_csv(
    "../data/consolidated_features_01_training_dataset.csv")
communication_level = pd.read_csv("../data/communication_level.csv")
industry_exp = pd.read_csv("../data/industry_exp.csv")

education_concentration.drop_duplicates("employee_code", inplace=True)
work_title.drop_duplicates("employee_code", inplace=True)
degree.drop_duplicates("employee_code", inplace=True)
sales_cus_exp.drop_duplicates("employee_code", inplace=True)
consolidate_dataset.drop_duplicates("employee_code", inplace=True)
communication_level.drop_duplicates("employee_code", inplace=True)
industry_exp.drop_duplicates("employee_code", inplace=True)
consolidate_dataset = consolidate_dataset[
    ['employee_code', 'no_lang_spoken', 'trilingual_flag', 'goal_record', \
     'sales_customer_base_exp', 'volunteer_exp', 'problem_solver', \
     'sports_mention', 'communication_skills', 'team_player', \
     'leadership_mention']]

train = pd.merge(train_original, resume_original, how="left",
                 on="employee_code").merge(
    avg_work, how="left", on="employee_code").merge(
    competitor, how="left", on="employee_code").merge(
    education_concentration, how="left", on="employee_code").merge(
    work_title, how="left", on="employee_code").merge(
    degree, how="left", on="employee_code").merge(
    sales_cus_exp, how="left", on="employee_code").merge(
    word_counts, how="left", on="employee_code").merge(
    communication_level, how="left", on="employee_code").merge(
    industry_exp, how="left", on="employee_code").merge(
    consolidate_dataset, how="left", on="employee_code"
)

train = train[["employee_code", 'rehired_', 'referral_flag',  # train_dataset
               "job_hopper",  # avg_work
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
               'administrative', 'assistant manager', 'blue collar', 'cashier',
               'cook',
               'customer service representative', 'driver', 'education',
               'financial services', 'fitness/sports', 'manager', 'no work',
               'other',
               'sales associate', 'technicians', 'telemarketers',  # work_title
               'highest_degree', 'background_highest_degree',
               'country_highest_degree',
               # degree
               'sales_exp', 'customer_serv_exp', 'leader_ship_exp',
               # sales_cus_exp
               'efficient service', 'mobile expert', 'high school',
               'information system', 'cash register',  # word count
               'raw_date_chall_readability',  # communication level
               'food_service_industry_exp',
               'apparel_industry_exp', 'supercenter_convenience_industry_exp',
               'automotive_sales_industry_exp', 'blue_collar_industry_exp',
               'consumer_electronics',  # industry_exp
               'no_lang_spoken', 'trilingual_flag', 'goal_record',
               'sales_customer_base_exp', 'volunteer_exp', 'problem_solver',
               'sports_mention', 'communication_skills', 'team_player',
               'leadership_mention',  # others features in consolidate dataset
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

new_industry_cat = pd.read_csv(
    "../data/job_category_features_2020_06_04_V3.csv")
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

train_new = pd.concat([train, res], axis=1)[:288]

train_new.to_csv("../data/0605_training_dataset.csv", index=False)
