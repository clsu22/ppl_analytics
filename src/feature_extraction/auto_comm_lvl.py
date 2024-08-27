# author: Thomas Pin
# date: 2020-06-05

'''This script generates the following features: raw_char_count	raw_word_count	raw_Flesch-Kincaid_readability
raw_Gunning_FOG_readability	raw_automate_readability raw_date_chall_readability
clean_char_count clean_word_count clean_Flesch-Kincaid_readability	clean_Gunning_FOG_readability
clean_automate_readability	clean_date_chall_readability

Usage: auto_knowledge_skills.py --file_path_1=<file_path_1> --file_path_2=<file_path_2> --file_outpath=<file_outpath>

Example:
    auto_knowledge_skills.py

Options:
--file_path_1=<file_path_1>  Path to csv file resume data.
--file_path_2=<file_path_2> Path to csv file of training or testing data
--file_outpath=<file_outpath>  Path for dataframe with created features.
'''

import pandas as pd
import textstat
pd.options.mode.chained_assignment = None
from docopt import docopt

opt = docopt(__doc__)

print("Start Communication Level Feature Extraction")

# file names for testing
# file_path_1 = '../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/english_clean_resumes.csv'
# file_path_2 = '../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/train_dataset.csv'
# file_outpath = '../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/auto_communication_level.csv'


def main(file_path_1, file_path_2, file_outpath):
    # read in csv files
    df = pd.read_csv(file_path_1)
    df_train = pd.read_csv(file_path_2)

    # merge
    df_train_code = df_train[['employee_code', 'hp_class']]
    df_resume = df[['employee_code', 'employee_name', 'clean_text', 'resume_text']]
    df_resume_clean = df_train_code.merge(df_resume, on='employee_code')

    # start new data frame for out
    df_resume_com_lvl = df_resume_clean[['employee_code', 'employee_name']]

    # find word counts
    clean_chr_count = []
    clean_word_count = []
    raw_chr_count = []
    raw_word_count = []

    for i in range(0, df_resume_clean.shape[0]):
        clean_chr_count.append(len(df_resume_clean['clean_text'].iloc[i]))
        clean_word_count.append(len(df_resume_clean['clean_text'].iloc[i].split()))
        raw_chr_count.append(len(df_resume_clean['resume_text'].iloc[i]))
        raw_word_count.append(len(df_resume_clean['resume_text'].iloc[i].split()))

    # resumes text (raw text)
    # character count
    df_resume_com_lvl['raw_char_count'] = raw_chr_count
    # word count
    df_resume_com_lvl['raw_word_count'] = raw_word_count

    kincaid_grade = []
    gunning_fog = []
    automate_readability = []
    date_chall_read = []
    for i in range(0, df_resume_clean.shape[0]):
        txt = df_resume_clean['resume_text'][i]
        fkg = textstat.flesch_kincaid_grade(txt)
        kincaid_grade.append(fkg)
        gf = textstat.gunning_fog(txt)
        gunning_fog.append(gf)
        ar = textstat.automated_readability_index(txt)
        automate_readability.append(ar)
        dcr = textstat.dale_chall_readability_score(txt)
        date_chall_read.append(dcr)

    # Flesch-Kincaid 9th grader can read at a 9.3 avg
    df_resume_com_lvl['raw_Flesch-Kincaid_readability'] = kincaid_grade
    # Gunning_Fog 9th grader can read at a 9.3 avg
    df_resume_com_lvl['raw_Gunning_FOG_readability'] = gunning_fog
    # Automated Readability Index 6.5 is for a 6th to 7th grader
    df_resume_com_lvl['raw_automate_readability'] = automate_readability
    # Dale-Chad Readability 9th grade can read at a 7.0 - 7.9
    df_resume_com_lvl['raw_date_chall_readability'] = date_chall_read

    # clean text
    # character count
    df_resume_com_lvl['clean_char_count'] = clean_chr_count
    # word count
    df_resume_com_lvl['clean_word_count'] = clean_word_count

    kincaid_grade = []
    gunning_fog = []
    automate_readability = []
    date_chall_read = []
    for i in range(0, df_resume_clean.shape[0]):
        txt = df_resume_clean['clean_text'][i]
        fkg = textstat.flesch_kincaid_grade(txt)
        kincaid_grade.append(fkg)
        gf = textstat.gunning_fog(txt)
        gunning_fog.append(gf)
        ar = textstat.automated_readability_index(txt)
        automate_readability.append(ar)
        dcr = textstat.dale_chall_readability_score(txt)
        date_chall_read.append(dcr)

    # Flesch-Kincaid 9th grader can read at a 9.3 avg
    df_resume_com_lvl['clean_Flesch-Kincaid_readability'] = kincaid_grade
    # Gunning_Fog 9th grader can read at a 9.3 avg
    df_resume_com_lvl['clean_Gunning_FOG_readability'] = gunning_fog
    # Automated Readability Index 6.5 is for a 6th to 7th grader
    df_resume_com_lvl['clean_automate_readability'] = automate_readability
    # Dale-Chad Readability 9th grade can read at a 7.0 - 7.9
    df_resume_com_lvl['clean_date_chall_readability'] = date_chall_read

    df_resume_com_lvl.to_csv(file_outpath)
    return df_resume_com_lvl

# main(file_path_1, file_path_2, file_outpath)

if __name__ == "__main__":
    main(opt["--file_path_1"], opt["--file_path_2"], opt["--file_outpath"])

print("End Communication Level Feature Extraction")
