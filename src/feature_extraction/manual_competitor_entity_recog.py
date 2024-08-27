# author: Haoyu Su
# date: 2020-06-02

"""
This script is used to identify competitor entity

Usage: src/work_experience_calculate.py --resume_path=<resume_path> --file_outpath=<file_outpath> --mode=<mode>

Options:
--resume_path=<resume_path>  Name of excel file containing all work time
    and company and employee code, must be within the /data directory.
--file_outpath=<file_outpath>  Name of directory to be saved in, no slashes necessary,
'results' folder recommended.
--mode=<mode> Set to train and test
"""

import pandas as pd
import re
import numpy as np
import string
from docopt import docopt
import warnings
warnings.simplefilter('ignore')

opt = docopt(__doc__)

print("Start Manual Competitor Entity Recognition")

# file paths for testing
# resume_path = '../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_raw_data/manual_extraction_template.xlsx'
# file_outpath = '../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/'



def load_data(resume_path):
    df = pd.read_excel(f"{resume_path}")
    df.columns = [remove_space(name) for name in df.columns.to_list()]
    return df


def remove_space(text):
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'^\s+', '', text)
    return text


def cap_words(words):
    try:
        new_word = string.capwords(words)
        return new_word
    except AttributeError:
        return np.nan


def find_competitor(company, competitor_lst):
    pat_comp = r'.*?(\b(?:{})\b).*?'.format('|'.join(competitor_lst))
    try:
        return re.match(pat_comp, company).group(1)
    except AttributeError:
        return np.nan
    except TypeError:
        return np.nan


def find_company(text, company):
    comp = [company]
    pat_comp = r'.*?(\b(?:{})\b).*?'.format('|'.join(comp))
    return bool(re.match(pat_comp, text))


def add_competitor_flag(df, colname_lst, competitor_lst):
    """
    Add competitor experience flag to employee
    :param df: a dataframe containing all companys the employee used to work
    for
    :param colname_lst: a list of selected columns
    :param competitor_lst: a list containing all competitors
    :return:
    """
    df2 = df[colname_lst]
    df2["work1_flag"] = df2.work1_company.apply(cap_words).apply(
        lambda x: find_competitor(x, competitor_lst))
    df2["work2_flag"] = df2.work2_company.apply(cap_words).apply(
        lambda x: find_competitor(x, competitor_lst))
    df2["work3_flag"] = df2.work3_company.apply(cap_words).apply(
        lambda x: find_competitor(x, competitor_lst))
    df2["work4_flag"] = df2.work4_company.apply(cap_words).apply(
        lambda x: find_competitor(x, competitor_lst))
    df2["work5_flag"] = df2.work5_company.apply(cap_words).apply(
        lambda x: find_competitor(x, competitor_lst))
    df2["work6_flag"] = df2.work6_company.apply(cap_words).apply(
        lambda x: find_competitor(x, competitor_lst))
    df2["work7_flag"] = df2.work7_company.apply(cap_words).apply(
        lambda x: find_competitor(x, competitor_lst))
    df3 = df2[["employee_code", "work1_flag", "work2_flag", "work3_flag",
               "work4_flag", "work5_flag", "work6_flag", "work7_flag"]]
    df3["competitor_experience"] = df3.apply(lambda row: 0 if
    pd.isna(row.work1_flag) and
    pd.isna(row.work2_flag) and
    pd.isna(row.work3_flag) and
    pd.isna(row.work4_flag) and
    pd.isna(row.work5_flag) and
    pd.isna(row.work6_flag) and
    pd.isna(row.work7_flag)
    else 1, axis=1)
    return df3


def add_comp_flag(df, competitor_lst):
    """
    Break down competitor experience into which competitor

    :param df: a dataframe containing work flags
    :param competitor_lst: a list containing all competitors
    :return: 1 if employee used to work in that competitor, returns 0 otherwise
    """
    for company in competitor_lst:
        df[company + "_competitor_exp"] = (df[["work1_flag", "work2_flag",
                                               "work3_flag", "work4_flag",
                                               "work5_flag",
                                               "work6_flag", "work7_flag"]]
            .fillna("")
            .apply(
            lambda row: 1 if find_company(row["work1_flag"], company) or
                             find_company(row["work2_flag"], company) or
                             find_company(row["work3_flag"], company) or
                             find_company(row["work4_flag"], company) or
                             find_company(row["work5_flag"], company) or
                             find_company(row["work6_flag"], company) or
                             find_company(row["work7_flag"], company)
            else 0, axis=1))
    return df


def main(resume_path, output, mode):
    resume = load_data(resume_path)
    colname_lst = ["employee_code", "work1_company", "work2_company",
                   "work3_company",
                   "work4_company", "work5_company", "work6_company",
                   "work7_company"]
    competitors = ['Freedom', 'Koodo', 'Shaw', 'Telus', 'Bell', 'Rogers',
                   'The Mobile Shop',
                   'Best Buy', 'Videotron', 'Wow[!]* Mobile', 'The Source',
                   'Walmart', 'Virgin Mobile', 'Osl']
    df_final = add_comp_flag(
        add_competitor_flag(resume, colname_lst, competitors), competitors)
    if mode == 'train':
        df_final.to_csv(f'./{output}/manual_competitor_experience.csv', index=False)
    elif mode == 'test':
        df_final.to_csv(f'./{output}/manual_competitor_experience_test.csv', index=False)
    else:
        print("Please pick a test or train mode")

# main(resume_path, file_outpath)

if __name__ == "__main__":
    main(resume_path=opt["--resume_path"], output=opt["--file_outpath"], mode=opt["--mode"])

print("End Manual Competitor Entity Recognition")