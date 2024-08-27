# author: Haoyu Su
# date: 2020-06-02

"""
This script is used to calculate work experience length for each job

Usage: src/work_experience_calculate.py --resume_path=<resume_path> --info_path=<info_path> --file_outpath=<file_outpath>

Options:
--resume_path=<resume_path>  Name of excel file containing all work time
    and company and employee code, must be within the /data directory.
--info_path=<info_path> Name of excel containing  hire date and
employee code, must be within the /data directory.
--file_outpath=<file_outpath>  Name of directory to be saved in, no slashes necessary,
'results' folder recommended.

"""

import pandas as pd
import re
import numpy as np
from docopt import docopt
opt = docopt(__doc__)

print("Start Manual Work Experience Feature Extraction")

#file paths for testing
# resume_path = '../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_raw_data/manual_extraction_template.xlsx'
# info_path = '../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_raw_data/05182020_matching_list_maunually-checked_V1.2.xlsx'
# file_outpath = '../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/'


def load_data(resume_path, info_path):
    """
    Merge two data frame
    :param resume_path: the path of an excel file containing all work time
    and company and employee code
    :param info_path: the path of an excel table containing employee code
    and hire date
    :return: a dataframe joined by two input tables
    """
    resume = pd.read_excel(f"{resume_path}")
    info = pd.read_excel(f"{info_path}")
    df = pd.merge(resume, info, how="left", on="employee_code")
    df.columns = [remove_space(name) for name in df.columns.to_list()]
    return df


def remove_space(text):
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'^\s+', '', text)
    return text


def format_date(date):
    try:
        date = re.sub(r'\n', '', date)
        date = re.sub(r'\u200b', '', date)
        date = remove_space(date)
        date = re.sub("[Cc][Uu][Rr][Rr][Ee][Nn][Tt]", "Present", date)
        date = re.sub("[Pp][Rr][Ee][Ss][Ee][Nn][Tt]", "Present", date)
        date = re.sub("[Oo]ngoing", "Present", date)
        date = re.sub("Feburary", "February", date)
        date = re.sub("J ULY", "JULY", date)
        date = re.sub("To", "-", date)
        date = re.sub("[Nn][Oo][Ww]", "Present", date)
        date = re.sub("[Tt]ill [Dd]ate", "- Present", date)
        date = re.sub(r"\sto\s", " - ", date)
        date = re.sub(r"[Ff][Ee][Bb][Uu][Aa][Rr][Yy]", "February", date)
        date = re.sub(r"—", " - ", date)
        date = re.sub(r"–", " - ", date)
        date = re.sub(r"–", " - ", date)
        date = re.sub(r"[\’]", " ", date)
        date = re.sub(r"[0-9]{4}[\,]", " - ", date)
        date = re.sub(r"-", " - ", date)
        date = re.sub(r" -", " - ", date)
        date = re.sub(r"- ", " - ", date)
        date = re.sub(r"/", " ", date)
        date = re.sub(r"([0-9]+)(\syear[s]*)", " - ", date)
        date = re.sub(r"([0-9]+)(\s[Mm]onth[s]*.*)", " - ", date)
        date = re.sub(r"\s+", " ", date)
        date = re.sub(r"(.*)\s18", r"\1 2018", date)
        date = re.sub(r"(.*)\s12", r"\1 2012", date)
        date = re.sub(r"(.*)\s15", r"\1 2015", date)
        date = re.sub(r"(.*)\s16", r"\1 2016", date)
        date = re.sub(r"(.*)\s14", r"\1 2014", date)
        return date
    except TypeError:
        return " - "


def format_all(df, colname_lst):
    for colname in colname_lst:
        df[colname] = df[colname].apply(
            format_date)
    return df


def calculate_num_month(start_date, end_date):
    try:
        num_months = (end_date.year - start_date.year) * 12 + (
                end_date.month - start_date.month)
        return num_months
    except TypeError:
        return "unknown"


def work_length_calculater(df, colname, hire="Hire Date"):
    """

    :param df: a dataframe with formatted work date
    :param colname: a column name of the work used to be calculated length
    :param hire: a column name of hire date, default is "Hire Date"
    :return:
    """
    start = df[colname].str.split(" - ", expand=True)[0]
    end = df[colname].str.split(" - ", expand=True)[1]
    hire = df[hire]
    new = pd.concat([end, hire], axis=1).rename(columns={1: "end"})
    new.end = new.apply(
        lambda row: row["Hire Date"] if "Present" in row['end'] else row[
            "end"], axis=1)
    new = pd.concat([new, start], axis=1).rename(columns={0: "start"})
    new.end = pd.to_datetime(new.end)
    new.start = pd.to_datetime(new.start)
    new["num_month"] = new.apply(
        lambda row: calculate_num_month(row.start, row.end), axis=1)
    df = new[["start", "end", "num_month"]]
    return df


def avg_work_calculater(df, colname_lst, hire="Hire Date"):
    """
    calculate work length for all jobs and calculate the avg work experience

    :param df: a dataframe with formatted work date
    :param colname_lst: column names of all work
    :param hire: a column name of hire date, default is "Hire Date"
    :return: a dataframe containing all work length and avg work length
    """
    work1 = work_length_calculater(df, colname_lst[0], hire)["num_month"]
    work2 = work_length_calculater(df, colname_lst[1], hire)["num_month"]
    work3 = work_length_calculater(df, colname_lst[2], hire)["num_month"]
    work4 = work_length_calculater(df, colname_lst[3], hire)["num_month"]
    work5 = work_length_calculater(df, colname_lst[4], hire)["num_month"]
    work6 = work_length_calculater(df, colname_lst[5], hire)["num_month"]
    work7 = work_length_calculater(df, colname_lst[6], hire)["num_month"]
    df_all = pd.concat([work1, work2, work3, work4, work5, work6, work7],
                       axis=1)
    df_all.columns = ["work1", "work2", "work3", "work4", "work5", "work6",
                      "work7"]

    # df_all['num'] = df_all.count(axis=1)
    # df_all['total'] = df_all.sum(axis=1)
    df_all["work1"] = df_all.work1.apply(lambda x: -x if x < 0 else x)
    df_all["work2"] = df_all.work2.apply(lambda x: -x if x < 0 else x)
    df_all["work3"] = df_all.work3.apply(lambda x: -x if x < 0 else x)
    df_all["work4"] = df_all.work4.apply(lambda x: -x if x < 0 else x)
    df_all["work5"] = df_all.work5.apply(lambda x: -x if x < 0 else x)
    df_all["work6"] = df_all.work6.apply(lambda x: -x if x < 0 else x)
    df_all["work7"] = df_all.work7.apply(lambda x: -x if x < 0 else x)
    df_all['avg'] = (df_all.sum(axis=1) / df_all.count(axis=1)).round(2)
    df_all["employee_code"] = df.employee_code
    df_all = df_all[
        ["employee_code", "work1", "work2", "work3", "work4", "work5", "work6",
         "work7", "avg"]]
    companys = df[
        ["work1_company", "work2_company", "work3_company", "work4_company",
         "work5_company", "work6_company", "work7_company"]]
    df_all2 = pd.concat([df_all, companys], axis=1)
    df_all3 = (df_all2[
        ["employee_code", "work1_company", "work1", "work2_company",
         "work2", "work3_company",
         "work3", "work4_company", "work4", "work5_company",
         "work5", "work6_company", "work6", "work7_company", "work7", "avg"]]
        .rename(
        columns={"work1": "work1_length", "work2": "work2_length",
                 "work3": "work3_length",
                 "work4": "work4_length", "work5": "work5_length",
                 "work6": "work6_length",
                 "work7": "work7_length"}))
    return df_all3


def job_hopper(num, threshold):
    """
    Decide job hopper based on avg work experience length

    :param num: avg work experience
    :param threshold: a number to be used as a cut-off of job hopper
    :return: 1 if the num is less than threshold, return 0 otherwise
    """
    if pd.isna(num):
        return np.nan
    elif num < threshold:
        return 1
    else:
        return 0


def main(resume_path, info_path, file_outpath):
    # path1 = "../data/05242020_manual_extraction_template.xlsx"
    # path2 = "../data/matching_list_maunually-checked_V1.2.xlsx"
    column_lst = ["work1_time", "work2_time", "work3_time", "work4_time",
                  "work5_time", "work6_time", "work7_time"]
    resume_incl_hire = load_data(resume_path, info_path)
    resume_incl_hire2 = format_all(resume_incl_hire, column_lst)
    df_final = avg_work_calculater(resume_incl_hire2, column_lst)
    df_final["job_hopper"] = df_final.avg.apply(lambda x: job_hopper(x, 12))
    df_final.to_csv(f'{file_outpath}manual_work_exp.csv', index=False)


# main(resume_path, info_path, file_outpath)

print("Finish Manual Work Experience Feature Extraction")

if __name__ == "__main__":
    main(resume_path=opt["--resume_path"], info_path=opt["--info_path"],
         file_outpath=opt["--file_outpath"])
