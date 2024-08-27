# author: Robert Pimentel (Following structure of Haoyu Su mnual feature)
# date: 2020-06-17


'''This script is used to identify competitor entity directly from resume text

Usage: auto_competitor_entity_recog.py --resume_path=<resume_path> --file_outpath=<file_outpath> --mode=<mode>

Example:
python auto_competitor_entity_recog.py --resume_path="data/input/" --file_outpath="data/output/" --mode="train"

Options:
--resume_path=<resume_path>  Name of excel file containing all work time
    and company and employee code, must be within the /data directory.
--file_outpath=<file_outpath>  Name of directory to be saved in, no slashes necessary,
'results' folder recommended.
--mode=<mode> Set to train and test
'''

import pandas as pd
import re
import numpy as np
import string
from docopt import docopt
import warnings
warnings.simplefilter('ignore')

opt = docopt(__doc__)

print()
print("Start Auto Competitor Entity Recognition")

# file paths for testing
# resume_path = '../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_raw_data/manual_extraction_template.xlsx'
# file_outpath = '../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/'


def load_data(resume_path):
#     df =  pd.read_csv(f"{resume_path}")
    df =  pd.read_csv(resume_path)
    df = df[['employee_code', 'employee_name', 'clean_text', 'resume_text']]
    return  df

def clean_txt(txt):
    txt = str(txt)
    punct_list = list(string.punctuation+"·"+"−"+"§"+"•"+"–"+"––"+"✓"+"❖"+"❑")
    punct_list.remove("&")
    punct_list.remove("@")
    punct_list.remove("/")
    exclude = set(punct_list)    
    clean_txt = "".join([i for i in txt.lower() if i not in exclude])
    clean_txt = re.sub(r'\s+', ' ', clean_txt)
    #Remove Emails
    clean_txt = re.sub(r'([\w\.-]+@[\w\.-]+)', '', clean_txt)
    clean_txt = clean_txt.replace("/", " ")
    return clean_txt
  

def find_competitor_list(txt, competitor_lst):
    txt = str(txt).lower()
    competitor_lst = [item.lower() for item in competitor_lst]
    pat_comp =r'.*?(\b(?:{})\b).*?'.format('|'.join(competitor_lst))
    
    found_list = re.findall(pat_comp, txt)
    new_list = found_list
    
    # Do not include item found before or after Experience section (Quick & Dirty)
    if ("work experience") in found_list:
        if ("work experience" and "references") in found_list: 
            new_list = found_list[found_list.index('work experience'):found_list.index('references')] 
        else:
            new_list = found_list[found_list.index('work experience'):]
            
    elif ("work history") in found_list:
        if ("work history" and "references") in found_list: 
            new_list = found_list[found_list.index('work history'):found_list.index('references')]
        else:
            new_list = found_list[found_list.index('work history'):]
            
    elif ("employment history") in found_list:
        if ("employment history" and "references") in found_list:
            new_list = found_list[found_list.index('employment history'):found_list.index('references')]
        else:
            new_list = found_list[found_list.index('employment history'):]
    
    elif ("volunteered") in found_list:
        if ("volunteered" and "references") in found_list:
            new_list = found_list[found_list.index('volunteered'):found_list.index('references')]
        else:
            new_list = found_list[found_list.index('volunteered'):]
            
    elif ("references") in found_list:
        new_list = found_list[:found_list.index('references')]
    else:
        new_list = found_list
    
    # Remove any reference word used
    ref_list = ['employment history', 'work experience', "work history", 'references', 'volunteered']
    new_list = [x for x in new_list if x not in ref_list]
    
    return new_list


def add_competitor_flag(df, competitor_lst):
    """
    Add competitor experience flag to employee
    
    :param df: a dataframe containing all companies the employee used to work for
    :param competitor_lst: a list containing all competitors
    :param tot_competitor: total number of competitors in the competitor_lst variable. Proxy for the ammount of experience employee has worked for the competition (e.g experience in the industry of interest to Glentel)
    :return: df with employee code, competitor_experience_list and competitor_experience_flag (1 if employee used to work for a competitor, and 0 otherwise)
    
    """
    
    df["competitor_experience_list"] = df.resume_text.apply(clean_txt).apply(lambda x: find_competitor_list(x, competitor_lst))
    df['tot_competitor'] = df.competitor_experience_list.apply(lambda x: len(list(set(x))))
    df["competitor_experience"] = df.competitor_experience_list.apply(lambda x: 1 if (len(x)> 0) else 0)
    df = df[['employee_code','competitor_experience_list', 'tot_competitor', 'competitor_experience']]
    
    return df


def main(resume_path, output, mode):
    """
    """
    resume = load_data(resume_path)
    
    
    competitors = ["Employment History", "Work History", "Work Experience", 
                   'Freedom Mobile', 'Koodo', 'Shaw', 'Telus', 'Bell', 'Rogers', 'The Mobile Shop','Best Buy', 
                   'Fido', 'Videotron', 'Wow[!]* Mobile', 'The Source', 'Walmart', 'Virgin Mobile', 'Osl', 
                   "References", "volunteered"]
    
    df_final = add_competitor_flag(resume, competitors)
    
    if mode == 'train':
        df_final.to_csv(f'./{output}/auto_competitor_experience.csv', index=False)
    elif mode == 'test':
        df_final.to_csv(f'./{output}/auto_competitor_experience_test.csv', index=False)
    else:
        print("Please pick a test or train mode")
        print()

# main(resume_path, file_outpath)

if __name__ == "__main__":
    main(resume_path=opt["--resume_path"], output=opt["--file_outpath"], mode=opt["--mode"])

print("End Auto Competitor Entity Recognition")