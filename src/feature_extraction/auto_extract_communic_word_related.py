# author: Robert Pimentel
# date: 2020-06-09

'''This script generates the following features: team_word, team_player_phrase

Usage: auto_fe_team_word_related.py --file_path=<file_path> --output_path=<output_path>

Example:
python auto_extract_communic_word_related.py --file_path="data/input/" --output_path="data/output/"

Options:
--file_path=<file_path>  Path (excluding filenames) to the .csv file.
--output_path=<output_path>  Path for exporting results in .csv file
'''
import pandas as pd
import re 
import spacy 
import os 
import numpy as np
from docopt import docopt

opt = docopt(__doc__)

def test_function():
    '''
    Check that the input data and specified directories paths are correct
    '''
    file_path_check = re.match("([A-Za-z]+[.]{1}[A-Za-z]+)", opt["--file_path"]) 
    out_path_check = re.match("([A-Za-z]+[.]{1}[A-Za-z]+)", opt["--output_path"])
    assert file_path_check == None, "you can not have extensions in dir_path, only directories."
    assert out_path_check == None, "you can not have extensions in dir_path, only directories."
    try:
        os.listdir(opt["--file_path"])
        os.listdir(opt["--output_path"])
    except Exception as e:
        print(e)

# Run Test function
test_function()

opt = docopt(__doc__)

def main(file_path, output_path):

    def communication_skills_manual(txt):
        txt = str(txt).lower()
        if any(x in txt for x in ['communic']):
            flag = 1
        else:
            flag = 0
        return flag

    def communication_skills_phrase(txt):
        txt = str(txt).lower()
        #if re.search("communication skills", txt):
        if re.search(r"\b(comm?unic?[^\s]+ ?skill?s?)\b", txt):
            flag = 1
        else:
            flag = 0
        return flag
    
    def communic_word(txt):
        txt = str(txt).lower()
        if re.search(r'\b(comm?unic[^\s]+)\b', txt):
            flag = 1
        else:
            flag = 0
        return flag



    # READ DATA
###############Enable these when testing in commong share drive###########################
    #df = pd.read_csv("../../../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/06_clean_data/05182020_cleaned_english_resumes_V1.
    #df_train = pd.read_csv("../../../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/02_eda_data/output/0607_02_training_dataset.csv")
###################################################################################################    
    
    print()
    print('Runing "COMMUNICATION WORD ROOT" related features')
    print("Step1: Reading Files from input folder {}".format(file_path))
    df = pd.read_csv(file_path+"05182020_cleaned_english_resumes_V1.0.csv")
    df_train = pd.read_csv(file_path+"0607_02_training_dataset.csv")
    
    # Crate Working Dataframe
    df_train_code = df_train[['employee_code', 'hp_class']]
    df_resume = df[['employee_code', 'employee_name', 'clean_text', 'resume_text']]
    df_resume_clean = df_train_code.merge(df_resume, on='employee_code')
    
    print("Step2: Generating features: communication_skills_manual, communication_skills_phrase, communication_word_root")
    # Generate Features
    df_resume_clean['communication_skills_manual'] = df_resume_clean.resume_text.apply(lambda x: communication_skills_manual(x))
    df_resume_clean['communication_skills_phrase'] = df_resume_clean.resume_text.apply(lambda x: communication_skills_phrase(x))
    df_resume_clean['communication_word_root'] = df_resume_clean.resume_text.apply(lambda x: communic_word(x))

   
    # Exporting Results
    output_path_filename = output_path+"auto_communic_word_related_features.csv"
    print("Step3: Writing csv file to {}".format(output_path_filename))
    
    df_resume_clean = df_resume_clean.drop(['hp_class', 'employee_name', 'clean_text', 'resume_text'], axis=1)   
    df_resume_clean.to_csv(output_path_filename, index= False)

if __name__ == "__main__":
    main(opt["--file_path"], opt["--output_path"])
    