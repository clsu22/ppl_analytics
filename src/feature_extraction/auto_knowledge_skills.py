# author: Manuel Maldonado
# date: 2020-06-05

'''This script generates the following features: 'no_lang_spoken', 'trilingual_flag', 'goal_record','sales_customer_base_exp', 'volunteer_exp', 'problem_solver',
'sports_mention', 'communication_skills', 'team_player','leadership_mention'

Usage: auto_knowledge_skills.py --file_path=<file_path> --file_outpath=<file_outpath> --mode=<mode>

Example:
    python scripts/knowledge_skills.py --file_path=data/ --file_outpath=data/

Options:
--file_path=<file_path>  Path (excluding filenames) to csv file containing the extracted resume data.
--file_outpath=<file_outpath>  Path for dataframe with created features.
--mode=<mode> Set to train and test
'''

import pandas as pd
import re
import os
from docopt import docopt

print("Start Auto Knowledge Skill Feature Extraction")

opt = docopt(__doc__)

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

#For Testing
# file_path = '../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/'
# file_outpath = '../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/'


def main(file_path, file_outpath, mode):
    ####Helper Functions
    def goal_record(txt):
        txt = str(txt).lower()
        if any(x in txt for x in ['target', 'achiev', 'award']):
            flag = 1
        else:
            flag = 0
        return flag

    def sales_customer_base_exp(txt):
        txt = str(txt).lower()
        if any(x in txt for x in ['sales']) and any(x in txt for x in ['customer']):
            flag = 1
        else:
            flag = 0
        return flag

    def volunteer_exp(txt):
        txt = str(txt).lower()
        if any(x in txt for x in ['volun']):
            flag = 1
        else:
            flag = 0
        return flag

    def problem_solver(txt):
        txt = str(txt).lower()
        if any(x in txt for x in ['solution', 'answers', 'analysis', 'decision']):
            flag = 1
        else:
            flag = 0
        return flag

    def sports_mention(txt):
        txt = str(txt).lower()
        if re.search(r"\bsport", txt):
            flag = 1
        else:
            flag = 0
        return flag

    def communication_skills(txt):
        txt = str(txt).lower()
        if any(x in txt for x in ['communic']):
            flag = 1
        else:
            flag = 0
        return flag

    def team_player(txt):
        txt = str(txt).lower()
        if any(x in txt for x in ['team']):
            flag = 1
        else:
            flag = 0
        return flag

    def leadership_mention(txt):
        txt = str(txt).lower()
        if any(x in txt for x in ['lead']):
            flag = 1
        else:
            flag = 0
        return flag

    def identify_languages(resume_text):
        myDict = dict()
        spoken_lang = list()
        lang_list = ['spanish', 'french', 'german', 'punjabi', 'hindi', 'urdu', 'arabic', 'mandarin', 'dari',
                     'japanese', 'filipino', 'tamil', 'cantonese', 'russian']
        # 'tagalog' --removed for possible double counting with filipino
        # lang_list = ['fluent','lang']
        txt = str(resume_text)
        txt = resume_text.lower()
        spoken_lang.append('english')
        for x in lang_list:
            if x in txt:
                spoken_lang.append(x)
                # removing duplicate languages (mentioned more than once in resume)
        spoken_lang = list(dict.fromkeys(spoken_lang))
        no_languages_spoken = len(spoken_lang)
        if no_languages_spoken > 2:
            trilingual_flag = 1
        else:
            trilingual_flag = 0
        myDict['spoken_languages'] = spoken_lang
        myDict['no_lang'] = no_languages_spoken
        myDict['trilingual_flag'] = trilingual_flag
        return myDict

    #### END OF HELPER FUNCTIONS

    # read in data

    cleaned_english_resume = pd.read_csv(file_path + 'english_clean_resumes.csv', index_col=0)
    if mode == "train":
        training_df = pd.read_csv(file_path + 'train_dataset.csv', index_col=0)
    elif mode == 'test':
        training_df = pd.read_csv(file_path + 'test_dataset.csv', index_col=0)
    else:
        print("Please pick a test or train mode")

    data = pd.merge(
        cleaned_english_resume[['employee_code', 'raw_resume', 'resume_text', 'resume_bline', 'clean_text']],
        training_df[['employee_code', 'hp_class']], on="employee_code", how="inner")

    ## Creating Features
    data['no_lang_spoken'] = data.resume_text.apply(lambda x: identify_languages(x)['no_lang'])
    data['trilingual_flag'] = data.resume_text.apply(lambda x: identify_languages(x)['trilingual_flag'])
    data['goal_record'] = data.resume_text.apply(lambda x: goal_record(x))
    data['sales_customer_base_exp'] = data.resume_text.apply(lambda x: sales_customer_base_exp(x))
    data['volunteer_exp'] = data.resume_text.apply(lambda x: volunteer_exp(x))
    data['problem_solver'] = data.resume_text.apply(lambda x: problem_solver(x))
    data['sports_mention'] = data.resume_text.apply(lambda x: sports_mention(x))
    data['communication_skills'] = data.resume_text.apply(lambda x: communication_skills(x))
    data['team_player'] = data.resume_text.apply(lambda x: team_player(x))
    data['leadership_mention'] = data.resume_text.apply(lambda x: leadership_mention(x))

    ### Exporting Result
    data = data.drop(['raw_resume', 'hp_class', 'resume_text', 'resume_bline', 'clean_text'], axis=1)

    # train test mode
    if mode == "train":
        data.to_csv(file_outpath + 'auto_knowledge_skills.csv', index=False)
    elif mode == 'test':
        data.to_csv(file_outpath + 'auto_knowledge_skills_test.csv', index=False)
    else:
        print("Please pick a test or train mode")





# main(file_path, file_outpath)

if __name__ == "__main__":
    main(opt["--file_path"], opt["--file_outpath"], opt["--mode"])

print("Finish Auto Knowledge Skill Feature Extraction")
