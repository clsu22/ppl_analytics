# author: Manuel Maldonado
# date: 2020-06-05

'''This script generates the following features:highest_degree,background_highest_degree,country_highest_degree

Usage: manual_highest_degree.py --file_path=<file_path> --file_outpath=<file_outpath> --mode=<mode>

Example:
    python scripts/feature_set_01.py --file_path=data/ --file_outpath_path=data/

Options:
--file_path=<file_path>  Path (excluding filenames) to the csv file.
--file_outpath=<file_outpath>  Path for dataframe with created features.
--mode=<mode> Set to train and test
'''
import pandas as pd
import re
import janitor
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from docopt import docopt
import os
opt = docopt(__doc__)

print("start manual highest degree feature extraction")

# For testing
# file_path= '../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_raw_data/'
# file_outpath= '../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/'

def test_function():
    '''
    Tests the input data and specified paths.
    '''
    file_path_check = re.match("([A-Za-z]+[.]{1}[A-Za-z]+)", opt["--file_path"])
    out_path_check = re.match("([A-Za-z]+[.]{1}[A-Za-z]+)", opt["--file_outpath"])
    assert file_path_check == None, "you can not have extensions in path, only directories."
    assert out_path_check == None, "you can not have extensions in path, only directories."
    try:
        os.listdir(opt["--file_path"])
        os.listdir(opt["--file_outpath"])
    except Exception as e:
        print(e)

# test function runs here
test_function()

def main(file_path, file_outpath, mode):

    ####Helper Functions
    def preprocess(text):
        text = str(text)
        text = text.lower()
        text = text.strip()
        text = re.sub('\W+',' ', text)
        text = re.sub("[\d-]", '', text)
        
        # Replace a sequence of whitespaces by a single whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove other strange characters
        text = re.sub(r'''[\*\~]+''', "", text)
        
        #hard code changes 
        if text == 'nan': text = "not_specified"
        if text == 'art': text = 'arts'
        if text == 'liberal arts': text = 'arts'
        if text == 'arts marketing': text = 'marketing'
        if text == 'arts psychology': text = 'psychology'
        if text == 'general arts': text = 'arts'
        
        #if they list two take the first degree
        text = re.split(" and ", text)[0]
        
        #replace word 
        text = re.sub("not mention.+", "not_specified", text)
        text = re.sub("advanced ", '', text)
        text = re.sub("degree", '', text)
        text = re.sub("certificate", '', text)
        text = re.sub("administrat", '', text)
        text = re.sub("management", '', text)
        text = re.sub("progress", '', text)
        text = re.sub("mba", 'business', text)
        text = re.sub("bachelor", '', text)
        text = re.sub("commerce", 'business', text)
        text = re.sub("hr", 'human resource', text)
        text = re.sub("mortgage", 'finance', text)
        text = re.sub("securities", 'finance', text)
        text = re.sub("radio", 'audio', text)
        text = re.sub("tv ", 'television', text)
        text = re.sub('television', 'film', text)
        text = re.sub('health care', 'healthcare', text)
        text = re.sub('hospital ', 'healthcare', text)
        text = re.sub('computing', 'computer', text)
        text = re.sub('techonology', 'technology', text)
        text = re.sub('socialogy', 'sociology', text)
        text = re.sub('child', 'education', text)
        text = re.sub('office', 'general', text)
        text = re.sub('general arts', 'arts', text)
        text = re.sub('public', 'general', text)
        text = re.sub('o s s d', 'general', text)
        text = re.sub('informatics security', 'computer', text) 
        text = re.sub("bank", 'business', text) 
        
        #group degree based off of key words
        if re.search("law", text): text = "law"
        if re.search("dental", text): text = "dental"
        if re.search("financ", text): text = "finance"
        if re.search("account", text): text = "accounting"
        if re.search("human resource", text): text = "human resource"
        if re.search("marketing", text): text = "marketing"
        if re.search("business", text): text = "business"
        if re.search("audio", text): text = "audio technician"
        if re.search("film", text): text = "film production"
        if re.search('healthcare', text): text = "healthcare"
        if re.search('counsel', text): text = "healthcare"
        if re.search('engineering', text): text = "engineering"
        if re.search('communication', text): text = "communication"
        if re.search('education', text): text = 'education'
        if re.search('kinesiology', text): text = 'kinesiology'
        if re.search('computer', text): text = 'computer systems'
        if re.search('software', text): text = 'computer systems'
        if re.search('police', text): text = 'law'
        if re.search('science', text): text = 'science'
        if re.search('english', text): text = 'english'
        if re.search('paralegal', text): text = 'law'
        if re.search('social service', text): text = 'arts'
        if re.search('esthetician', text): text = 'health'
        if re.search('fitness', text): text = 'health'
        if re.search('carpent', text): text = 'blue collar'
        if re.search('plumb', text): text = 'blue collar'
        if re.search('electric', text): text = 'blue collar'
        if re.search('crane', text): text = 'blue collar'
        if re.search('media', text): text = 'interactive arts and technology'
        if re.search('technology', text): text = 'interactive arts and technology'
        if re.search('game', text): text = 'interactive arts and technology'
        if re.search('entertainment', text): text = 'interactive arts and technology'
        if re.search('information', text): text = 'interactive arts and technology'
        if re.search('digital', text): text = 'interactive arts and technology'
        if re.search('medica', text): text = "healthcare"
        if re.search('nurs', text): text = 'healthcare'
        if re.search('pharma', text): text = 'healthcare'
        if re.search('linguist', text): text = 'english'
        if re.search('translation', text): text = 'arts'
        if re.search('general', text): text = 'general'
        if re.search('photo', text): text = 'interactive arts and technology'
        if re.search('screen', text): text = 'interactive arts and technology'
        if re.search('profession', text): text = 'arts'
        if re.search('liberal', text): text = 'arts'
        if re.search('graphic', text): text = 'interactive arts and technology'
        if text == 'not mention': text = "not_specified"    
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def concentration_labeling(con):
        con = str(con)
        if any(x in con for x in ['business']):
            label = "business"
        elif any(x in con for x in ['science']):
            label = "science"
        elif con == 'arts':
            label = "arts"
        elif any(x in con for x in ['dental']):
            label = "dental"
        elif con == 'interactive arts and technology': 
            label = 'interactive arts and technology'
        elif any(x in con for x in ['computer systems']):
            label = 'computer systems'
        elif any(x in con for x in ['hospitality']):
            label = 'hospitality'
        elif any(x in con for x in ['sociology']):
            label = 'sociology'
        elif any(x in con for x in ['finance']):
            label = 'finance'
        elif any(x in con for x in ['marketing']):
            label = 'marketing'
        elif any(x in con for x in ['law']):
            label = 'law'
        elif any(x in con for x in ['engineering']):
            label = 'engineering'
        elif any(x in con for x in ['psychology']):
            label = 'psychology'
        elif any(x in con for x in ['general']):
            label = 'general'
        elif any(x in con for x in ['healthcare']):
            label = 'healthcare'
        elif any(x in con for x in ['human resource']):
            label = 'human resource'
        elif any(x in con for x in ['physics']):
            label = 'physics'
        elif any(x in con for x in ['accounting']):
            label = 'accounting'
        elif any(x in con for x in ['kinesiology']):
            label = 'kinesiology'
        elif any(x in con for x in ['audio technician']):
            label = 'audio technician'
        elif any(x in con for x in ['education']):
            label = 'education'
        elif any(x in con for x in ['blue collar']):
            label = 'blue collar'
        elif any(x in con for x in ['communication']):
            label = 'communication'
        elif any(x in con for x in ['english']):
            label = 'english'
        elif any(x in con for x in ['criminology']):
            label = 'criminology'
        elif any(x in con for x in ['statistics']):
            label = 'statistics'
        elif any(x in con for x in ['economics']):
            label = 'economics'
        elif any(x in con for x in ['not_specified']):
            label = 'not_specified'
        else:
            label = "other" 
        return label

    def background_label_generator(txt):
        txt = preprocess(txt)
        txt = concentration_labeling(txt)
        return txt

    def clean_token(str_value,tpe='t'):
        if len(str(str_value))<2 or str(str_value)=="nan" or ("mention" in str(str_value)):
            str_value = "unknown"
        # "Other" is not considered a stopword, because it was generating labelling issues
        stop_list = stopwords.words('english')
        stop_list.remove('other')
        stop = set(stop_list)
        exclude = set(string.punctuation+"·"+"−"+"§"+"•"+"–"+"––"+"✓")
        lemma = WordNetLemmatizer()
        stop_free = " ".join([i for i in str_value.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized 


    def degree_labeling(deg):
        code = 2
        label = "other"
        if any(x in deg for x in ['phd']):
            code = 8
            label = "phd"
        elif any(x in deg for x in ['master','msc']):
            code = 7
            label = "master"
        elif any(x in deg for x in ['postgrad']):
            code = 6
            label = "postgrad"
        elif any(x in deg for x in ['bac','bsc']):
            code = 5
            label = "bachelor"
        elif any(x in deg for x in ['high','second','ba','grade','school']):
            code = 3
            label = "highschool"
        elif any(x in deg for x in ['prof','certif','techn','continu','advance','college','program','diploma','graduate']):
            code = 4
            label = "certificate"
        elif "unknown" in deg:
            code = 1
            label = "not_specified"
        return [code,label]


    def edu_dict(manual_df):  
        myDict = dict()
        for n in range(4):
            for t in list(['school_','degree_','clean_background','country_']):
                col_name = "education"+str(n+1)+"_"+t
                i = manual_df[col_name]
                clean_i = clean_token(i)
                #clean_i = str(i)
                myDict.setdefault(t, []).append(clean_i)
        ####Labelling Degrees
        myDict["Degree_Code"] = list(map(lambda x: degree_labeling(x)[0], myDict['degree_']))
        myDict["Degree_Label"] = list(map(lambda x: degree_labeling(x)[1], myDict['degree_']))
        pos_highest_degree = myDict['Degree_Code'].index(max(myDict['Degree_Code']))
        myDict["highest_degree"] = myDict['Degree_Label'][pos_highest_degree]
        myDict["background_highest_degree"] = myDict['clean_background'][pos_highest_degree]
        myDict["country_highest_degree"] = myDict['country_'][pos_highest_degree]    
        return myDict


    def highest_degree_flag_01(deg):
        if deg in ['bachelor','master','phd']:
            label = 1
        else:
            label = 0
        return label

    def highest_degree_flag_02(deg):
        if deg in ['highschool']:
            label = 1
        else:
            label = 0
        return label

    def business_flag(deg):
        if deg in ['business']:
            label = 1
        else:
            label = 0
        return label


#### END OF HELPER FUNCTIONS

    # read in data in train test
    if mode == 'train':
        manual_extraction = pd.read_excel(file_path+"manual_extraction_template.xlsx").clean_names()
    elif mode == 'test':
        manual_extraction = pd.read_excel(file_path + "manual_extraction_template_test.xlsx").clean_names()
    else:
        print("Please pick a test or train mode")

    manual_extraction.rename(columns={'education2_country':'education2_country_','education3_country':'education3_country_','education4_country':'education4_country_'},inplace=True)
    manual_extraction['education1_clean_background'] = manual_extraction.education1_concentration_.apply(lambda x: background_label_generator(x))
    manual_extraction['education2_clean_background'] = manual_extraction.education2_concentration_.apply(lambda x: background_label_generator(x))
    manual_extraction['education3_clean_background'] = manual_extraction.education3_concentration_.apply(lambda x: background_label_generator(x))
    manual_extraction['education4_clean_background'] = manual_extraction.education4_concentration_.apply(lambda x: background_label_generator(x))
    manual_extraction['highest_degree'] = manual_extraction.apply(lambda x: edu_dict(x)['highest_degree'],axis=1)
    manual_extraction['background_highest_degree'] = manual_extraction.apply(lambda x: edu_dict(x)['background_highest_degree'],axis=1)
    manual_extraction['country_highest_degree'] = manual_extraction.apply(lambda x: edu_dict(x)['country_highest_degree'],axis=1)
    manual_extraction['flag_hd_bachelor_plus'] = manual_extraction.highest_degree.apply(lambda x:highest_degree_flag_01(x))
    manual_extraction['flag_hd_highschool'] = manual_extraction.highest_degree.apply(lambda x:highest_degree_flag_02(x))
    manual_extraction['business_flag'] = manual_extraction.background_highest_degree.apply(lambda x:business_flag(x))
    ### Exporting Result
    manual_extraction['business_flag'] = manual_extraction.background_highest_degree.apply(lambda x:business_flag(x))

    if mode == 'train':
        manual_extraction[['employee_code','highest_degree','background_highest_degree','country_highest_degree', 'flag_hd_bachelor_plus', 'flag_hd_highschool', 'business_flag']].to_csv(file_outpath + "manual_higher_degree.csv", index= False)
    elif mode == 'test':
        manual_extraction[['employee_code', 'highest_degree', 'background_highest_degree', 'country_highest_degree',
                           'flag_hd_bachelor_plus', 'flag_hd_highschool', 'business_flag']].to_csv(
            file_outpath + "manual_higher_degree_test.csv", index=False)
    else:
        print("Please pick a test or train mode")


# main(file_path, file_outpath)

if __name__ == "__main__":
    main(opt["--file_path"], opt["--file_outpath"], opt['--mode'])

print("end manual highest degree feature extraction")