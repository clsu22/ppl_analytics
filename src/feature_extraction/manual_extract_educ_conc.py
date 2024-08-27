# author: Thomas Pin
# date: 2020-06-05

'''This script generates the following features: 
accounting_concentration, arts_concentration, audio_technician_concentration, business_concentration, communication_concentration, computer_systems_concentration, criminology_concentration, dental_concentration, economics_concentration, education_concentration, engineering_concentration, english_concentration, finance_concentration, general_concentration, healthcare_concentration, hospitality_concentration, human_resource_concentration,interactive arts and technology, kinesiology_concentration, law_concentration, marketing_concentration, no_education_specified, not_specified_concentration, other_concentration, physics_concentration, psychology_concentration, science_concentration, sociology_concentration, statistics_concentration


Usage: manual_extract_educ_conc.py --file_path=<file_path> --file_outpath=<file_outpath> --mode=<mode>

Example:
    python src/feature_extraction/manual_extract_educ_conc.py --file_path=../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/01_resume_scan_data/manual_extraction_template.xlsx --file_outpath=../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_feature/education_concentration.csv

Options:
--file_path=<file_path>  Path (excluding filenames) to the csv file.
--file_outpath=<file_outpath>  Path for dataframe with created features.
--mode=<mode> Set to train and test
'''

import pandas as pd
import numpy as np
import re
from docopt import docopt
from sklearn.preprocessing import MultiLabelBinarizer
opt = docopt(__doc__)

# file_path = "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/01_resume_scan_data/manual_extraction_template.xlsx"
# file_outpath = '../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/manual_education_concentration.csv'


print("Start education concentration extraction")



def main(file_path, file_outpath, mode):
    #read in files
    extract = pd.read_excel(file_path)

    extract.columns = extract.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')','')

    #helper funcation 
    def preprocess(text):
        """
        Preprocess the education concentrationn text for labeling
        
        Parameters
        ----------
        text : str
            education concentration column from manual extraction template
        
        Returns:
        ----------
        str
            preprocessed str ready for labeling

        """
        text = str(text)
        text = text.lower()
        text = text.strip()
        text = re.sub('\W+', ' ', text)
        text = re.sub("[\d-]", '', text)

        # Replace a sequence of whitespaces by a single whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove other strange characters
        text = re.sub(r'''[\*\~]+''', "", text)

        # hard code changes
        if text == 'nan': text = "not_specified"
        if text == 'art': text = 'arts'
        if text == 'liberal arts': text = 'arts'
        if text == 'arts marketing': text = 'marketing'
        if text == 'arts psychology': text = 'psychology'
        if text == 'general arts': text = 'arts'

        # if they list two take the first degree
        text = re.split(" and ", text)[0]

        # replace word
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

        # group degree based off of key words
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

        # text = re.sub("arts ", '', text)

        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text


    def concentration_labeling(con):
        """
        Uses the processed education concentration text and assigns final labeling
        
        Parameters
        ----------
        con : str
            preprocessed education concentration text
        
        Returns:
        ----------
        list
            a list of education concentration employees have

        """
        con = str(con)
        if any(x in con for x in ['business']):
            label = "business_concentration"
        elif any(x in con for x in ['science']):
            label = "science_concentration"
        elif con == 'arts': label = "arts_concentration"
        elif any(x in con for x in ['dental']):
            label = "dental_concentration"
        elif con == 'interactive arts and technology': label = 'interactive_arts_and technology_concentration'
        elif any(x in con for x in ['computer systems']):
            label = 'computer_systems_concentration'
        elif any(x in con for x in ['hospitality']):
            label = 'hospitality_concentration'
        elif any(x in con for x in ['sociology']):
            label = 'sociology_concentration'
        elif any(x in con for x in ['finance']):
            label = 'finance_concentration'
        elif any(x in con for x in ['marketing']):
            label = 'marketing_concentration'
        elif any(x in con for x in ['law']):
            label = 'law_concentration'
        elif any(x in con for x in ['engineering']):
            label = 'engineering_concentration'
        elif any(x in con for x in ['psychology']):
            label = 'psychology_concentration'
        elif any(x in con for x in ['general']):
            label = 'general_concentration'
        elif any(x in con for x in ['healthcare']):
            label = 'healthcare_concentration'
        elif any(x in con for x in ['human resource']):
            label = 'human_resource_concentration'
        elif any(x in con for x in ['physics']):
            label = 'physics_concentration'
        elif any(x in con for x in ['accounting']):
            label = 'accounting_concentration'
        elif any(x in con for x in ['kinesiology']):
            label = 'kinesiology_concentration'
        elif any(x in con for x in ['audio technician']):
            label = 'audio_technician_concentration'
        elif any(x in con for x in ['education']):
            label = 'education_concentration'
        elif any(x in con for x in ['communication']):
            label = 'communication_concentration'
        elif any(x in con for x in ['english']):
            label = 'english_concentration'
        elif any(x in con for x in ['criminology']):
            label = 'criminology_concentration'
        elif any(x in con for x in ['statistics']):
            label = 'statistics_concentration'
        elif any(x in con for x in ['economics']):
            label = 'economics_concentration'
        elif any(x in con for x in ['not_specified']):
            label = 'not_specified_concentration'
        else:
            label = "other_concentration" 
        return label

    #Run preprocess on dataframe
    for i in range(1, 4):
        col_num = i
        extract["education" + str(col_num) + "_concentration_clean"] = extract[
            "education" + str(col_num) + "_concentration"].apply(preprocess)
    #run labeling on 
    for i in range(1, 4):
        col_num = i
        extract["education" + str(col_num) + "_label"] = extract["education" + str(col_num) + "_concentration_clean"].apply(
            concentration_labeling)
    
    #reformate to proper lables 
    result = []
    for i in range(0, len(extract['education1_label'])):
        lst = []
        if (str(extract['education1_label'][i]) == 'not_specified_concentration' and str(
                extract['education2_label'][i]) == 'not_specified_concentration' and str(
            extract['education3_label'][i]) == 'not_specified_concentration'):
            lst.append("no_education_specified")
        if extract['education1_label'][i] != 'not_specified':
            lst.append(extract['education1_label'][i])
        if extract['education2_label'][i] != 'not_specified':
            lst.append(extract['education2_label'][i])
        if extract['education3_label'][i] != 'not_specified':
            lst.append(extract['education3_label'][i])
        result.append(lst)

    extract["education_list"] = result

    #reformate for model downstream
    mlb = MultiLabelBinarizer()
    extract = extract.join(pd.DataFrame(mlb.fit_transform(extract.pop('education_list')),
                                        columns=mlb.classes_,
                                        index=extract.index))

    extract = extract.drop(['found', 'work1_title',
                            'work1_company', 'work1_time', 'work2_title', 'work2_company',
                            'work2_time', 'work3_title', 'work3_company', 'work3_time',
                            'work4_title', 'work4_company', 'work4_time', 'work5_title',
                            'work5_company', 'work5_time', 'work6_title', 'work6_company',
                            'work6_time', 'work7_title', 'work7_company', 'work7_time',
                            'education1_school', 'education1_degree', 'education1_concentration',
                            'education1_country', 'education2_school', 'education2_degree',
                            'education2_concentration', 'education2_country', 'education3_school',
                            'education3_degree', 'education3_concentration', 'education3_country',
                            'education4_school', 'education4_degree', 'education4_concentration',
                            'education4_country', 'education1_concentration_clean',
                            'education2_concentration_clean', 'education3_concentration_clean'], axis=1)

    #save csv

    # train test mode
    if mode == "test":
        extract['human_resource_concentration'] = np.zeros(extract.shape[0])

    extract.to_csv(file_outpath)
    return extract

if __name__ == '__main__':
    main(opt["--file_path"], opt["--file_outpath"], opt["--mode"])


print("Finish education concentration extraction")
