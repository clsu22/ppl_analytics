# author: Thomas Pin
# date: 2020-06-05

'''This script generates the following features: 

Usage: manual_extract_job_title.py --file_path=<file_path> --file_outpath=<file_outpath> --mode=<mode>

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
import textdistance
from sklearn.preprocessing import MultiLabelBinarizer
from docopt import docopt
opt = docopt(__doc__)

#for testing
# file_path = "../../../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/01_resume_scan_data/manual_extraction_template.xlsx"
# file_outpath ='../../../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/06_clean_data/work_title.csv'


print("Start jobtitle extraction") 

def main(file_path, file_outpath, mode):
    #load excel
    extract = pd.read_excel(file_path)
    extract.columns = extract.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

    #clean text
    #helper functions
    def preprocess(text):
        """
        Preprocess the jobtitle text for labeling
        
        Parameters
        ----------
        text : str
            jobtitle column from manual extraction template
        
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

        # change text
        text = re.sub("part time", '', text)
        text = re.sub("seasona[^\s]+", '', text)
        text = re.sub("superv[^\s]+", 'manager', text)
        text = re.sub("team.+lea[^\s]+", 'manager', text)
        text = re.sub("wait[er][er]", 'server', text)
        text = re.sub("chie[^\s]+", 'culinary', text)
        text = re.sub("chef", 'culinary', text)
        text = re.sub("bran[^\s]+", 'sales', text)
        text = re.sub("instr[^\s]+", 'teacher', text)
        text = re.sub("gues[^\s]+", 'customer', text)
        text = re.sub("clien[^\s]+", 'customer', text)
        text = re.sub("kios[^\s]+", 'retail', text)
        text = re.sub("sale[^s]", 'sales', text)
        text = re.sub("specia[^s]", '', text)

        # group jobtiles
        # sales associate
        if re.search('mobil[^\s]+', text): text = 'sales associate'
        if re.search('sales a([^\sn]+)', text): text = 'sales associate'
        if re.search('sal.+re[^\s]+', text): text = 'sales associate'
        if re.search('sales.+[pca]', text): text = 'sales associate'
        if re.search('sales.+ex[^\s]+', text): text = 'sales associate'

        # assistant manager
        if re.search('assis.+ manager', text): text = 'assistant manager'

        # manager
        if re.search('s[ta][ol][er].+manager', text): text = 'manager'
        if re.search('service.+manager', text): text = 'manager'
        if re.search('sale.+man', text): text = 'manager'
        if any(x in text for x in ['manager']) and any(x not in text for x in ['assistant']): text = 'manager'

        # customer service representativ
        if re.search('customer.+[sc][^\s]+', text): text = 'customer service representative'
        if re.search('clerk', text): text = 'customer service representative'
        if re.search('team [m]', text): text = 'customer service representative'

        # server
        if re.search('serve', text): text = 'server'
        if re.search('barte', text): text = 'server'
        if re.search('host', text): text = 'server'

        # cashier
        if re.search('cashi', text): text = 'cashier'

        # education
        if re.search('teach[^\s]+', text): text = 'education'
        if re.search('lectur[^\s]+', text): text = 'education'
        if re.search('tutor', text): text = 'education'
        if re.search('educat', text): text = 'education'
        if re.search('student', text): text = 'education'
        if re.search('education', text): text = 'education'

        # culinary
        if re.search('culinary', text): text = 'culinary'
        if re.search('kitchen', text): text = 'culinary'
        if re.search('cook', text): text = 'culinary'

        # adminstration
        if re.search('administr[^\s]+', text): text = 'administrative'
        if re.search('office', text): text = 'administrative'
        if re.search('executi', text): text = 'administrative'
        if re.search('coordinator', text): text = 'administrative'
        if re.search('auditor', text): text = 'administrative'

        # driver
        if re.search('drive[^\s]+', text): text = 'driver'
        if re.search('deliv', text): text = 'driver'

        # blue collar
        if re.search('labo[^\s]+', text): text = 'blue collar'
        if re.search('electrici', text): text = 'blue collar'
        if re.search('plumber', text): text = 'blue collar'
        if re.search('carpent', text): text = 'blue collar'
        if re.search('construc', text): text = 'blue collar'
        if re.search('renovat', text): text = 'blue collar'
        if re.search('manpower', text): text = 'blue collar'

        # technicians
        if re.search('technici', text): text = 'technician'

        # fitness/sports
        if re.search('coach', text): text = 'fitness/sports'
        if re.search('fitnes', text): text = 'fitness/sports'
        if re.search('traine', text): text = 'fitness/sports'
        if re.search('referee', text): text = 'fitness/sports'

        # financial services
        if re.search('financia', text): text = 'financial services'
        if re.search('analy', text): text = 'financial services'
        if re.search('bookee', text): text = 'financial services'
        if re.search('mortgag', text): text = 'financial services'
        if re.search('broker', text): text = 'financial services'

        # telemarketers
        if re.search('call cent', text): text = 'telemarketers'

        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        if text == '': text = "not_specified"
        return text

    categorical_list = ['sales associate', 'assistant manager', 'customer service representative', 'cashier', 'education', 'cook', 'administrative', 'driver', 'blue collar', 'technicians', 'financial services', 'telemarketers', 'fitness/sports', 'not_specified', 'manager', 'server']
    
    #helper function
    def worktitle_labeling(text, categorical_list=categorical_list, word_tolarance = .3):
        """
        Uses the processed jobtitle text and assigns final labeling
        
        Parameters
        ----------
        text : str
            preprocessed jobtitle text
        categorical_list : list
            List of all potential job tiles labels
        word_tolarance : int
            Tolerance for word similarity for grouping non-obvious words in the
            textdistance.levenshtein.normalized_similarity 
            
        
        Returns:
        ----------
        list
            a list of jobtitle of employees

        """
        label = ""
        if any(x in text for x in categorical_list):
            label = text
        else:
            results = {}
            for i in range(0, len(categorical_list)):
                simil = textdistance.levenshtein.normalized_similarity(text, categorical_list[i])
                results[str(categorical_list[i])] = simil
                results_sort = sorted(results.items(), key=lambda x: x[1], reverse=True)

            if results_sort[0][1] > word_tolarance:
                label = results_sort[0][0]
            else:
                label = "other_jobtitle"
        return label

    #apply preprocessing to text
    for i in range(1, 8):
        col_num = i
        extract['work' + str(col_num) + '_title_clean'] = extract['work' + str(col_num) + '_title'].apply(preprocess)
    
    #assign lables
    for i in range(1, 8):
        col_num = i
        extract['work' + str(col_num) + '_title_label'] = extract['work' + str(col_num) + '_title_clean'].apply(
            worktitle_labeling)

    result = []
    for i in range(0, len(extract['work1_title_label'])):
        lst = []
        if (str(extract['work1_title_label'][i]) == 'not_specified' and str(extract['work2_title_label'][i]) == 'not_specified' and str(extract['work3_title_label'][i]) == 'not_specified' and str(extract['work4_title_label'][i]) == 'not_specified' and str(extract['work5_title_label'][i]) == 'not_specified' and str(extract['work6_title_label'][i]) == 'not_specified'and str(extract['work7_title_label'][i]) == 'not_specified'):
            lst.append("no_work_title")
        if extract['work1_title_label'][i] != 'not_specified':
            lst.append(extract['work1_title_label'][i])
        if extract['work2_title_label'][i] != 'not_specified':
            lst.append(extract['work2_title_label'][i])
        if extract['work3_title_label'][i] != 'not_specified':
            lst.append(extract['work3_title_label'][i])
        if extract['work4_title_label'][i] != 'not_specified':
            lst.append(extract['work4_title_label'][i])
        if extract['work5_title_label'][i] != 'not_specified':
            lst.append(extract['work5_title_label'][i])
        if extract['work6_title_label'][i] != 'not_specified':
            lst.append(extract['work6_title_label'][i])
        if extract['work7_title_label'][i] != 'not_specified':
            lst.append(extract['work7_title_label'][i])
        result.append(lst)

    extract["work_title_list"] = result

    #reformat labels for downstream model
    mlb = MultiLabelBinarizer()
    extract = extract.join(pd.DataFrame(mlb.fit_transform(extract.pop('work_title_list')),
                                       columns = mlb.classes_,
                                       index= extract.index))

    categorical_list = ['sales associate', 'assistant manager', 'customer service representative', 'cashier', 'education', 'cook', 'administrative', 'driver', 'blue collar', 'technicians', 'financial services', 'telemarketers', 'fitness/sports', 'not_specified', 'manager', 'server']
    final_label = categorical_list.copy()

    for i in range(0, len(categorical_list)):
        final_label[i] = re.sub('\s+', "_", categorical_list[i].strip())
        final_label[i] = re.sub('/', "_", final_label[i])
        final_label[i] = final_label[i]+"_jobtitle"

    results ={}
    for i in range(0, len(categorical_list)):
        results[categorical_list[i]] = final_label[i]

    extract_work_title = extract.drop(['found', 'work1_title',
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
           'education4_country', 'work1_title_clean', 'work2_title_clean',
           'work3_title_clean', 'work4_title_clean', 'work5_title_clean',
           'work6_title_clean', 'work7_title_clean'], axis=1)

    #save to csv
    extract_work_title = extract_work_title.rename(columns=results)

    if mode =='test':
        extract_work_title["telemarketers_jobtitle"] = np.zeros(extract_work_title.shape[0])
        extract_work_title["no_work_title"] = np.zeros(extract_work_title.shape[0])

    extract_work_title.to_csv(file_outpath)
    
if __name__ == '__main__':
    main(opt["--file_path"], opt["--file_outpath"], opt["--mode"])
    
print("Finish jobtitle extraction")   