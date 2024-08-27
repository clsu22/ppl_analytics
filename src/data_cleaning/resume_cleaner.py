# author: Haoyu Su
# date: 2020-06-08

'''This script cleans and process resume data

Usage: resume_cleaner.py --file_path=<file_path> --file_outpath=<file_outpath> --file_cities=<files_cities> --file_otherwords=<file_otherwords>

Options:
--file_path=<file_path>  Path (excluding filenames) to the csv file
--file_outpath=<file_outpath>  Path for dataframe with created features
--file_cities=<files_cities> Path for cities.txt
--file_otherwords=<file_otherwords> Path for other stopwords otherwords.txt

'''

import pandas as pd
import re
import spacy
import os
import numpy as np
from docopt import docopt
opt = docopt(__doc__)

# !python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")


# file_path="../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_interim/english_resumes.csv"
# file_outpath="../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/english_clean_resumes.csv"
# file_cities="../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_external/cities.txt"
# file_otherwords="../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_external/otherwords.txt"

def preprocess(text, pat_names, pat_cities,
                                pat_provinces, pat_otherwords, min_token_len=2,
               irrelevant_pos=['ADV', 'PRON', 'CCONJ', 'PUNCT', 'PART', 'DET',
                               'ADP', 'SPACE']):
    try:
        text = re.sub(r'([\n]+)', ' ', text)
        # Remove Emails
        # text = re.sub(r'\S*@\S*\s?', '', text)
        text = re.sub(r'([\w\.-]+@[\w\.-]+)', '', text)

        # Replace names, cities and provinces
        text = re.sub(pat_names, '', text)

        text = re.sub(pat_cities, '', text)

        text = re.sub(pat_provinces, '', text)

        text = re.sub(pat_otherwords, '', text)

        # Replace a sequence of whitespaces by a single whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove other strange characters
        text = re.sub(r'''[\*\~]+''', "", text)

        doc = nlp(text)
        result = ' '.join([token.lemma_ for token in doc
                           if token.is_stop == False
                           and token.pos_ not in irrelevant_pos
                           and token.is_alpha
                           and not token.like_url
                           and not token.like_email
                           and not token.is_stop
                           # and not token.text.isupper()
                           and not token.text.isdigit()
                           and not token.is_punct
                           and len(token) >= min_token_len])
    except TypeError:
        return ("no resume found")
    return result


def main(file_path, file_outpath, file_cities, file_otherwords):
    print("Start preparing words to be removed")
    # Load data
    eng_resumes = pd.read_csv(file_path)

    # Prepare all employee names
    first_last = eng_resumes.employee_name.str.replace(",", "").apply(
        lambda x: x.split(" ")[1] + " " + x.split(" ")[0]).to_list()
    last_first = eng_resumes.employee_name.str.replace(",", "").apply(
        lambda x: x.split(" ")[0] + " " + x.split(" ")[1]).to_list()
    last = eng_resumes.employee_name.str.replace(",", "").apply(
        lambda x: (x.split(" ")[0])).to_list()
    first = eng_resumes.employee_name.str.replace(",", "").apply(
        lambda x: x.split(" ")[1]).to_list()
    last_upper = list(map(lambda x: x.upper(), last))
    first_upper = list(map(lambda x: x.upper(), first))
    middle = (eng_resumes.employee_name.str.replace(",", "")
              .apply(lambda x: x if len(x.split(" ")) > 2 else "")
              .where(lambda x: x != "")
              .dropna()
              .apply(lambda x: x.split(" ")[2])
              .to_list())
    middle_upper = list(map(lambda x: x.upper(), middle))
    names = first_last + last_first + last + first + middle + first_upper + \
            last_upper + middle_upper
    pat_names = r'\b(?:{})\b'.format('|'.join(names))

    # Prepare all cities
    f = open(file_cities)
    lines = f.readlines()
    f.close()

    cities = [re.match('[0-9]+\:\s(.+)\,\s(.+)', line).group(1) for line in
              lines]
    provinces = [re.match('[0-9]+\:\s(.+)\,\s(.+)', line).group(2) for line in
                 lines]
    cities = list(set(cities))
    provinces = list(set(provinces))
    abbre_prov = ["AB", "BC", "MB", "NB", "NL", "NT", "NS", "NU", "ON", "PE",
                  "QC", "SK", "YT"]
    provinces = provinces + abbre_prov

    pat_cities = r'\b(?:{})\b'.format('|'.join(cities))
    pat_provinces = r'\b(?:{})\b'.format('|'.join(provinces))

    # Prepare otherwords
    f = open(file_otherwords)
    otherwords = [re.sub('\n', '', word) for word in f.readlines()]
    f.close()

    pat_otherwords = r'\b(?:{})\b'.format('|'.join(otherwords))

    print("Finished words preparation")
    print("Start preprocessing")
    # Start preprocessing
    eng_resumes["clean_text"] = eng_resumes['resume_text'].apply(lambda x: preprocess(text=x, 
                                                                                pat_otherwords=pat_otherwords,
                                                                                pat_cities=pat_cities,
                                                                                pat_provinces=pat_provinces,
                                                                                pat_names=pat_names))

    eng_resumes.to_csv(file_outpath)
    print("Finished preprocessing")

if __name__ == "__main__":
    main(opt["--file_path"], opt["--file_outpath"], opt["--file_cities"], opt["--file_otherwords"])

