import pandas as pd
import re
from sklearn.preprocessing import MultiLabelBinarizer

infile_resume_path = "../../../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/06_clean_data/05182020_cleaned_english_resumes_V1.0.csv"
infile_train_path = "../../../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/02_eda_data/output/train_dataset.csv"
outfile_path = '../../../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/06_clean_data/auto_education_concentration.csv'

print("start auto extraction education concentration")
# load and merge the data frames
df = pd.read_csv(infile_resume_path)
df_train = pd.read_csv(infile_train_path)
df_train_code = df_train[['employee_code', 'hp_class']]
df_resume = df[['employee_code', 'employee_name', 'clean_text', 'resume_text']]
df_resume_clean = df_train_code.merge(df_resume, on='employee_code')


# preprocess function
def prepocess(text):
    labels = []

    text = str(text)
    text = text.lower()
    text = text.strip()
    text = re.sub('\W+', ' ', text)
    text = re.sub("[\d-]", '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'''[\*\~]+''', "", text)

    # Replace a sequence of whitespaces by a single whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove other strange characters
    text = re.sub(r'''[\*\~]+''', "", text)

    # try spliting on education
    try:
        text = re.split('educati[^\s]+', text)[1]

        # hardplacement
        if text == 'liberal arts': text = 'arts'
        if text == 'arts marketing': text = 'marketing'
        if text == 'arts psychology': text = 'psychology'
        if text == 'general arts': text = 'arts'
        text = re.sub("mba", 'business', text)
        text = re.sub("commerce", 'business', text)
        text = re.sub("hr", 'human resource', text)
        text = re.sub("mortgage", 'finance', text)
        text = re.sub("securities", 'finance', text)
        text = re.sub('computing', 'computer', text)
        text = re.sub('techonology', 'technology', text)
        text = re.sub('socialogy', 'sociology', text)
        text = re.sub('general arts', 'arts', text)
        text = re.sub('public', 'general', text)
        text = re.sub('o s s d', 'general', text)
        text = re.sub('informatics security', 'computer', text)
        text = re.sub("bank", 'business', text)
        text = re.sub('informatics security', 'computer', text)
        text = re.sub('softwar[^\s]+', 'computer', text)
        text = re.sub('program[^\s]+', 'computer', text)

        # label search
        # accounting
        if re.search('accoun[^\s]+', text): labels.append("accounting_concentration")

        # arts
        # if arts and not digital arts
        if re.search('arts', text) and bool(re.search('digita[^\s]+', text)) == False: labels.append(
            "arts_concentration")
        if re.search('arts', text) and bool(re.search('fine', text)) == False: labels.append("arts_concentration")
        if re.search('sociol[^\s]+', text): labels.append("arts_concentration")
        if re.search('psychol[^\s]+', text): labels.append("arts_concentration")
        if re.search('engli[^\s]+', text): labels.append("arts_concentration")
        if re.search('communicat[^\s]+', text): labels.append("arts_concentration")
        if re.search('criminol[^\s]+', text): labels.append("arts_concentration")

        # business
        if re.search('busine[^\s]+', text): labels.append("business_concentration")
        if re.search('econom[^\s]+', text): labels.append("business_concentration")

        # computer system
        if re.search('comput[^\s]+', text): labels.append("computer_systems_concentration")

        # engineering
        if re.search('enginee[^\s]+', text): labels.append("engineering_concentration")

        # finance
        if re.search('financ[^\s]+', text): labels.append("finance_concentration")

        # human resource
        if re.search('human resour[^\s]+', text): labels.append("human_resource_concentration")

        # interactive arts and techology
        if re.search('media', text): labels.append('interactive_arts_and_technology_concentration')
        if re.search('technol[^\s]', text): labels.append('interactive_arts_and_technology_concentration')
        if re.search('game', text): labels.append('interactive_arts_and_technology_concentration')
        if re.search('entertainm[^\s]', text): labels.append('interactive_arts_and_technology_concentration')
        if re.search('informat[^\s]', text): labels.append('interactive_arts_and_technology_concentration')
        if re.search('digit[^\s]', text): labels.append('interactive_arts_and_technology_concentration')
        if re.search('Photogra[^\s]', text): labels.append('interactive_arts_and_technology_concentration')
        if re.search('scree[^\s]', text): labels.append('interactive_arts_and_technology_concentration')

        # marketing
        if re.search('marke[^\s]+', text): labels.append("marketing_concentration")

        # other
        if not labels:
            labels.append("other_concentration")

    except:
        # try to split on university
        try:
            text = re.split('univers[^\s]+', text)[1]

            # hardplacement
            if text == 'liberal arts': text = 'arts'
            if text == 'arts marketing': text = 'marketing'
            if text == 'arts psychology': text = 'psychology'
            if text == 'general arts': text = 'arts'
            text = re.sub("mba", 'business', text)
            text = re.sub("commerce", 'business', text)
            text = re.sub("hr", 'human resource', text)
            text = re.sub("mortgage", 'finance', text)
            text = re.sub("securities", 'finance', text)
            text = re.sub('computing', 'computer', text)
            text = re.sub('techonology', 'technology', text)
            text = re.sub('socialogy', 'sociology', text)
            text = re.sub('general arts', 'arts', text)
            text = re.sub('public', 'general', text)
            text = re.sub('o s s d', 'general', text)
            text = re.sub('informatics security', 'computer', text)
            text = re.sub("bank", 'business', text)
            text = re.sub('informatics security', 'computer', text)
            text = re.sub('softwar[^\s]+', 'computer', text)
            text = re.sub('program[^\s]+', 'computer', text)

            # label search
            # accounting
            if re.search('accoun[^\s]+', text): labels.append("accounting_concentration")

            # arts
            # if arts and not digital arts
            if re.search('arts', text) and bool(re.search('digita[^\s]+', text)) == False: labels.append(
                "arts_concentration")
            if re.search('arts', text) and bool(re.search('fine', text)) == False: labels.append("arts_concentration")
            if re.search('sociol[^\s]+', text): labels.append("arts_concentration")
            if re.search('psychol[^\s]+', text): labels.append("arts_concentration")
            if re.search('engli[^\s]+', text): labels.append("arts_concentration")
            if re.search('communicat[^\s]+', text): labels.append("arts_concentration")
            if re.search('criminol[^\s]+', text): labels.append("arts_concentration")

            # business
            if re.search('busine[^\s]+', text): labels.append("business_concentration")
            if re.search('econom[^\s]+', text): labels.append("business_concentration")

            # computer system
            if re.search('comput[^\s]+', text): labels.append("computer_systems_concentration")

            # engineering
            if re.search('enginee[^\s]+', text): labels.append("engineering_concentration")

            # finance
            if re.search('financ[^\s]+', text): labels.append("finance_concentration")

            # human resource
            if re.search('human resour[^\s]+', text): labels.append("human_resource_concentration")

            # interactive arts and techology
            if re.search('media', text): labels.append('interactive_arts_and_technology_concentration')
            if re.search('technol[^\s]', text): labels.append('interactive_arts_and_technology_concentration')
            if re.search('game', text): labels.append('interactive_arts_and_technology_concentration')
            if re.search('entertainm[^\s]', text): labels.append('interactive_arts_and_technology_concentration')
            if re.search('informat[^\s]', text): labels.append('interactive_arts_and_technology_concentration')
            if re.search('digit[^\s]', text): labels.append('interactive_arts_and_technology_concentration')
            if re.search('Photogra[^\s]', text): labels.append('interactive_arts_and_technology_concentration')
            if re.search('scree[^\s]', text): labels.append('interactive_arts_and_technology_concentration')

            # marketing
            if re.search('marke[^\s]+', text): labels.append("marketing_concentration")

            # other
            if not labels:
                labels.append("no_education_section")

        except:
            # try spliting on college
            try:
                text = re.split('univers[^\s]+', text)[1]

                # hardplacement
                if text == 'liberal arts': text = 'arts'
                if text == 'arts marketing': text = 'marketing'
                if text == 'arts psychology': text = 'psychology'
                if text == 'general arts': text = 'arts'
                text = re.sub("mba", 'business', text)
                text = re.sub("commerce", 'business', text)
                text = re.sub("hr", 'human resource', text)
                text = re.sub("mortgage", 'finance', text)
                text = re.sub("securities", 'finance', text)
                text = re.sub('computing', 'computer', text)
                text = re.sub('techonology', 'technology', text)
                text = re.sub('socialogy', 'sociology', text)
                text = re.sub('general arts', 'arts', text)
                text = re.sub('public', 'general', text)
                text = re.sub('o s s d', 'general', text)
                text = re.sub('informatics security', 'computer', text)
                text = re.sub("bank", 'business', text)
                text = re.sub('informatics security', 'computer', text)
                text = re.sub('softwar[^\s]+', 'computer', text)
                text = re.sub('program[^\s]+', 'computer', text)

                # label search
                # accounting
                if re.search('accoun[^\s]+', text): labels.append("accounting_concentration")

                # arts
                # if arts and not digital arts
                if re.search('arts', text) and bool(re.search('digita[^\s]+', text)) == False: labels.append(
                    "arts_concentration")
                if re.search('arts', text) and bool(re.search('fine', text)) == False: labels.append(
                    "arts_concentration")
                if re.search('sociol[^\s]+', text): labels.append("arts_concentration")
                if re.search('psychol[^\s]+', text): labels.append("arts_concentration")
                if re.search('engli[^\s]+', text): labels.append("arts_concentration")
                if re.search('communicat[^\s]+', text): labels.append("arts_concentration")
                if re.search('criminol[^\s]+', text): labels.append("arts_concentration")

                # business
                if re.search('busine[^\s]+', text): labels.append("business_concentration")
                if re.search('econom[^\s]+', text): labels.append("business_concentration")

                # computer system
                if re.search('comput[^\s]+', text): labels.append("computer_systems_concentration")

                # engineering
                if re.search('enginee[^\s]+', text): labels.append("engineering_concentration")

                # finance
                if re.search('financ[^\s]+', text): labels.append("finance_concentration")

                # human resource
                if re.search('human resour[^\s]+', text): labels.append("human_resource_concentration")

                # interactive arts and techology
                if re.search('media', text): labels.append('interactive_arts_and_technology_concentration')
                if re.search('technol[^\s]', text): labels.append('interactive_arts_and_technology_concentration')
                if re.search('game', text): labels.append('interactive_arts_and_technology_concentration')
                if re.search('entertainm[^\s]', text): labels.append('interactive_arts_and_technology_concentration')
                if re.search('informat[^\s]', text): labels.append('interactive_arts_and_technology_concentration')
                if re.search('digit[^\s]', text): labels.append('interactive_arts_and_technology_concentration')
                if re.search('Photogra[^\s]', text): labels.append('interactive_arts_and_technology_concentration')
                if re.search('scree[^\s]', text): labels.append('interactive_arts_and_technology_concentration')

                # marketing
                if re.search('marke[^\s]+', text): labels.append("marketing_concentration")

                # other
                if not labels:
                    labels.append("no_education_section")

            except:
                labels.append("no_education_section")

    return labels


df_resume_clean["educatation_concentration_list"] = df_resume_clean["clean_text"].apply(prepocess)
mlb = MultiLabelBinarizer()
df_resume_clean = df_resume_clean.join(
    pd.DataFrame(mlb.fit_transform(df_resume_clean.pop('educatation_concentration_list')),
                 columns=mlb.classes_,
                 index=df_resume_clean.index))

# save files
df_resume_clean = df_resume_clean.drop(['hp_class', 'employee_name', 'clean_text', 'resume_text'], axis=1)
df_resume_clean.to_csv(outfile_path)

print("finish auto extraction education concentration")
