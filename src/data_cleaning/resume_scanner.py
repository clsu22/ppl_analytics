# author: Haoyu Su
# date: 2020-06-08

'''This script is used to read all resumes in from Career Builder (Resumes) folder:

Usage: resume_scanner.py --file_outpath_all=<file_outpath_all> --file_outpath_english=<file_outpath_english> --file_path_match=<file_path_match>

Example:

Options:
--file_outpath_all=<file_outpath_all> Path all_langueage_reusmes to the csv
--file_outpath_english=<file_outpath_english> Path english_reusmes to csv file
--file_path_match=<file_path_match> Path manually checked MS Excel file

'''

from tika import parser
import pandas as pd
import os
import spacy
import numpy as np
from IPython.display import clear_output
import re
from docopt import docopt
opt = docopt(__doc__)

#For testing
# file_outpath_all = "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/all_language_resumes.csv"
# file_outpath_english = "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/english_resumes.csv"
# file_path_match = "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_raw_data/05182020_matching_list_maunually-checked_V1.2.xlsx"

print("Start resume scanner")

def get_names(dirname):
    """
    Get employee names, store ids and filenames
    """
    employee_names = []
    store_ids = []
    filepaths = []
    uids = []
    for root, dirs, files in os.walk(dirname):
        for item in files:
            if ".xlsx" in item:
                skip = True
                continue
            else:
                filename = os.path.join(root, item)
                filepaths.append(filename)
                item = re.sub('[\s]+', ' ',
                              re.sub('^\s\-\s', ' - ',
                                     re.sub('[\(\)]+', ' - ', item)))
                item = re.sub(' - not formatted', ' ', item)
                item = re.sub('(\s\-)[A-Z0-9]', ' - ', item)
                # item = re.sub('[A-Z0-9](\-\s)[A-Z0-9]', ' - ', item)
                item = re.sub('.pdf', ' ', item)
                item = re.sub('.rtf', ' ', item)
                item = re.sub('.doc[x]*', ' ', item)
                item = re.sub('[\s]+', ' ', item)
                # id = re.findall('[A-Z0-9]{3}', item)
                splits = item.split(" - ")
                employee_name = splits[0]
                # print(employee_name)
                # print(item)
                if len(splits) != 1:
                    # print(item)
                    store_id = splits[1]
                    if len(splits) == 3:
                        uid = splits[2]
                    else:
                        uid = ''
                        print(employee_name, "don't have uid")
                    # print(store_id)
                else:
                    store_id = ''
                    print(employee_name, "don't have store id")
                # print(uid)
                employee_names.append(employee_name)
                store_ids.append(store_id)
                uids.append(uid)
    # print("get names: finished")
    return employee_names, store_ids, filepaths, uids


def tika_parser(file_path):
    file_data = parser.from_file(file_path)
    text = file_data['content']
    return text


def convert_to_table(dirname):
    dirpath = dirname
    employee_names, store_ids, filepaths, uids = get_names(dirpath)
    all_dict = {"employee_name": [], "store": [], "raw_resume": [], "uid": [],
                "filename": []}
    useless = 0
    useless_lst = []
    for i in range(len(filepaths)):
        try:
            clear_output(wait=True)
            employee_name = employee_names[i]
            store_id = store_ids[i]
            uid = uids[i]
            path = filepaths[i]
            resume = tika_parser(path)
            if "Scanned by CamScanner" in resume:
                useless += 1
                useless_lst.append(employee_name)
            all_dict["employee_name"].append(employee_name)
            all_dict["store"].append(store_id)
            all_dict["raw_resume"].append(resume)
            all_dict["uid"].append(uid)
            all_dict["filename"].append(path)
            print("Current progress:", i, "/", len(filepaths) - 1,
                  np.round((i + 1) / len(filepaths) * 100, 2), "%")
            print("useless (scanned):", str(useless), "/", len(filepaths) - 1,
                  np.round(useless / len(filepaths) * 100, 2), "%")
            print("finished", employee_name)
            print("scanned:", useless_lst)
        except TypeError:
            continue
    return pd.DataFrame(all_dict)


def remove_last_space(string):
    if string != "" and string[-1] == ' ':
        string = string[:-1]
    if string != "" and string[-3:] == 'FRE':
        string = string[:-4]
    return string


def remove_slash_nt(text):
    text = re.sub('[\s]+', ' ',
                  re.sub('[\n\t]', ' ',
                         text))
    return text


def main(file_outpath_all, file_outpath_english, file_path_match):
    dirpath = "../Glentel Inc/HR Analytics - Documents/Capstone Data/Career Builder (Resumes)/Career Builder (Resumes) Before May 18/"
    df = convert_to_table(dirpath)
    df.employee_name = df.employee_name.apply(remove_last_space)

    # Manually modify tables for joining
    # Drop duplicate Momin, Anil
    df = df.drop(index=297)
    df = df.reset_index(drop=True)

    df.iloc[300, 0] = "Muhoza Juste"
    df.iloc[300, 1] = "WW 353"
    df.iloc[300, 3] = "N5M"

    df.iloc[175, 0] = "Harvey - Jones, Derrick"
    df.iloc[175, 1] = "TB 423"
    df.iloc[175, 3] = "MYY"

    df.iloc[205, 0] = "Kaur, Ramandeep*"

    # Akash, Sadana -> Sadana, Akash
    df.iloc[14, 0] = "Sadana, Akash"
    # Bell, Sarah -> Ball, Sarah
    df.iloc[62, 0] = "Ball, Sarah"
    # Mcgregor, Ethan -> McGregor, Ethan
    df.iloc[275, 0] = "McGregor, Ethan"
    # Nasir, Mohammad -> Nasir, Mohammad Humayon
    df.iloc[310, 0] = "Nasir, Mohammad Humayon"
    # Sohi, Jaipal -> Sohi, Jaipal Singh
    df.iloc[422, 0] = "Sohi, Jaipal Singh"
    # Jonaid, Naizi -> Niazi, Jonaid
    df.iloc[539, 0] = "Niazi, Jonaid"

    df.iloc[541, 0] = "Kumta, Ajit"
    df.iloc[541, 1] = "TB 499"
    df.iloc[541, 3] = "JJB"

    df.iloc[71, 0] = "Bois, Dan Kevin"
    df.iloc[71, 1] = "TB 421"
    df.iloc[71, 3] = "MVT"

    # Load matching list
    matchtable = pd.read_excel(file_path_match)

    # Join dataframes
    df2 = pd.merge(df, matchtable, how="left", on="employee_name")

    # Print the person whose employee_code can't be found
    print("{0}'s employee code can't be found".format(
        df2[df2.employee_code.isnull()].employee_name))

    # Manually made changed to some very unstructured resume
    df2.iloc[
        145, 2] = "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n" \
                  "\n\n\n\n\n\n\n\n\n1|Page\n\nJASHANGANDHI\n\n342519 Street " \
                  "" \
                  "" \
                  "NW Edmonton " \
                  "AB.T6T2B5\n\nEmail:jashangandhi07@gmail.com\n\nCELLNUMBER" \
                  "–780-803-8328\n\nOBJECTIVE\n\nSeeking a challenging part " \
                  "time position within professional environment providing " \
                  "an \n\nopportunity for growth and career advancement " \
                  "using my educational and work\n\nexperience\n\nWork " \
                  "Experience\n\nSAMSUNG STORE INDIA PUNJAB\n\nWorked as " \
                  "sales associate in the Samsung store for 2 years(MARCH " \
                  "2014–\n\nAPRIL 2016)\n\nWALMART STORE CANADA " \
                  "ALBERTA\n\nWorked as a sales associate for AVENTURA (" \
                  "DIRECTENERGY) company for 6months\n\nWALMART STORE CANADA " \
                  "" \
                  "" \
                  "ALBERTA\n\nWorked for walmart wireless company OSL as a " \
                  "bike builder\n\nRoles and Responsibilities:\n\n● " \
                  "Responsible for ensuring the delivery of exceptional " \
                  "customer service at\n\nall times.\n\n● Drive sales " \
                  "through an understanding of customer requirements " \
                  "while\n\nproviding an appropriate product solution\n\n● " \
                  "Generate interest and awareness by proactively soliciting " \
                  "" \
                  "" \
                  "existing Walmart\n\ncustomers Accountable for achieving " \
                  "operational excellence through\n\nongoing coaching and " \
                  "development of sales associates\n\n● Responsible for " \
                  "achieving all key performance indicators " \
                  "including\n\nsales, customer experience and operational " \
                  "targets\n\n● Collaborate with leadership to determine " \
                  "ongoing strategic action plans\n\nthat support all key " \
                  "business objectives\n\n● Participate in all required " \
                  "training, with a focus on continued personal\n\nand " \
                  "professional development\n\n\n\n\n\n● Answering questions " \
                  "" \
                  "" \
                  "regarding current customers accounts and\n\n offering " \
                  "assistance\n\n● Keeping track of logs and paperwork " \
                  "including contracts with\n\nsensitive " \
                  "information\n\nEDUCATION\n\nNORQUEST COLLEGE Edmonton," \
                  "Alberta\n\nBusiness Administration–management candidate, " \
                  "SEPT 2017-CURRENT\n\nM.G.NPUBLIC SCHOOL Jalandhar," \
                  "India\n\nAccounting, Marketing, Business, Maths, English, " \
                  "" \
                  "" \
                  "March2016-April2017\n\nREFERNCES\n\nAvailable upon " \
                  "Request\n\n\n"
    # Remove cover letter
    df2.iloc[
        481, 2] = "\n\nJimmy Yang\n\n9 Cashmere Crescent \n\nMarkham, " \
                  "Ontario L3S 4P9\n\nTelephone: (416) 833-7528\n\nEmail: " \
                  "cyang0519@gmail.com\nObjective \n\n\n\nTo broaden my " \
                  "customer service skill by obtaining a job as mobile sales " \
                  "" \
                  "" \
                  "at Costco Wholesale \n\nQualifications and skills " \
                  "\n\n\n\n· Ability to work at a fast paced environment " \
                  "\n\n· Can learn very quickly\n· Friendly, approachable " \
                  "and outgoing\n· Dealing with situation with strong " \
                  "interpersonal, written and verbal communication skills " \
                  "\n\n· Fluent in English, Mandarin and Cantonese " \
                  "\n\nEducation \n\n\n\nCompleted Hospitality & Tourism " \
                  "Administration \n\n2006-2008\nCentennial College " \
                  "\n\n\nExperience \n\n\n\nSupervisor\n\nAxia Café Korean " \
                  "and Japanese Grill Restaurant \n\nSep. 2005-2019\n· " \
                  "Responsibilities include greeting guests, taking orders " \
                  "and serve food\n· Make sure tables are set with linen, " \
                  "dishware and flatware\n\n· Operation with the cash " \
                  "register \n\n· Dealing with customer complaints and " \
                  "assisting their needs\n\nServer\nCentennial College " \
                  "Banquet Hall \n\n\n\n\n2008\n\n· Set banquet rooms and " \
                  "halls as per instructions of the event manager\n\n· " \
                  "Carrying food trays \n· Ensure that food is replenished " \
                  "in a quick manner\nReferences \n\n\n\nReferences will be " \
                  "available upon request / Availability 9am -9pm 7 days\n"
    df2.iloc[
        482, 2] = '\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n' \
                  '\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nKAYASITH \t YANG ' \
                  '\t\n204-960-4939 \t | \t NOYYANG@LIVE.COM ' \
                  '\t\n\t\n\t\n\n\t OBJECTIVE ' \
                  '\t\n\nTo\tobtain\ta\tposition\tutilizing\tprior' \
                  '\texperience\twith\ttelecom,\ttechnology,' \
                  '\t\ncustomer\tservice,\tand\tsales.\t\n\n\t\n\nXPLORE \t ' \
                  'MOBILE \t\n\nSEPTEMBER\t2018\t-\tPRESENT\t\n\nSALES \t ' \
                  'ASSOCIATE \t\n\n- ' \
                  'Assisted\tdivested\tcustomers\twith\ttheir\ttransition' \
                  '\tfrom\tBell\tMTS\tand\t\nVirgin\tMobile\tto\tXplore' \
                  '\tMobile.\t\n\n- ' \
                  'Investigated\tand\tsolved\ttechnical\tissues\tregarding' \
                  '\thandsets\tand\ta\t\nbrand\tnew\tmobile\tnetwork.\t\n\n- ' \
                  '' \
                  '' \
                  'Met\toverall\tsales\tand\tNPS\tscore\ttargets\tset\tby' \
                  '\tmanagement.\t\n- ' \
                  'Helped\tto\ttrain\tand\teducate\tless\texperienced' \
                  '\ttelecom\temployees.\t\n\nCANADA \t REVENUE \t\nAGENCY ' \
                  '\t\n\nMAY\t2017\t–\tPRESENT\t\n\nT 3 \t TRUST \t AND \t ' \
                  'ESTATE \t RETURN \t ASSESING \t OFFICER \t / ' \
                  '\t\nDISABILITY \t TAX \t CREDIT \t PROCESSING \t CLERK ' \
                  '\t\n\n- Assessed\tand\tprocessed\tT3\treturns\tsubmitted' \
                  '\tby\tthe\tgeneral\tpublic.\t\n- T3\taccount\tupkeep,' \
                  '\twhich\tincludes\tdebit\tand\tcredit\tjournal\tentries,' \
                  '\t\n\narrears\tinterest\tcalculations,' \
                  '\tand\tgeneral\taccount\tmaintenance.\t\n- ' \
                  'Reviewed\tthe\twork\tof\tpeers\tand\tnew\ttrainees' \
                  '.\tAssigned\terrors\tto\t\n\nemployees\tif\twork\twas' \
                  '\tnot\tsufficient.\t\n- ' \
                  'Processed\ttaxpayer\trequests\tfor\tthe\tdisability\ttax' \
                  '\tcredit\tto\tpreviously\t\n\nassessed\ttax\tyears.\t\n' \
                  '\nROGERS \t\nCOMMUNICATIONS ' \
                  '\t\n\nMARCH\t2016\t–\tAUGUST\t2017\t\t\n\nSALES \t ' \
                  'ASSOCIATE / SMALL \t BUSINESS \t REPRESENTATIVE \t\n\n- ' \
                  'General\tknowledge\ton\tphones,\ttablets,' \
                  '\tsmart\twatches\tand\t\naccessories.\t\n\n- ' \
                  'Met\tsales\ttargets\tin\ta\tcompetitive\tenvironment.\t\n' \
                  '- Ensured\tstore\tsmall\tbusiness\tactivation\tpercentage' \
                  '\twas\ton\tpar\twith\t\n\ncorporate\texpectations.\t\n- ' \
                  'Provided\tcustomer\tservice\tand\tproblem\tsolve\tto\tfix' \
                  '\tcustomer\t\n\nconcerns\tand/or\tissues.\t\n\nVALUE \t ' \
                  'VILLAGE \t\nTHRIFT \t STORE ' \
                  '\t\n\nJULY\t2015\t–\tMARCH\t2016\t\t\n\nSALES \t ' \
                  'ASSOCIATE \t\n\n- ' \
                  'Ran\tcash\tregisters\tand\tguided\tcustomers\tthrough' \
                  '\tstore\twhen\t\nassistance\twas\tneeded.\t\n\n- ' \
                  'Accepted\tdonations\tand\tunloaded\tcustomer’s\tcars' \
                  '\twhile\tgreeting\t\nthem\ton\tbehalf\tof\tthe\tCanadian' \
                  '\tDiabetes\tAssociation.\t\n\n- ' \
                  'Unloaded\tdonation\ttrucks\tand\tmoved\tboxes\tfrom' \
                  '\tproduction\tareas\t\nto\tstore\tfront\tto\tprepare\tfor' \
                  '\tseasonal\tset\tups.\t\n\nVALLEY \t GARDENS ' \
                  '\t\nCOMMUNITY \t\n\nCENTRE ' \
                  '\t\nNOVEMBER\t2014\t–\t\nNOVEMBER\t2015\t\n\nEVENT \t ' \
                  'CARETAKER\t\n\n- ' \
                  'Set\tup\tevents\tand\tmade\tsure\tthey\tran\thow\tthe' \
                  '\thall\trenter\trequested.\t\n- Cleaned\tevent\thall,' \
                  '\tchange\trooms,\tbathrooms,\tand\tskate\trooms.\t\n- ' \
                  'Assisted\tin\tice\trink\tclean-up.\t\n\n\t\n\n\t\n\n\n\n' \
                  '\t EDUCATION \t\n\nUNIVERSITY \t OF \t\nMANITOBA ' \
                  '\t\n\nJANUARY\t2013\t–\tAPRIL\t2018\t\n\nBACHELOR \t OF ' \
                  '\t COMMERCE \t\n\n- ' \
                  'Left\tin\tApril\tof\t2018\tas\ta\tresult\tof\ta' \
                  '\tpermanent\tjob\toffer\tfrom\tthe\t\nCanada\tRevenue' \
                  '\tAgency.\t\n\t\n\nKILDONAN - EAST \t\nCOLLEGIATE ' \
                  '\t\n\nSEPTEMBER\t2008–\tJUNE\t2012\t\n\nHIGH \t SCHOOL \t ' \
                  '' \
                  '' \
                  'DIPLOMA \t\n\n- ' \
                  'Graduated\tin\tJune\t2012\t\n\t\n\n\t\n\n\t ' \
                  '\t\n\nREFERENCES \t\n\nJULIET \t GAGNON ' \
                  '\t\n\n\t\n\t\n\nCANADA \t REVENUE \t AGENCY \t TEAM \t ' \
                  'LEADER \t\n\n- 204-984-7779\t\n\t\n\nAJ \t DE \t LEON ' \
                  '\t\n\n\t\n\n\t\n\nALIA \t CAMANONG ' \
                  '\t\n\n\t\n\n\t\n\nSUBHDEEP \t SIDHU \t\n\t\n\nROGERS \t ' \
                  'SUPERVISOR \t\n\n- 204-955-8958\t\n\n\t\n\nVALUE \t ' \
                  'VILLAGE \t MANAGER \t\n\n- 204-881-6841\t\n\n\t\n\nVALLEY ' \
                  '' \
                  '' \
                  '\t GARDENS \t CC \t PRESIDENT \t\n\n- ' \
                  '204-979-5415\t\n\n\t\n\n\t\n\n\t\n\t\n\n\t\n\n\n'

    # Add columns of plain text, resume_bline and file_type
    df2["resume_text"] = df2.raw_resume.apply(remove_slash_nt)
    df2["resume_bline"] = df2.raw_resume.apply(lambda x: x.split("\n"))
    df2["file_type"] = df2.filename.apply(
        lambda x: re.match(r".+\.(.*)", x).group(1))

    df3 = df2[['employee_name', 'employee_code', 'store_y', 'raw_resume',
               'resume_text', 'resume_bline', 'language', 'file_type']]

    df3 = df3.rename(columns={"store_y": "store"})

    # Filter out English resumes
    df4 = df3[df3.language == "English"]

    # Save two csv file (all language and english only)
    df3.to_csv(file_outpath_all, index=False)
    df4.to_csv(file_outpath_english, index=False)

    
if __name__ == "__main__":
    main(opt['--file_outpath_all'], opt['--file_outpath_english'], opt['--file_path_match'])
    
print("Finish resume scanner")
