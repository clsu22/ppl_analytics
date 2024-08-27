## Feature Structure 


### Feature directory:

- Feature: features' name
- Feature file: file of feature location 
- Parent Script: script that generates the feature
- Final Model: feature included in final model



| Features                                      | Feature file                       | Parent Script                     | Final Model |
|-----------------------------------------------|------------------------------------|-----------------------------------|-------------|
| competitor_experience                         | manual_competitor_experience.csv   | manual_competitor_entity_recog.py | Yes         |
| finance_concentration                         | manual_education_concentration.csv | manual_extract_educ_conc.py       | Yes         |
| cashier_jobtitle                              | manual_jobtitle.csv                | manual_extract_job_title.py       | Yes         |
| fitness_sports_jobtitle                       | manual_jobtitle.csv                | manual_extract_job_title.py       | Yes         |
| flag_hd_highschool                            | manual_higher_degree.csv           | manual_higher_degree.py           | Yes         |
| trilingual_flag                               | auto_knowledge_skills.csv          | auto_knowledge_skills.py          | Yes         |
| sales_customer_base_exp                       | auto_knowledge_skills.csv          | auto_knowledge_skills.py          | Yes         |
| communication_skills                          | auto_knowledge_skills.csv          | auto_knowledge_skills.py          | Yes         |
| shortest_tenure                               | manual_tenure.csv                  | manual_tunure.py                  | Yes         |
| rehired_                                      | test_dataset.csv/train_dataset.csv | target_and_splitter.py            | No          |
| referral_flag                                 | test_dataset.csv/train_dataset.csv | target_and_splitter.py            | No          |
| job_hopper                                    | test_dataset.csv/train_dataset.csv | target_and_splitter.py            | No          |
| Freedom_competitor_exp                        | manual_competitor_experience.csv   | manual_competitor_entity_recog.py | No          |
| Koodo_competitor_exp                          | manual_competitor_experience.csv   | manual_competitor_entity_recog.py | No          |
| Shaw_competitor_exp                           | manual_competitor_experience.csv   | manual_competitor_entity_recog.py | No          |
| Telus_competitor_exp                          | manual_competitor_experience.csv   | manual_competitor_entity_recog.py | No          |
| Bell_competitor_exp                           | manual_competitor_experience.csv   | manual_competitor_entity_recog.py | No          |
| Rogers_competitor_exp                         | manual_competitor_experience.csv   | manual_competitor_entity_recog.py | No          |
| The_Mobile_Shop_competitor_exp                | manual_competitor_experience.csv   | manual_competitor_entity_recog.py | No          |
| Best_Buy_competitor_exp                       | manual_competitor_experience.csv   | manual_competitor_entity_recog.py | No          |
| Videotron_competitor_exp                      | manual_competitor_experience.csv   | manual_competitor_entity_recog.py | No          |
| Wow_Mobile_competitor_exp                     | manual_competitor_experience.csv   | manual_competitor_entity_recog.py | No          |
| The_Source_competitor_exp                     | manual_competitor_experience.csv   | manual_competitor_entity_recog.py | No          |
| Walmart_competitor_exp                        | manual_competitor_experience.csv   | manual_competitor_entity_recog.py | No          |
| Virgin_Mobile_competitor_exp                  | manual_competitor_experience.csv   | manual_competitor_entity_recog.py | No          |
| Osl_competitor_exp                            | manual_competitor_experience.csv   | manual_competitor_entity_recog.py | No          |
| accounting_concentration                      | manual_education_concentration.csv | manual_extract_educ_conc.py       | No          |
| arts_concentration                            | manual_education_concentration.csv | manual_extract_educ_conc.py       | No          |
| business_concentration                        | manual_education_concentration.csv | manual_extract_educ_conc.py       | No          |
| computer_systems_concentration                | manual_education_concentration.csv | manual_extract_educ_conc.py       | No          |
| engineering_concentration                     | manual_education_concentration.csv | manual_extract_educ_conc.py       | No          |
| general_concentration                         | manual_education_concentration.csv | manual_extract_educ_conc.py       | No          |
| human_resource_concentration                  | manual_education_concentration.csv | manual_extract_educ_conc.py       | No          |
| interactive_arts_and_technology_concentration | manual_education_concentration.csv | manual_extract_educ_conc.py       | No          |
| marketing_concentration                       | manual_education_concentration.csv | manual_extract_educ_conc.py       | No          |
| not_specified_concentration                   | manual_education_concentration.csv | manual_extract_educ_conc.py       | No          |
| other_concentration                           | manual_education_concentration.csv | manual_extract_educ_conc.py       | No          |
| administrative_jobtitle                       | manual_jobtitle.csv                | manual_extract_job_title.py       | No          |
| assistant_manager_jobtitle                    | manual_jobtitle.csv                | manual_extract_job_title.py       | No          |
| blue_collar_jobtitle                          | manual_jobtitle.csv                | manual_extract_job_title.py       | No          |
| cook_jobtitle                                 | manual_jobtitle.csv                | manual_extract_job_title.py       | No          |
| customer_service_representative_jobtitle      | manual_jobtitle.csv                | manual_extract_job_title.py       | No          |
| driver_jobtitle                               | manual_jobtitle.csv                | manual_extract_job_title.py       | No          |
| education_jobtitle                            | manual_jobtitle.csv                | manual_extract_job_title.py       | No          |
| financial_services_jobtitle                   | manual_jobtitle.csv                | manual_extract_job_title.py       | No          |
| manager_jobtitle                              | manual_jobtitle.csv                | manual_extract_job_title.py       | No          |
| no_work_title                                 | manual_jobtitle.csv                | manual_extract_job_title.py       | No          |
| other_jobtitle                                | manual_jobtitle.csv                | manual_extract_job_title.py       | No          |
| sales_associate_jobtitle                      | manual_jobtitle.csv                | manual_extract_job_title.py       | No          |
| technicians_jobtitle                          | manual_jobtitle.csv                | manual_extract_job_title.py       | No          |
| telemarketers_jobtitle                        | manual_jobtitle.csv                | manual_extract_job_title.py       | No          |
| highest_degree                                | manual_higher_degree.csv           | manual_higher_degree.py           | No          |
| background_highest_degree                     | manual_higher_degree.csv           | manual_higher_degree.py           | No          |
| country_highest_degree                        | manual_higher_degree.csv           | manual_higher_degree.py           | No          |
| flag_hd_bachelor_plus                         | manual_higher_degree.csv           | manual_higher_degree.py           | No          |
| business_flag                                 | manual_higher_degree.csv           | manual_higher_degree.py           | No          |
| sales_exp_months                              | manual_sales_custom_exp.csv        | manual_extract_sales_cust_exp.py  | No          |
| customer_serv_exp_months                      | manual_sales_custom_exp.csv        | manual_extract_sales_cust_exp.py  | No          |
| leader_ship_exp_months                        | manual_sales_custom_exp.csv        | manual_extract_sales_cust_exp.py  | No          |
| raw_dale_chall_readability                    | auto_communication.csv             | auto_comm_lvl.py                  | No          |
| raw_Flesch-Kincaid_readability                | auto_communication.csv             | auto_comm_lvl.py                  | No          |
| raw_Gunning_FOG_readability                   | auto_communication.csv             | auto_comm_lvl.py                  | No          |
| raw_automate_readability                      | auto_communication.csv             | auto_comm_lvl.py                  | No          |
| clean_Flesch-Kincaid_readability              | auto_communication.csv             | auto_comm_lvl.py                  | No          |
| clean_Gunning_FOG_readability                 | auto_communication.csv             | auto_comm_lvl.py                  | No          |
| clean_automate_readability                    | auto_communication.csv             | auto_comm_lvl.py                  | No          |
| clean_dale_chall_readability                  | auto_communication.csv             | auto_comm_lvl.py                  | No          |
| telco_electro_jobs                            | manual_additional_feats.csv        | manual_additional_feats.py        | No          |
| telco_electro_recency                         | manual_additional_feats.csv        | manual_additional_feats.py        | No          |
| recency_type_telco_electro_exp                | manual_additional_feats.csv        | manual_additional_feats.py        | No          |
| telco_electro_perc_group                      | manual_additional_feats.csv        | manual_additional_feats.py        | No          |
| read_score_categorical                        | manual_additional_feats.csv        | manual_additional_feats.py        | No          |
| no_lang_spoken                                | auto_knowledge_skills.csv          | auto_knowledge_skills.py          | No          |
| goal_record                                   | auto_knowledge_skills.csv          | auto_knowledge_skills.py          | No          |
| volunteer_exp                                 | auto_knowledge_skills.csv          | auto_knowledge_skills.py          | No          |
| problem_solver                                | auto_knowledge_skills.csv          | auto_knowledge_skills.py          | No          |
| sports_mention                                | auto_knowledge_skills.csv          | auto_knowledge_skills.py          | No          |
| team_player                                   | auto_knowledge_skills.csv          | auto_knowledge_skills.py          | No          |
| leadership_mention                            | auto_knowledge_skills.csv          | auto_knowledge_skills.py          | No          |
| total_experience_months                       | manual_tenure.csv                  | manual_tunure.py                  | No          |
| longest_tenure                                | manual_tenure.csv                  | manual_tunure.py                  | No          |
| average_tenure_per_job                        | manual_tenure.csv                  | manual_tunure.py                  | No          |
| no_jobs                                       | manual_tenure.csv                  | manual_tunure.py                  | No          |
| no_job_categorical                            | manual_tenure.csv                  | manual_tunure.py                  | No          |
| clean_text                                    | test_dataset.csv/train_dataset.csv | target_and_splitter.py            | No          |
| hp_class                                      | test_dataset.csv/train_dataset.csv | target_and_splitter.py            | No          |
| Clothing_and_Footwear_industry_exp            | manual_job_category.csv            | manual_extract_industry.py        | No          |
| Consumer_electronics_industry_exp             | manual_job_category.csv            | manual_extract_industry.py        | No          |
| Food_Service_industry_exp                     | manual_job_category.csv            | manual_extract_industry.py        | No          |
| Food-Convenience-Pharmacy_industry_exp        | manual_job_category.csv            | manual_extract_industry.py        | No          |
| Other_industry_exp                            | manual_job_category.csv            | manual_extract_industry.py        | No          |
| Sport_Travel_Enterntain_Hotel_industry_exp    | manual_job_category.csv            | manual_extract_industry.py        | No          |
| Telecommunications_industry_exp               | manual_job_category.csv            | manual_extract_industry.py        | No          |
| unknown_industry_exp                          | manual_job_category.csv            | manual_extract_industry.py        | No          |