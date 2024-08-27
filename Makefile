all: process_resumes auto_features manual_features dataset train_model app

process_resumes:
	python ./src/data_cleaning/resume_scanner.py --file_outpath_all='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_interim/all_language_resumes.csv' --file_outpath_english='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_interim/english_resumes.csv' --file_path_match='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_raw_data/05182020_matching_list_maunually-checked_V1.2.xlsx'
	python ./src/data_cleaning/resume_cleaner.py --file_path='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_interim/english_resumes.csv' --file_outpath='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/english_clean_resumes.csv' --file_cities='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_external/cities.txt' --file_otherwords='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_external/otherwords.txt'
	python ./src/data_cleaning/target_and_splitter.py --file_path_1='../Glentel Inc/HR Analytics - Documents/Capstone Data/V2 data 5.1.2020/' --file_path_2='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/' --clean_path='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/'

auto_features:
	#Train set
	python ./src/feature_extraction/auto_knowledge_skills.py --file_path='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/' --file_outpath='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/' --mode='train'
	python ./src/feature_extraction/auto_comm_lvl.py --file_path_1='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/english_clean_resumes.csv' --file_path_2='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/train_dataset.csv' --file_outpath='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/auto_communication_level.csv'

	#Test set
	python ./src/feature_extraction/auto_knowledge_skills.py --file_path='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/' --file_outpath='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/' --mode='test'
	python ./src/feature_extraction/auto_comm_lvl.py --file_path_1='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/english_clean_resumes.csv' --file_path_2='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/test_dataset.csv' --file_outpath='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/auto_communication_level_test.csv'


manual_features: auto_features
	#Train set
	python ./src/feature_extraction/manual_extract_educ_conc.py --file_path='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_raw_data/manual_extraction_template.xlsx' --file_outpath='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/manual_education_concentration.csv' --mode='train'
	python ./src/feature_extraction/manual_extract_job_title.py --file_path='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_raw_data/manual_extraction_template.xlsx' --file_outpath='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/manual_jobtitle.csv' --mode='train'
	python ./src/feature_extraction/manual_highest_degree.py --file_path='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_raw_data/' --file_outpath='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/' --mode='train'
	python ./src/feature_extraction/manual_work_experience_calculate.py --resume_path='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_raw_data/manual_extraction_template.xlsx' --info_path='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_raw_data/05182020_matching_list_maunually-checked_V1.2.xlsx' --file_outpath='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/' --mode='train'
	python ./src/feature_extraction/manual_competitor_entity_recog.py --resume_path='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_raw_data/manual_extraction_template.xlsx' --file_outpath='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/' --mode='train'
	python ./src/feature_extraction/manual_tenure.py --file_path='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/' --file_outpath='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/' --mode='train'
	python ./src/feature_extraction/manual_extract_sales_cust_exp.py --file_path='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/' --file_outpath='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/' --mode='train'
	# # python ./src/feature_extraction/manual_extract_industry_category.py --file_path='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_external/' --file_path_2='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_raw_data/' --file_outpath='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/'
	python ./src/feature_extraction/manual_additional_feats.py --file_path='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/' --file_outpath='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/' --mode='train'

	#Test set
	python ./src/feature_extraction/manual_extract_educ_conc.py --file_path='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_raw_data/manual_extraction_template_test.xlsx' --file_outpath='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/manual_education_concentration_test.csv' --mode='test'
	python ./src/feature_extraction/manual_extract_job_title.py --file_path='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_raw_data/manual_extraction_template_test.xlsx' --file_outpath='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/manual_jobtitle_test.csv' --mode='test'
	python ./src/feature_extraction/manual_highest_degree.py --file_path='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_raw_data/' --file_outpath='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/' --mode='test'
	python ./src/feature_extraction/manual_work_experience_calculate.py --resume_path='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_raw_data/manual_extraction_template_test.xlsx' --info_path='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_raw_data/05182020_matching_list_maunually-checked_V1.2.xlsx' --file_outpath='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/' --mode='test'
	python ./src/feature_extraction/manual_competitor_entity_recog.py --resume_path='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_raw_data/manual_extraction_template_test.xlsx' --file_outpath='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/' --mode='test'
	python ./src/feature_extraction/manual_tenure.py --file_path='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/' --file_outpath='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/' --mode='test'
	python ./src/feature_extraction/manual_extract_sales_cust_exp.py --file_path='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/' --file_outpath='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/' --mode='test'
	# # python ./src/feature_extraction/manual_extract_industry_category.py --file_path='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_external/' --file_path_2='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_raw_data/' --file_outpath='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/'
	python ./src/feature_extraction/manual_additional_feats.py --file_path='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/' --file_outpath='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/' --mode='test'


dataset:
	#train dataset
	python ./src/data_cleaning/build_dataset.py --file_path_features='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/' --file_path_processed='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/' --file_outpath='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/' --mode='train'
	
	#test dataset
	python ./src/data_cleaning/build_dataset.py --file_path_features='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/' --file_path_processed='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/' --file_outpath='../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/' --mode='test'


train_model:
	python ./src/model_pipeline/ml_pipeline.py --train="../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/" --test="../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/" --output_model="model/" --output_results="model/model_results/" --output_img="img/model_result_img/"
	python ./src/model_pipeline/feature_experiment.py --train="../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/" --test="../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/" --finalist="model/" --output="model/model_results/"

#To Do
generate_report:

app:
	python ./src/dashboard/data_prep_dashboard.py --file_path_1="../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/" --file_outpath="../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_dashboard/"
	Rscript ./src/dashboard/run_dashboard.r --app_location=./src/dashboard/app.R

clean :
	#clean resumes
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_interim/english_resumes.csv"
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_interim/all_language_resumes.csv"
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/english_clean_resumes.csv"
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/train_dataset.csv"
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/test_dataset.csv"
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/consolidated_tabular_data_df.csv"

	#clean manual features train
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/manual_education_concentration.csv"
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/manual_jobtitle.csv"
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/manual_higher_degree.csv"
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/manual_work_exp.csv"
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/manual_competitor_experience.csv"
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/manual_tenure.csv"
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/manual_extract_sales_cust_exp.csv"
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/manual_consolidated_feats.csv"
	#rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/manual_job_category.csv"
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/manual_additional_feats.csv"

	#clean manual features test
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/manual_education_concentration_test.csv"
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/manual_jobtitle_test.csv"
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/manual_higher_degree_test.csv"
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/manual_work_exp_test.csv"
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/manual_competitor_experience_test.csv"
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/manual_tenure_test.csv"
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/manual_extract_sales_cust_exp_test.csv"
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/manual_consolidated_feats_test.csv"
	#rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/manual_job_category.csv"
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/manual_additional_feats_test.csv"
	
	#clean auto features train
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/auto_knowledge_skills.csv"
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/auto_communication_level.csv"
	
	#clean auto features test
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/auto_knowledge_skills_test.csv"
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_features/auto_communication_level_test.csv"
	
	#clean manual dataset
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/manual_clean_training_dataset.csv"

	#clean manual dataset
	rm -f "../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_processed/manual_clean_training_dataset.csv"

	#clean models 
	rm -f "model/result_fitted_finalist.sav"
	rm -f "model/result_unfitted_finalist.sav"

	#clean models results
	rm -f "model/model_results/*.csv"