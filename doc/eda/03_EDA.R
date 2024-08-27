library(tidyverse)
library(readxl)
library(janitor)

# load sheets
active_leave <- read_xlsx('../../SharePoint/HR Analytics - Documents/Capstone Data/V2 data 5.1.2020/Active + Leave (Weekly) - 2020_04_28.xlsx')
active_leave <- clean_names(active_leave)

new_hire <- read_xlsx('../../SharePoint/HR Analytics - Documents/Capstone Data/V2 data 5.1.2020/New Hires - 2020.04.28.xlsx')
new_hire <- clean_names(new_hire)

term <- read_xlsx("../../SharePoint/HR Analytics - Documents/Capstone Data/V2 data 5.1.2020/Terminations - 2020.04.28.xlsx")
term <- clean_names(term)

#grab employee code
df_resume <- read_csv('../../SharePoint/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/02_eda_data/output/train_dataset.csv')
df_resume <- clean_names(df_resume)
employee_code_t <- df_resume$employee_code

term_join <- left_join(df_resume, term, by="employee_code")

# How many ended being terminated, why? Is the termination rate different in non-high vs high perf? Are there high performers who were fired? Why? - Thomas

## counts of employees termination filter by high and low performers after
term_counts <- term_join %>% 
  select(employee_code, termination_type.y ) %>%
  distinct(employee_code, .keep_all = TRUE) %>% 
  group_by(termination_type.y) %>% 
  count()

term_counts[is.na(term_counts)] = "Active"
  
ggplot(term_counts, aes(x=termination_type.y, y=n))+
    geom_bar(position="dodge", stat="identity")+
    theme_minimal()

# How long did people last before they quit


term_diff_day <- term_join %>% 
  filter(termination_type.x == "Resignation") %>%
  select(max_hire_date, termination_date.x) %>% 
  mutate(tenure = termination_date.x - max_hire_date)


ggplot(term_diff_day, aes(x=tenure))+
  geom_histogram()+
  geom_vline(xintercept=90, color="red")


## termination reasons, filter train set
# term_reason_count <- term %>% 
#   select(employee_code, termination_type) %>% 
#   filter(employee_code %in% employee_code_t) %>% 
#   distinct(employee_code, .keep_all = TRUE) %>%
#   group_by(termination_type) %>% 
#   count() %>%
#   drop_na()
# 
# ggplot(term_reason_count, aes(x=termination_type, y=n))+
#   geom_bar(position="dodge", stat="identity")+
#   coord_flip()+
#   theme_minimal()


# transfers and promotions. (active+leaves) Lets understand how employees move within the company within the performance window. - Thomas

dist_jobtitle_changes <- with(active_leave, tapply(job_title, employee_code, FUN = function(x) length(unique(x))))
jobtitle_change <- tibble(employee_code = unique(active_leave$employee_code), num_jobtitle_change = dist_jobtitle_changes)

jobtitle_change_join <- left_join(df_resume, jobtitle_change, by = "employee_code")

ggplot(jobtitle_change_join, aes(x=as.factor(num_jobtitle_change), fill=job_title))+
  geom_bar(stat="count")+
  xlab("Number of Job Changes")+
  theme_minimal()



# contract change (parttime to permanent or viceversa), how many had changes done over the performance period in their contract? - Thomas


dist_count_worker_category <- with(active_leave, tapply(worker_category, employee_code, FUN = function(x) length(unique(x))))

count_worker_catagory <- tibble(employee_code = unique(active_leave$employee_code), worker_count = dist_count_worker_category)

sum(count_worker_catagory$worker_count > 1)
#with only 12 employees swaping workstatus not much can be extracted
