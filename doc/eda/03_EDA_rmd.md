03\_EDA
================

### How many ended being terminated, why? Is the termination rate different in non-high vs high perf? Are there high performers who were fired? Why?

### Termination rate

![](03_EDA_rmd_files/figure-gfm/term%20count-1.png)<!-- -->

### Termination tenure

How long are people lasting before termination

    ## Don't know how to automatically pick scale for object of type difftime. Defaulting to continuous.

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](03_EDA_rmd_files/figure-gfm/term%20tenure-1.png)<!-- -->

## Transfers and promotions. (active+leaves) Lets understand how employees move within the company within the performance window.

### Job title change distribution + job type

``` r
dist_jobtitle_changes <- with(active_leave, tapply(job_title, employee_code, FUN = function(x) length(unique(x))))
jobtitle_change <- tibble(employee_code = unique(active_leave$employee_code), num_jobtitle_change = dist_jobtitle_changes)

jobtitle_change_join <- left_join(df_resume, jobtitle_change, by = "employee_code")

ggplot(jobtitle_change_join, aes(x=as.factor(num_jobtitle_change), fill=job_title))+
  geom_bar(stat="count")+
  xlab("Number of Job Changes")+
  theme_minimal()
```

![](03_EDA_rmd_files/figure-gfm/job%20title-1.png)<!-- -->

### contract change (parttime to permanent or viceversa), how many had changes done over the performance period in their contract?

``` r
dist_count_worker_category <- with(active_leave, tapply(worker_category, employee_code, FUN = function(x) length(unique(x))))

count_worker_catagory <- tibble(employee_code = unique(active_leave$employee_code), worker_count = dist_count_worker_category)

sum(count_worker_catagory$worker_count > 1)
```

    ## [1] 12

**only 12 for the whole employment population not very insightful**
