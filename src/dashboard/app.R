"Renders the Dashboard.
Usage:
  app.r [--dashboard_data=<dashboard_data>] [--app_location=<app_location>]

Options:
  --dashboard_data=<dashboard_data> Path (including filename) to dashboard_data [default: ../../../Glentel Inc/HR Analytics - Documents/Capstone Data/ubc_mds_team_share/make_dashboard/].
  --[app_location=<app_location>] Not needed in this function, it is just being passed from previous doc oct call.
" -> doc

library(shiny)
library(dplyr)
library(DT)
library(ggplot2)
library(shinydashboard)
library(ggthemes)
library(janitor)
library(tidyverse)
#library(plotly)
library(patchwork)
library(scales)
library(xlsx)
library(docopt)
#library(testthat)


opt <- docopt(doc)
print(opt)

main <- function(dashboard_data) {
  
  #test_that("test input must include a .csv file location",{
  #  expect_equal(file_ext(dashboard_data), "csv")})
  #install.packages('gifski')

  ##data reading and name cleaning

  input_data <-read.csv(paste(dashboard_data,"data_dashboard_01.csv",sep = ""),stringsAsFactors = FALSE, colClasses=c("hp_class"="double"))
  weights_data <-read.csv(paste(dashboard_data,"data_dashboard_02.csv",sep = ""),stringsAsFactors = FALSE)
  input_data$hp_class <- as.numeric(input_data$hp_class)


  # Sort of a python-Dict, a list for linking the "Nice Name" of a variable with its actual name in the dataset.
  control_legend <- read.xlsx(paste(dashboard_data,"legend_name_control.xlsx",sep = ""),sheetIndex = 1)

  weights_data$Feature_nice_name <- control_legend$nice_name[match(weights_data$Feature,control_legend$name)]

  control_legend <- control_legend %>%
    filter(var_type!="numeric")

  ##Data preparation, just substituting "1"s with "Yes" for comprehension on plots.
  ##we place in ft_list the categorical variables that need to mutated.
  #ft_list = c("food_service_industry_exp","flag_hd_highschool","trilingual_flag","competitor_experience","finance_concentration","sales_customer_base_exp","communication_skills","flag_hd_bachelor_plus")
  ft_list = c(control_legend$name)
  # print(0)
  mutator <- function(x, na.rm = FALSE) (ifelse(x == 1,"Yes","No"))
  input_data<-input_data %>%
    mutate_at(filter(control_legend,var_type=="binary")$name, mutator) %>%
    mutate(cat_tenure = factor(cat_tenure, levels = c("not identified","1-6","7-11",">=12")))
  head(input_data$trilingual_flag)

  ##Small data mutuation for plotting general level plot
  a = aggregate(input_data$hp_class, by=list(Target=input_data$hp_class), FUN=length) %>%
    mutate(Target = as.factor(Target)) %>%
    mutate(Target, Target = ifelse(Target == 1, "High Performer", "Non-High Perfomer")) %>%
    mutate(Percentage=round(x/sum(x),2))


  ######### Helper functions ###########

  ### Function 1 ###

  plot_def<- function(df,feat,name_control){
    control_legend = name_control
    nice_name = control_legend$nice_name[match(feat,control_legend$name)]
    plots_stored <- list()
    feat <- as.name(feat)
    data_plot <-df %>%
      group_by({{feat}}) %>%
      summarise(cnt = n(),perf = mean(hp_class)) %>%
      mutate(perc=round(cnt/sum(cnt),2)) %>%
      mutate(perf=round(perf,2)*100)
    
    plot_a <- ggplot(data_plot,aes(x = ({{feat}}),fill={{feat}}, y = cnt)) +
      geom_bar(stat = "identity") +
      labs(title = "Sample Distribtion",
          x = nice_name,
          y = "Number of Observations")+
      #theme_bw()+
      theme_tufte(base_size = 15)+
      theme(legend.position = "none")+
      geom_text(aes(label = cnt, y = cnt+5.5 ,size=0.0005))+
      geom_text(aes(label = percent(perc), y = cnt/2 ,size=0.0005))
    plot_b <- ggplot(data_plot,aes(x = ({{feat}}),fill=perf, y = perf)) +
      geom_bar(stat = "identity") +
      labs(title = "High Performance Rate ",
          x = nice_name,
          y = "High Performance Rate %",
          fill ="Rate %")+
      #theme_bw() +
      theme_tufte(base_size = 15)+
      scale_fill_gradient(low = "#CCFFCC", high = "#003300",limits = c(0, 100))+
      geom_text(aes(label = percent(round(perf,2)/100), y = perf + 0.002),
                position = position_dodge(0.9),
                vjust = 0
      )
    #plots_stored[[1]] <-plot_a
    #plots_stored[[2]] <-plot_b
    #plots_stored
    #grid.arrange(plot_a, plot_b, ncol=2)
    #plot_a +plot_spacer()+ plot_b
    plot_spacer()+plot_a +plot_spacer()+ plot_b+plot_spacer()+plot_layout(widths = c(0.25,1, 0.25,1,0.25))
    #grid.arrange(plot_a,plot_spacer(),plot_b, ncol=3)
  }

  ### Function 2 ###

  save_multiple_plots<- function(list_with_features){
    saved_plots<- list()
    n<-1
    for (each_feat in list_with_features){
      saved_plots[[n]]<-plot_def(input_data,each_feat,control_legend)
      n = n + 1
    }
    saved_plots
  }

  ## Plotting all feats and saving them in an list object called "all_plots"

  all_plots<-save_multiple_plots(control_legend$name)


  ui <- fluidPage(
    br(),
    titlePanel("Traits of high performers in Glentel"),

    br(),
    br(),
    sidebarLayout(
      sidebarPanel(
        radioButtons("typeInput", "Features",
                    choices = control_legend$nice_name,
                    selected = control_legend$nice_name[[1]]),
      ),
    mainPanel(
      tabsetPanel(
        tabPanel("Overview",
                fluidRow(tags$h2("Overall Target Distribution")),
                fluidRow(column(4,plotOutput(outputId = "plot_G01",width = "100%",height = "300px")),column(8,plotOutput(outputId = "plot_G02",width = "100%",height = "300px"))),
                #fluidRow(tags$h5("The present dashboard shows the behavior of the main applicant trait's correlated to performance"))
                ), 
        tabPanel("Individual Feats",
                fluidRow(tags$h2(textOutput(outputId ="text01"))),
                fluidRow(htmlOutput(outputId ="text02")),
                fluidRow(plotOutput(outputId = "plot_F01",  width = "100%",height = "300px"))
        )
      )
    )
    )
  )


  server <- function(input, output) {
    output$plot_G01 <- renderPlot({
      ggplot(a,aes(x = Target,fill=Target, y = x)) +
        geom_bar(stat = "identity") +
        labs(title = "Sample Distribtion",
            x = "Target Label",
            y = "Count")+
        theme_tufte(base_size = 15)+
        theme(legend.position = "none")+
        geom_text(aes(label = x, y = x + 3))+
        geom_text(aes(label = percent(Percentage), y = x/2))
    })
    output$plot_G02 <- renderPlot({
      weights_data %>%
        arrange(abs(Coefficient)) %>%    # First sort by val. This sort the dataframe but NOT the factor levels
        mutate(Feature_nice_name=factor(Feature_nice_name, levels=Feature_nice_name)) %>%  # This trick update the factor levels
        mutate(Color = ifelse(Coefficient > 0, "red", "green")) %>%
        ggplot(aes(x=Feature_nice_name, y=Coefficient, color = Color)) +
        geom_segment( aes(xend=Feature_nice_name, yend=0)) +
        geom_point( size=4) +
        geom_hline(yintercept = 0,color = "black",linetype="dashed") +
        geom_text(aes(label = round(Coefficient,3), y = Coefficient/2),nudge_x=0.15) +
        coord_flip() +
        theme_tufte(base_size = 15)+
        #theme_bw() +
        xlab("") +
        labs(title = "Traits associated with Performance",
            x = "Features",
            y = "Logistic Regression Coefficient")+
        theme(legend.position = "none")
      
      })
    output$plot_F01 <- renderPlot({
      all_plots[[match(input$typeInput,control_legend$nice_name)]]
    })
    output$text01 <- renderText({
      input$typeInput
    })
    output$text02 <- renderUI({
      HTML(control_legend$description[match(input$typeInput,control_legend$nice_name)])
    })
    output$text03 <- renderUI({
      HTML(" Analyzing We can observe: <br> <ul> <li>32% of high performers identified </li><li>68% of high performers identified</li></ul>")
    })
  }
  shinyApp(ui = ui, server = server)
}

main(opt[["--dashboard_data"]])