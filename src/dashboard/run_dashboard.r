"Renders the Dashboard APP on a browser.
Usage:
  run_dashboard.r --app_location=<app_location>

Options:
  --app_location=<app_location> Path that includes the dashboard_app

" -> doc

library(docopt)

opt <- docopt(doc)
main <- function(app_location) {

  shiny::runApp(
  app_location,
  port = 7088, 
  launch.browser = TRUE,
  host = "127.0.0.1")

}
main(opt[["--app_location"]])


