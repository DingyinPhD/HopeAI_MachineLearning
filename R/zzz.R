.onLoad <- function(libname, pkgname) {
  required_packages <- c(
    "adabag", "caret", "data.table", "e1071", "glmnet", "iml",
    "kernelshap", "neuralnet", "pROC", "randomForest", "xgboost",
    "dplyr", "doParallel", "foreach", "rlang", "stringr"
  )

  # Install missing packages
  missing_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
  if (length(missing_packages) > 0) {
    install.packages(missing_packages, dependencies = TRUE, lib = .libPaths()[1])
  }

  # Load all required packages without attaching them to the search path
  invisible(lapply(required_packages, function(pkg) {
    suppressPackageStartupMessages(library(pkg, character.only = TRUE))
  }))

}
