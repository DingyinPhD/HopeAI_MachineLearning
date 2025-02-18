.onLoad <- function(libname, pkgname) {
  required_packages <- c(
    "adabag", "caret", "data.table", "e1071", "glmnet", "iml",
    "kernelshap", "neuralnet", "pROC", "randomForest", "xgboost",
    "dplyr", "doParallel", "foreach", "rlang", "parallel"
  )

  missing_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]

  if (length(missing_packages)) {
    install.packages(missing_packages, dependencies = TRUE, lib = .libPaths()[1])
  }

  # Load packages quietly without attaching them to the search path
  invisible(lapply(required_packages, function(pkg) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop(paste("Package", pkg, "is required but could not be loaded."))
    }
  }))
}
