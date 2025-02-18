.onLoad <- function(libname, pkgname) {
  required_packages <- c(
    "adabag", "caret", "data.table", "e1071", "glmnet", "iml",
    "kernelshap", "neuralnet", "pROC", "randomForest", "xgboost",
    "dplyr", "doParallel", "foreach", "rlang"
  )

  missing_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]

  if (length(missing_packages)) {
    install.packages(missing_packages, dependencies = TRUE, lib = .libPaths()[1])
  }

  # Fix rgl issue (prevent X11 error)
  if ("rgl" %in% installed.packages()[,"Package"]) {
    suppressMessages(rgl::rgl.useNULL(TRUE))
  }

  # Ensure parallel is loaded
  if (!"parallel" %in% loadedNamespaces()) {
    library(parallel)
  }

  invisible(lapply(required_packages, function(pkg) {
    requireNamespace(pkg, quietly = TRUE)
  }))
}
