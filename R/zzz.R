.onLoad <- function(libname, pkgname) {
  required_packages <- c(
    "adabag", "caret", "data.table", "e1071", "glmnet", "iml",
    "kernelshap", "neuralnet", "pROC", "randomForest", "xgboost",
    "dplyr", "doParallel", "foreach", "rlang"
  )

  invisible(lapply(required_packages, function(pkg) {
    suppressPackageStartupMessages(requireNamespace(pkg, quietly = TRUE))
  }))
}
