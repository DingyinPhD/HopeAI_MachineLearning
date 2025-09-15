suppressPackageStartupMessages({
  library(caret)
  library(pROC)
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(parallel)
  library(doParallel)
  library(rlang)
})

# =============== Utilities =====================================================

# pick stratified folds for classification; quantile-stratified for regression
.make_outer_folds <- function(y, k_outer, model_type, seed = 123) {
  set.seed(seed)
  if (model_type == "Classification") {
    if (!is.factor(y)) y <- factor(y)
    createFolds(y, k = k_outer, returnTrain = TRUE)
  } else {
    # quantile binning improves balance for regression folds
    bins <- cut(y, breaks = min(5, length(unique(y))), include.lowest = TRUE)
    createFolds(bins, k = k_outer, returnTrain = TRUE)
  }
}

# build an inner trainControl suited to the task and whether the model can output probabilities
.make_inner_ctrl <- function(k_inner, model_type, wants_probs) {
  if (model_type == "Classification" && wants_probs) {
    trainControl(
      method = "cv",
      number = k_inner,
      classProbs = TRUE,
      summaryFunction = twoClassSummary,
      savePredictions = "final",
      allowParallel = TRUE
    )
  } else if (model_type == "Classification") {
    trainControl(
      method = "cv",
      number = k_inner,
      classProbs = FALSE,
      savePredictions = "final",
      allowParallel = TRUE
    )
  } else {
    trainControl(
      method = "cv",
      number = k_inner,
      savePredictions = "final",
      allowParallel = TRUE
    )
  }
}

# per-model recipes: method, tune grid/length, extra args, prob support, metric
.get_model_spec <- function(model_name, model_type, p) {
  # p = number of predictors
  # returns a list: method, tuneGrid OR tuneLength, extra (list), wants_probs (bool), metric
  model_name <- tolower(model_name)

  if (model_name == "random forest") {
    list(
      method = "rf",
      tuneGrid = data.frame(mtry = unique(pmax(1L, round(c(sqrt(p)/2, sqrt(p), sqrt(p)*2, p/3, p/2))))),
      extra = list(ntree = 500),
      wants_probs = (model_type == "Classification"),
      metric = if (model_type == "Classification") "ROC" else "RMSE"
    )

  } else if (model_name == "elastic net") {
    list(
      method = "glmnet",
      tuneGrid = expand.grid(alpha = seq(0, 1, length.out = 10),
                             lambda = 10^seq(5, -5, length.out = 60)),
      extra = list(family = if (model_type == "Classification") "binomial" else "gaussian"),
      wants_probs = (model_type == "Classification"),
      metric = if (model_type == "Classification") "ROC" else "RMSE"
    )

  } else if (model_name == "svm") {
    # use Radial SVM; supports probabilities when prob.model is TRUE internally (caret handles)
    list(
      method = "svmRadial",
      tuneLength = 8,  # covers C and sigma
      extra = list(),
      wants_probs = (model_type == "Classification"),
      metric = if (model_type == "Classification") "ROC" else "RMSE"
    )

  } else if (model_name == "xgboost") {
    # caret method wrapper
    list(
      method = "xgbTree",
      tuneLength = 20,
      extra = list(
        objective = if (model_type == "Classification") "binary:logistic" else "reg:squarederror",
        verbose = 0
      ),
      wants_probs = (model_type == "Classification"),
      metric = if (model_type == "Classification") "ROC" else "RMSE"
    )

  } else if (model_name == "adaboost") {
    # classification only
    list(
      method = "AdaBoost.M1",
      tuneGrid = expand.grid(
        mfinal = seq(50, 150, by = 25),
        maxdepth = c(1, 2, 3),
        coeflearn = c("Breiman", "Freund", "Zhu")
      ),
      extra = list(),
      wants_probs = TRUE,
      metric = "ROC"
    )

  } else if (model_name == "neural network") {
    list(
      method = "nnet",
      tuneGrid = expand.grid(size = 1:10, decay = c(0.001, 0.01, 0.1)),
      extra = list(linout = (model_type == "Regression"), trace = FALSE, MaxNWts = 250000),
      wants_probs = FALSE, # caret nnet probs are not reliable for twoClassSummary; use Accuracy/RMSE
      metric = if (model_type == "Classification") "Accuracy" else "RMSE"
    )

  } else if (model_name == "knn") {
    list(
      method = "knn",
      tuneGrid = data.frame(k = seq(1, min(50, max(5, floor(p/2))), by = 2)),
      extra = list(),
      wants_probs = FALSE,
      metric = if (model_type == "Classification") "Accuracy" else "RMSE"
    )

  } else if (model_name == "decision tree") {
    list(
      method = "rpart",
      tuneGrid = data.frame(cp = seq(0.001, 0.05, length.out = 10)),
      extra = list(),
      wants_probs = FALSE,
      metric = if (model_type == "Classification") "Accuracy" else "RMSE"
    )

  } else if (model_name == "naïve bayes" || model_name == "naive bayes") {
    list(
      method = "nb",
      tuneGrid = expand.grid(
        usekernel = c(TRUE, FALSE),
        fL = seq(0, 2, by = 0.5),
        adjust = c(0.5, 1, 2)
      ),
      extra = list(),
      wants_probs = TRUE,
      metric = "ROC"
    )

  } else {
    stop("Unsupported model: ", model_name)
  }
}

# safely compute outer metrics
.compute_outer_metrics <- function(model_type, y_te, pred_class, pred_prob = NULL) {
  if (model_type == "Classification") {
    # y_te should be factor with 2 levels; pred_class factor; pred_prob numeric for pos class if available
    cm <- caret::confusionMatrix(pred_class, y_te, positive = levels(y_te)[2])
    out <- list(
      Accuracy = unname(cm$overall["Accuracy"]),
      Kappa    = unname(cm$overall["Kappa"])
    )
    if (!is.null(pred_prob)) {
      roc_obj <- tryCatch(pROC::roc(response = y_te, predictor = pred_prob, quiet = TRUE), error = function(e) NULL)
      out$AUROC <- if (is.null(roc_obj)) NA_real_ else as.numeric(pROC::auc(roc_obj))
    } else {
      out$AUROC <- NA_real_
    }
    out
  } else {
    data.frame(
      RMSE = sqrt(mean((y_te - pred_class)^2)),
      MAE  = mean(abs(y_te - pred_class)),
      R2   = caret::R2(pred = pred_class, obs = y_te)
    ) %>% as.list()
  }
}

# =============== Main: Nested CV Benchmark ====================================

model_benchmark_V6 <- function(
    Features,
    Target,
    Input_Data,
    max_tuning_iteration = 100,        # kept for compatibility; not used in nested CV
    fold = 10,                          # kept for compatibility; used as k_outer if outer_k not supplied
    model = c("Random Forest", "Naïve Bayes", "Elastic Net", "SVM",
              "XGBoost", "AdaBoost", "Neural Network", "KNN", "Decision Tree"),
    model_type = c("Classification", "Regression"),
    dependency_threshold = NULL,        # only used for your binarization step (Classification)
    gene_hits_percentage_cutoff_Lower = 0.2,
    gene_hits_percentage_cutoff_Upper = 0.8,
    XBoost_tuning_grid = "Simple",      # not used here; caret handles xgb tuning via tuneLength
    Finding_Optimal_Threshold = TRUE,   # not used in nested CV summary; we report standard metrics
    Enable_prechecking = FALSE,         # optional precheck; set TRUE if you retain your original gating
    testing_percentage = NA,            # ignored in nested CV
    SHAP = FALSE,                       # SHAP not included here
    outer_k = 5,
    inner_k = 5,
    seed = 123,
    n_cores = NULL
) {

  model_type <- match.arg(model_type)

  # parallel
  if (is.null(n_cores)) {
    n_cores <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK"))
    if (is.na(n_cores)) n_cores <- max(1, parallel::detectCores() - 1)
  }
  cl <- NULL
  if (n_cores > 1) {
    cl <- parallel::makeCluster(n_cores)
    doParallel::registerDoParallel(cl)
    on.exit({
      try(parallel::stopCluster(cl), silent = TRUE)
      doParallel::registerDoSEQ()
    }, add = TRUE)
  } else {
    doParallel::registerDoSEQ()
  }

  df <- Input_Data
  stopifnot(Target %in% colnames(df))

  # Optional precheck (kept minimal; customize as needed)
  if (Enable_prechecking && !is.null(dependency_threshold) && model_type == "Classification") {
    frac_below <- mean(df[[Target]] < dependency_threshold, na.rm = TRUE)
    if (frac_below < gene_hits_percentage_cutoff_Lower || frac_below > gene_hits_percentage_cutoff_Upper) {
      warning(sprintf("Skip: fraction_below=%.3f outside [%.2f, %.2f]",
                      frac_below, gene_hits_percentage_cutoff_Lower, gene_hits_percentage_cutoff_Upper))
      return(tibble())
    }
  }

  # Prepare X, y
  if (model_type == "Classification" && !is.null(dependency_threshold)) {
    df[[Target]] <- ifelse(df[[Target]] <= dependency_threshold, 1L, 0L)
  }
  # remove NAs
  df <- na.omit(df)

  y <- df[[Target]]
  X <- df %>% select(-all_of(Target))
  p <- ncol(X)

  if (model_type == "Classification") {
    y <- factor(y)  # ensure factor with two levels; positive = second level by caret convention
  } else {
    y <- as.numeric(y)
  }

  # Outer folds
  outer_folds <- .make_outer_folds(y, k_outer = outer_k, model_type = model_type, seed = seed)

  # results store
  all_rows <- list()

  # Loop over models
  for (mdl in model) {
    message("=== Model: ", mdl, " ===")
    spec <- .get_model_spec(mdl, model_type, p)
    wants_probs <- isTRUE(spec$wants_probs)
    metric <- spec$metric

    # For each outer fold: inner tuning via caret::train on the outer-train set
    for (i in seq_along(outer_folds)) {
      tr_idx <- outer_folds[[i]]
      te_idx <- setdiff(seq_len(nrow(df)), tr_idx)

      X_tr <- X[tr_idx, , drop = FALSE]
      y_tr <- y[tr_idx]
      X_te <- X[te_idx, , drop = FALSE]
      y_te <- y[te_idx]

      # Inner CV control
      ctrl_inner <- .make_inner_ctrl(k_inner = inner_k, model_type = model_type, wants_probs = wants_probs)

      # Assemble train() args
      train_args <- list(
        x = X_tr,
        y = y_tr,
        method = spec$method,
        trControl = ctrl_inner,
        metric = metric
      )
      if (!is.null(spec$tuneGrid))   train_args$tuneGrid   <- spec$tuneGrid
      if (!is.null(spec$tuneLength)) train_args$tuneLength <- spec$tuneLength
      if (length(spec$extra))        train_args <- c(train_args, spec$extra)

      # Fit inner CV (tuning) on outer-train
      set.seed(seed + i)
      tuned_model <- tryCatch(do.call(caret::train, train_args), error = function(e) e)

      if (inherits(tuned_model, "error")) {
        warning(sprintf("[Outer %d] %s failed: %s", i, mdl, tuned_model$message))
        all_rows[[length(all_rows) + 1]] <- tibble(
          Algorithm = mdl, Fold = i, Tuned_Value = NA_character_,
          Training_Accuracy = NA_real_, Prediction_Accuracy = NA_real_,
          Training_Kappa = NA_real_,     Prediction_Kappa = NA_real_,
          Training_AUROC = NA_real_,     Prediction_AUROC = NA_real_,
          Training_RMSE = NA_real_,      Prediction_RMSE = NA_real_,
          Training_MAE  = NA_real_,      Prediction_MAE  = NA_real_,
          Training_R2   = NA_real_,      Prediction_R2   = NA_real_,
          model_type = model_type
        )
        next
      }

      # Best params readable
      tuned_value <- paste(names(tuned_model$bestTune), tuned_model$bestTune, sep = "=", collapse = ",")

      # Fit final model on outer-train with best params (train already refits by default with bestTune)
      # Evaluate on outer-test
      if (model_type == "Classification") {
        # predicted classes
        pred_class_te <- predict(tuned_model, X_te, type = "raw")
        # probabilities if available
        pred_prob_te <- if (wants_probs) {
          pp <- tryCatch(predict(tuned_model, X_te, type = "prob"), error = function(e) NULL)
          if (!is.null(pp)) {
            # caret uses columns by class labels; take the second level as "positive"
            pos_lab <- levels(y_tr)[2]
            if (pos_lab %in% colnames(pp)) as.numeric(pp[[pos_lab]]) else NULL
          } else NULL
        } else NULL

        # Also compute training metrics for reference
        pred_class_tr <- predict(tuned_model, X_tr, type = "raw")
        pred_prob_tr <- if (wants_probs) {
          pp <- tryCatch(predict(tuned_model, X_tr, type = "prob"), error = function(e) NULL)
          if (!is.null(pp)) {
            pos_lab <- levels(y_tr)[2]
            if (pos_lab %in% colnames(pp)) as.numeric(pp[[pos_lab]]) else NULL
          } else NULL
        } else NULL

        trm <- .compute_outer_metrics("Classification", y_tr, pred_class_tr, pred_prob_tr)
        tem <- .compute_outer_metrics("Classification", y_te, pred_class_te, pred_prob_te)

        all_rows[[length(all_rows) + 1]] <- tibble(
          Algorithm = mdl,
          Fold = i,
          Tuned_Value = tuned_value,
          Training_Accuracy = unname(trm$Accuracy),
          Prediction_Accuracy = unname(tem$Accuracy),
          Training_Kappa = unname(trm$Kappa),
          Prediction_Kappa = unname(tem$Kappa),
          Training_AUROC = unname(trm$AUROC),
          Prediction_AUROC = unname(tem$AUROC),
          Training_RMSE = NA_real_, Prediction_RMSE = NA_real_,
          Training_MAE  = NA_real_, Prediction_MAE  = NA_real_,
          Training_R2   = NA_real_, Prediction_R2   = NA_real_,
          model_type = model_type
        )

      } else {
        # Regression
        pred_tr <- predict(tuned_model, X_tr)
        pred_te <- predict(tuned_model, X_te)

        trm <- .compute_outer_metrics("Regression", y_tr, pred_tr)
        tem <- .compute_outer_metrics("Regression", y_te, pred_te)

        all_rows[[length(all_rows) + 1]] <- tibble(
          Algorithm = mdl,
          Fold = i,
          Tuned_Value = tuned_value,
          Training_Accuracy = NA_real_, Prediction_Accuracy = NA_real_,
          Training_Kappa = NA_real_,     Prediction_Kappa = NA_real_,
          Training_AUROC = NA_real_,     Prediction_AUROC = NA_real_,
          Training_RMSE = unname(trm$RMSE),
          Prediction_RMSE = unname(tem$RMSE),
          Training_MAE  = unname(trm$MAE),
          Prediction_MAE  = unname(tem$MAE),
          Training_R2   = unname(trm$R2),
          Prediction_R2   = unname(tem$R2),
          model_type = model_type
        )
      }
    } # end outer folds
  }   # end models

  final_df <- bind_rows(all_rows) %>%
    arrange(Algorithm, Fold)

  # Also provide per-model summary if helpful
  summary_df <- final_df %>%
    group_by(Algorithm) %>%
    summarise(
      mean_outer_acc  = if (model_type == "Classification") mean(Prediction_Accuracy, na.rm = TRUE) else NA_real_,
      mean_outer_auc  = if (model_type == "Classification") mean(Prediction_AUROC, na.rm = TRUE) else NA_real_,
      mean_outer_rmse = if (model_type == "Regression") mean(Prediction_RMSE, na.rm = TRUE) else NA_real_,
      mean_outer_r2   = if (model_type == "Regression") mean(Prediction_R2,   na.rm = TRUE) else NA_real_,
      .groups = "drop"
    )

  attr(final_df, "summary") <- summary_df
  final_df
}
