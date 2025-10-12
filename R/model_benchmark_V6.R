suppressPackageStartupMessages({
  library(caret)
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(readr)
  library(pROC)
  library(kernelshap)
  library(shapviz)
})

# ------------------ helpers ------------------
.supports_prob <- function(method) {
  info <- getModelInfo(method, regex = FALSE)[[1]]
  !is.null(info$prob)
}

.make_outer_folds <- function(y, k_outer, seed = 123) {
  set.seed(seed)
  if (is.factor(y)) {
    createFolds(y, k = k_outer, returnTrain = TRUE)
  } else {
    # stratify continuous y by binning so folds cover its range
    bins <- cut(y,
                breaks = min(5, max(2, length(unique(y)))),
                include.lowest = TRUE, ordered_result = TRUE)
    createFolds(bins, k = k_outer, returnTrain = TRUE)
  }
}

.compute_metrics <- function(task, y_true, y_pred_label, y_pred_prob = NULL) {
  if (task == "Classification") {
    cm <- caret::confusionMatrix(y_pred_label, y_true, positive = levels(y_true)[2])
    acc <- unname(cm$overall["Accuracy"]); kappa <- unname(cm$overall["Kappa"]); auroc <- NA_real_
    if (!is.null(y_pred_prob)) {
      roc_obj <- try(pROC::roc(response = y_true, predictor = y_pred_prob, quiet = TRUE), silent = TRUE)
      if (!inherits(roc_obj, "try-error")) auroc <- as.numeric(pROC::auc(roc_obj))
    }
    list(Accuracy = acc, Kappa = kappa, AUROC = auroc)
  } else {
    rmse <- sqrt(mean((y_true - y_pred_label)^2))
    mae  <- mean(abs(y_true - y_pred_label))
    r2   <- caret::R2(pred = y_pred_label, obs = y_true)
    list(RMSE = rmse, MAE = mae, R2 = r2)
  }
}

.inner_fold_validation_metrics <- function(tuned_model, task = c("Classification", "Regression")) {
  task <- match.arg(task)
  res <- tuned_model$results
  if (is.null(res) || !nrow(res)) return(list())
  idx <- rep(TRUE, nrow(res))
  for (nm in names(tuned_model$bestTune)) if (nm %in% names(res)) idx <- idx & (res[[nm]] == tuned_model$bestTune[[nm]])
  res_best <- if (any(idx)) res[idx, , drop = FALSE] else res[1, , drop = FALSE]
  col_mean_if <- function(df, col) if (col %in% names(df)) mean(df[[col]], na.rm = TRUE) else NA_real_
  if (task == "Classification") {
    out <- list(
      Validation_ROC      = col_mean_if(res_best, "ROC"),
      Validation_Accuracy = col_mean_if(res_best, "Accuracy"),
      Validation_Kappa    = col_mean_if(res_best, "Kappa")
    )
  } else {
    out <- list(
      Validation_RMSE = col_mean_if(res_best, "RMSE"),
      Validation_Rsq  = col_mean_if(res_best, "Rsquared"),
      Validation_MAE  = col_mean_if(res_best, "MAE")
    )
  }
  for (nm in names(tuned_model$bestTune)) out[[paste0("Best_", nm)]] <- tuned_model$bestTune[[nm]]
  out
}

# --------------- main function ---------------
model_benchmark_V6 <- function(
    data,
    target,   # either a formula (e.g., y ~ (.)^2) or a single string "y"
    models = c("rf", "glmnet", "xgbTree", "svmRadial", "svmLinear", "svmPoly", "rpart", "nb", "knn"),
    k_outer = 5,
    k_inner = 5,
    seed = 123,
    shap = FALSE,
    shap_bg_max = 200,
    shap_pred_max = Inf,
    outdir = ".",
    save_plots = FALSE,
    model_grids = list()
) {
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

  # ---- handle target as formula or string ----
  if (inherits(target, "formula")) {
    target_var <- all.vars(target)[1]
    form <- target
  } else if (is.character(target) && length(target) == 1) {
    target_var <- target
    form <- as.formula(paste(target_var, "~ ."))
  } else {
    stop("`target` must be a formula (e.g., y ~ (.)^2) or a single character column name.")
  }
  if (!(target_var %in% colnames(data))) stop("Target variable not found in `data`: ", target_var)

  # ---- task & splits ----
  task <- if (is.factor(data[[target_var]])) "Classification" else "Regression"
  X_all <- data %>% dplyr::select(-all_of(target_var))
  y_all <- data[[target_var]]
  outer_folds <- .make_outer_folds(y_all, k_outer = k_outer, seed = seed)

  # ---- inner CV controls ----
  ctrl_class <- trainControl(
    method = "cv", number = k_inner,
    classProbs = TRUE, summaryFunction = twoClassSummary,
    savePredictions = "final", allowParallel = TRUE
  )
  ctrl_reg <- trainControl(
    method = "cv", number = k_inner,
    savePredictions = "final", allowParallel = TRUE
  )

  # ---- default tuning grids ----
  p <- ncol(X_all)
  default_grids <- list(
    rf = data.frame(mtry = unique(pmax(1L, round(c(sqrt(p)/2, sqrt(p), sqrt(p)*2, p/3, p/2))))),
    glmnet = expand.grid(alpha = seq(0, 1, length.out = 10),
                         lambda = 10^seq(5, -5, length.out = 60)),
    xgbTree = expand.grid(
      nrounds = c(100, 300),
      max_depth = c(3, 6, 10),
      eta = c(0.05, 0.1, 0.3),
      gamma = c(0, 0.5, 1.0),
      colsample_bytree = c(0.6, 0.8, 1.0),
      min_child_weight = c(1, 3, 5),
      subsample = c(0.6, 0.8, 1.0)
    ),
    svmRadial = expand.grid(C = 10^seq(-2, 2, length.out = 7),
                            sigma = 10^seq(-3, 1, length.out = 7)),
    svmLinear = data.frame(C = 10^seq(-3, 2, length.out = 8)),
    svmPoly   = expand.grid(degree = c(2,3,4),
                            scale = 10^seq(-3, 1, length.out = 5),
                            C = 10^seq(-2, 2, length.out = 7)),
    rpart = data.frame(cp = seq(0.001, 0.05, length.out = 10)),
    nb = expand.grid(usekernel = c(TRUE, FALSE),
                     fL = seq(0, 2, by = 0.5),
                     adjust = c(0.5, 1, 2)),
    knn = data.frame(k = seq(1, min(50, max(5, floor(p/2))), by = 2)),
    nnet = expand.grid(size = 1:10, decay = c(0.001, 0.01, 0.1)),
    `AdaBoost.M1` = expand.grid(mfinal = seq(50, 150, by = 25),
                                maxdepth = c(1,2,3),
                                coeflearn = c("Breiman","Freund","Zhu"))
  )
  for (nm in names(model_grids)) default_grids[[nm]] <- model_grids[[nm]]

  # ---- collectors ----
  perf_rows <- list(); imp_rows <- list(); shap_rows <- list()

  # ---- outer CV loop ----
  for (i in seq_along(outer_folds)) {
    message(sprintf("=== Outer fold %d / %d ===", i, length(outer_folds)))
    tr_idx <- outer_folds[[i]]
    te_idx <- setdiff(seq_len(nrow(data)), tr_idx)

    X_tr <- X_all[tr_idx, , drop = FALSE]; y_tr <- y_all[tr_idx]
    X_te <- X_all[te_idx, , drop = FALSE]; y_te <- y_all[te_idx]

    if (task == "Classification") {
      y_tr <- factor(y_tr)
      y_te <- factor(y_te, levels = levels(y_tr))
    }

    for (method in models) {
      message(sprintf(" -> %s", method))
      metric  <- if (task == "Classification") { if (.supports_prob(method)) "ROC" else "Accuracy" } else "RMSE"
      tr_ctrl <- if (task == "Classification") ctrl_class else ctrl_reg
      tg      <- default_grids[[method]]

      # optional per-method extras
      extra <- list()
      if (method == "xgbTree") {
        extra$verbose   <- 0
        extra$objective <- if (task == "Classification") "binary:logistic" else "reg:squarederror"
      }
      if (method == "nnet") { extra$linout <- (task == "Regression"); extra$trace <- FALSE; extra$MaxNWts <- 250000 }
      if (method == "rf")   { extra$ntree <- 500; extra$nodesize <- if (task == "Classification") 1 else 5 }

      # training set as a df that matches the formula
      train_df <- data.frame(y = y_tr, X_tr)
      names(train_df)[1] <- target_var

      tuned <- try(
        do.call(caret::train, c(
          list(form = form, data = train_df, method = method, trControl = tr_ctrl, metric = metric),
          if (!is.null(tg)) list(tuneGrid = tg) else list(tuneLength = 5),
          extra
        )),
        silent = TRUE
      )

      if (inherits(tuned, "try-error")) {
        warning(sprintf("[Fold %d] %s failed: %s", i, method, as.character(tuned)))
        next
      }

      # save model
      message("Saving model for target_var = ", target_var,
              " | Fold = ", i, " | Algorithm = ", method,
              " → ", file.path(outdir, sprintf("%s_%s_Fold%d.rds", target_var, method, i)))

      saveRDS(tuned, file = file.path(outdir, sprintf("%s_%s_Fold%d.rds", target_var, method, i)))

      message("Class of tuned object: ", class(tuned)[1])


      # inner-CV metrics
      inner_metrics <- .inner_fold_validation_metrics(
        tuned_model = tuned,
        task = if (is.factor(y_tr)) "Classification" else "Regression"
      )

      # training metrics
      if (task == "Classification") {
        pred_tr_lab  <- predict(tuned, newdata = as.data.frame(X_tr), type = "raw")
        pred_tr_prob <- NULL
        if (.supports_prob(method)) {
          prob_tr <- try(predict(tuned, newdata = as.data.frame(X_tr), type = "prob"), silent = TRUE)
          if (!inherits(prob_tr, "try-error")) {
            pos <- levels(y_tr)[2]
            if (!is.na(pos) && pos %in% colnames(prob_tr)) pred_tr_prob <- as.numeric(prob_tr[[pos]])
          }
        }
        m_tr <- .compute_metrics(task, y_tr, pred_tr_lab, pred_tr_prob)
      } else {
        pred_tr <- predict(tuned, newdata = as.data.frame(X_tr))
        m_tr <- .compute_metrics(task, y_tr, pred_tr)
      }

      # test metrics
      if (task == "Classification") {
        pred_te_lab  <- predict(tuned, newdata = as.data.frame(X_te), type = "raw")
        pred_te_prob <- NULL
        if (.supports_prob(method)) {
          prob_te <- try(predict(tuned, newdata = as.data.frame(X_te), type = "prob"), silent = TRUE)
          if (!inherits(prob_te, "try-error")) {
            pos <- levels(y_tr)[2]
            if (!is.na(pos) && pos %in% colnames(prob_te)) pred_te_prob <- as.numeric(prob_te[[pos]])
          }
        }
        m_te <- .compute_metrics(task, y_te, pred_te_lab, pred_te_prob)

        perf_rows[[length(perf_rows) + 1]] <- tibble(
          Fold = i, Algorithm = method,
          Tuned_Parameters = paste(names(tuned$bestTune), tuned$bestTune[1, ], sep = "=", collapse = ";"),
          Training_Accuracy = m_tr$Accuracy,
          Training_Kappa    = m_tr$Kappa,
          Training_AUROC    = m_tr$AUROC,
          Test_Accuracy = m_te$Accuracy,
          Test_Kappa    = m_te$Kappa,
          Test_AUROC    = m_te$AUROC,
          Validation_Accuracy = inner_metrics$Validation_Accuracy,
          Validation_Kappa    = inner_metrics$Validation_Kappa,
          Validation_AUROC    = inner_metrics$Validation_ROC
        )
      } else {
        pred_te <- predict(tuned, newdata = as.data.frame(X_te))
        m_te <- .compute_metrics(task, y_te, pred_te)

        perf_rows[[length(perf_rows) + 1]] <- tibble(
          Fold = i, Algorithm = method,
          Tuned_Parameters = paste(names(tuned$bestTune), tuned$bestTune[1, ], sep = "=", collapse = ";"),
          Training_RMSE = m_tr$RMSE,
          Training_MAE  = m_tr$MAE,
          Training_R2   = m_tr$R2,
          Test_RMSE = m_te$RMSE,
          Test_MAE  = m_te$MAE,
          Test_R2   = m_te$R2,
          Validation_RMSE = inner_metrics$Validation_RMSE,
          Validation_MAE  = inner_metrics$Validation_MAE,
          Validation_R2   = inner_metrics$Validation_Rsq
        )
      }

      # varImp
      vi <- try(varImp(tuned)$importance, silent = TRUE)
      if (!inherits(vi, "try-error")) {
        vi_df <- vi %>% tibble::rownames_to_column("Feature") %>%
          mutate(Algorithm = method, Fold = i)
        imp_rows[[length(imp_rows) + 1]] <- vi_df
      }

      # SHAP (optional) — kernel SHAP around caret model
      # ---- SHAP (works with formulas & factors) ----
      if (shap) {
        bg_n  <- min(shap_bg_max, nrow(X_tr))
        bg_ix <- sample.int(nrow(X_tr), size = bg_n, replace = FALSE)

        # keep as data.frames (preserve factors)
        bg_df        <- as.data.frame(X_tr[bg_ix, , drop = FALSE])
        X_explain_df <- as.data.frame(
          if (is.finite(shap_pred_max)) X_te[seq_len(min(nrow(X_te), shap_pred_max)), , drop = FALSE] else X_te
        )

        if (method == "glmnet") {
          # Use the same formula to rebuild the design matrix for new data,
          # then call the underlying glmnet model directly.
          pred_fun <- function(object, newdata) {
            mm <- model.matrix(form, data = newdata)[, -1, drop = FALSE]  # drop intercept
            as.numeric(predict(object$finalModel, newx = mm, s = object$bestTune$lambda))
          }
        } else if (task == "Classification" && .supports_prob(method)) {
          pos <- levels(y_tr)[2]
          pred_fun <- function(object, newdata) {
            probs <- predict(object, newdata = newdata, type = "prob")
            as.numeric(probs[[pos]])
          }
        } else {
          pred_fun <- function(object, newdata) {
            as.numeric(predict(object, newdata = newdata))
          }
        }

        ks <- kernelshap(
          tuned,
          X    = X_explain_df,   # keep as data.frame
          bg_X = bg_df,          # keep as data.frame
          pred_fun = pred_fun
        )

        saveRDS(ks, file = file.path(outdir, sprintf("%s_%s_Fold%d.kernelshap.rds", target_var, method, i)))

        sv <- shapviz::shapviz(ks, X = X_explain_df, X_pred = as.matrix(X_explain_df))
        imp_tbl <- shapviz::sv_importance(sv, kind = "no", show_numbers = TRUE) %>%
          as.data.frame() %>% tibble::rownames_to_column("Feature") %>%
          dplyr::mutate(Algorithm = method, Fold = i)
        colnames(imp_tbl)[2] <- "MeanAbsSHAP"
        shap_rows[[length(shap_rows) + 1]] <- imp_tbl

        if (save_plots) {
          pdf(file.path(outdir, sprintf("Fold%d_%s_SHAP_importance.pdf", i, method)))
          shapviz::sv_importance(sv, kind = "both", show_numbers = TRUE, max_display = 20)
          dev.off()
        }
      }
    }
  }

  # -------- aggregate & write outputs --------
  perf_df <- dplyr::bind_rows(perf_rows)
  imp_df  <- dplyr::bind_rows(imp_rows)

  # performance summary
  if (nrow(perf_df)) {
    if ("Test_AUROC" %in% names(perf_df)) {
      summary_df <- perf_df %>%
        group_by(Algorithm) %>%
        summarise(
          mean_Train_Accuracy = mean(Training_Accuracy, na.rm = TRUE),
          sd_Train_Accuracy   = sd(Training_Accuracy, na.rm = TRUE),
          mean_Test_Accuracy  = mean(Test_Accuracy, na.rm = TRUE),
          sd_Test_Accuracy    = sd(Test_Accuracy, na.rm = TRUE),
          mean_Train_Kappa    = mean(Training_Kappa, na.rm = TRUE),
          sd_Train_Kappa      = sd(Training_Kappa, na.rm = TRUE),
          mean_Test_Kappa     = mean(Test_Kappa, na.rm = TRUE),
          sd_Test_Kappa       = sd(Test_Kappa, na.rm = TRUE),
          mean_Train_AUROC    = mean(Training_AUROC, na.rm = TRUE),
          sd_Train_AUROC      = sd(Training_AUROC, na.rm = TRUE),
          mean_Test_AUROC     = mean(Test_AUROC, na.rm = TRUE),
          sd_Test_AUROC       = sd(Test_AUROC, na.rm = TRUE),
          mean_Validation_AUROC  = mean(Validation_AUROC, na.rm = TRUE),
          sd_Validation_AUROC    = sd(Validation_AUROC, na.rm = TRUE),
          mean_Validation_Kappa  = mean(Validation_Kappa, na.rm = TRUE),
          sd_Validation_Kappa    = sd(Validation_Kappa, na.rm = TRUE),
          mean_Validation_Accuracy  = mean(Validation_Accuracy, na.rm = TRUE),
          sd_Validation_Accuracy    = sd(Validation_Accuracy, na.rm = TRUE),
          .groups = "drop"
        ) %>% arrange(desc(mean_Test_AUROC))
    } else {
      summary_df <- perf_df %>%
        group_by(Algorithm) %>%
        summarise(
          mean_Train_RMSE = mean(Training_RMSE, na.rm = TRUE),
          sd_Train_RMSE   = sd(Training_RMSE, na.rm = TRUE),
          mean_Test_RMSE  = mean(Test_RMSE, na.rm = TRUE),
          sd_Test_RMSE    = sd(Test_RMSE, na.rm = TRUE),
          mean_Train_MAE  = mean(Training_MAE, na.rm = TRUE),
          sd_Train_MAE    = sd(Training_MAE, na.rm = TRUE),
          mean_Test_MAE   = mean(Test_MAE, na.rm = TRUE),
          sd_Test_MAE     = sd(Test_MAE, na.rm = TRUE),
          mean_Train_R2   = mean(Training_R2, na.rm = TRUE),
          sd_Train_R2     = sd(Training_R2, na.rm = TRUE),
          mean_Test_R2    = mean(Test_R2, na.rm = TRUE),
          sd_Test_R2      = sd(Test_R2, na.rm = TRUE),
          mean_Validation_R2   = mean(Validation_R2, na.rm = TRUE),
          sd_Validation_R2     = sd(Validation_R2, na.rm = TRUE),
          mean_Validation_RMSE = mean(Validation_RMSE, na.rm = TRUE),
          sd_Validation_RMSE   = sd(Validation_RMSE, na.rm = TRUE),
          mean_Validation_MAE  = mean(Validation_MAE, na.rm = TRUE),
          sd_Validation_MAE    = sd(Validation_MAE, na.rm = TRUE),
          .groups = "drop"
        ) %>% arrange(mean_Test_RMSE)
    }
  } else summary_df <- tibble()

  shap_df <- if (shap && length(shap_rows)) dplyr::bind_rows(shap_rows) else NULL
  if (!is.null(shap_df)) {
    shap_df <- shap_df %>% mutate(target = target_var)
    shap_df_summary <- shap_df %>%
      group_by(Feature) %>%
      summarise(MeanAbsSHAP = mean(MeanAbsSHAP, na.rm = TRUE), .groups = "drop") %>%
      arrange(desc(MeanAbsSHAP)) %>%
      mutate(target = target_var)
  } else shap_df_summary <- NULL

  # annotate and write CSVs
  perf_df   <- perf_df   %>% mutate(target = target_var)
  imp_df    <- imp_df    %>% mutate(target = target_var)
  summary_df<- summary_df%>% mutate(target = target_var)

  readr::write_csv(perf_df,   file.path(outdir, paste0(target_var, "_performance_perFold.csv")))
  readr::write_csv(imp_df,    file.path(outdir, paste0(target_var, "_feature_importance.csv")))
  readr::write_csv(summary_df,file.path(outdir, paste0(target_var, "_performance_summary.csv")))
  if (!is.null(shap_df)) {
    readr::write_csv(shap_df,         file.path(outdir, paste0(target_var, "_shap_perFold.csv")))
    readr::write_csv(shap_df_summary, file.path(outdir, paste0(target_var, "_shap_summary.csv")))
  }

  list(performance = perf_df, importance = imp_df, shap = shap_df, summary = summary_df)
}
