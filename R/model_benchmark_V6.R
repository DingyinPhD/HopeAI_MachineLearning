suppressPackageStartupMessages({
  library(caret)
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(readr)
  library(pROC)
  library(kernelshap)
  library(shapviz)
  library(gausscov)
})

# ------------------ helpers ------------------
.supports_prob <- function(method) {
  info <- getModelInfo(method, regex = FALSE)[[1]]
  !is.null(info$prob)
}

.make_strata_bins <- function(y, k = 5, min_per_fold_bin = 3, max_bins = 10) {
  n <- sum(is.finite(y))
  Bmax <- min(max_bins, max(2, length(unique(y))))       # upper bound
  # try from min(5, Bmax) downward until bins are big enough
  for (B in seq(from = min(5, Bmax), to = 2, by = -1)) {
    q <- unique(quantile(y, probs = seq(0, 1, length.out = B + 1), na.rm = TRUE))
    if (length(q) < 3) next
    q[1] <- -Inf; q[length(q)] <- Inf
    bins <- cut(y, breaks = q, include.lowest = TRUE, ordered_result = TRUE)
    if (all(table(bins) >= k * min_per_fold_bin)) return(bins)
  }
  # fallback: 2 quantile bins
  cut(y, breaks = unique(quantile(y, probs = c(0, .5, 1), na.rm = TRUE)),
      include.lowest = TRUE, ordered_result = TRUE)
}

# Quantile-based bins for regression stratification
.make_strata_bins <- function(y, k, min_per_fold_bin = 3, max_bins = 10) {
  Bmax <- min(max_bins, max(2, length(unique(y))))
  for (B in seq(from = min(5, Bmax), to = 2, by = -1)) {
    qs <- unique(quantile(y, probs = seq(0, 1, length.out = B + 1), na.rm = TRUE))
    if (length(qs) < 3) next
    qs[1] <- -Inf; qs[length(qs)] <- Inf
    bins <- cut(y, breaks = qs, include.lowest = TRUE, ordered_result = TRUE)
    if (all(table(bins) >= k * min_per_fold_bin)) return(bins)
  }
  cut(y, breaks = unique(quantile(y, probs = c(0, .5, 1), na.rm = TRUE)),
      include.lowest = TRUE, ordered_result = TRUE)
}

# Sellect top features using gauss distribution
.select_features_gausscov <- function(
    y_tr, X_tr,
    lm = 25, p0 = 0.01,
    kmn = 0, kmx = 0, qq = -1,
    approx_strategy = c("last","union","first"),
    vc = 0.01, nu = NULL
){
  approx_strategy <- match.arg(approx_strategy)

  # ---- 0) Coerce y to a plain numeric vector
  if (is.data.frame(y_tr)) {
    if (ncol(y_tr) != 1) stop("y_tr must be a single column or a numeric vector.")
    y_tr <- y_tr[[1]]
  }
  y_tr <- as.numeric(y_tr)
  if (!all(is.finite(y_tr))) stop("y_tr contains non-finite values.")

  # ---- 1) Clean X: numeric only, finite, variance > vc
  X_tr <- as.data.frame(X_tr)  # drop tibble class
  # drop list columns
  is_listcol <- vapply(X_tr, is.list, TRUE)
  if (any(is_listcol)) X_tr <- X_tr[ , !is_listcol, drop = FALSE]
  # coerce factors/characters to numeric if needed (safer: drop them)
  is_num <- vapply(X_tr, is.numeric, TRUE)
  X_tr <- X_tr[ , is_num, drop = FALSE]
  if (!ncol(X_tr)) return(character(0))

  # finite only
  X_tr <- X_tr[ , colSums(!is.finite(as.matrix(X_tr))) == 0, drop = FALSE]
  if (!ncol(X_tr)) return(character(0))

  # variance filter
  v <- vapply(X_tr, function(z) stats::var(z, na.rm = TRUE), numeric(1))
  keep <- which(is.finite(v) & v > vc)
  if (!length(keep)) return(character(0))
  X_tr <- X_tr[ , keep, drop = FALSE]

  # ensure unique names
  colnames(X_tr) <- make.names(colnames(X_tr), unique = TRUE)

  # ---- 2) Validate scalar numeric controls
  to_num_scalar <- function(z, nm) {
    if (length(z) != 1) stop(nm, " must be length-1.")
    z <- suppressWarnings(as.numeric(z))
    if (!is.finite(z)) stop(nm, " must be a finite numeric.")
    z
  }
  lm  <- to_num_scalar(lm,  "lm")
  p0  <- to_num_scalar(p0,  "p0")
  kmn <- to_num_scalar(kmn, "kmn")
  kmx <- to_num_scalar(kmx, "kmx")
  qq  <- to_num_scalar(qq,  "qq")
  if (kmn < 0 || kmx < 0) stop("kmn/kmx must be >= 0.")
  if (kmx > 0 && kmn > kmx) stop("kmn cannot exceed kmx.")

  # ---- 3) Call f2st with a plain numeric matrix
  x_mat <- as.matrix(X_tr)
  storage.mode(x_mat) <- "double"

  f2_formals <- names(formals(gausscov::f2st))
  args <- list(y = y_tr, x = x_mat, lm = lm, p0 = p0, kmn = kmn, kmx = kmx, qq = qq)
  if (!is.null(nu) && "nu" %in% f2_formals) args$nu <- nu

  b <- try(do.call(gausscov::f2st, args), silent = TRUE)
  if (inherits(b, "try-error")) {
    stop("gausscov::f2st failed: ", as.character(b))
  }
  if (is.null(b) || !length(b) || nrow(b[[1]]) == 0) return(character(0))

  pv <- b[[1]]  # columns: approx_id, covariate_index, pvalue, ...
  approx_ids <- unique(pv[,1])

  idx <- switch(
    approx_strategy,
    first = pv[pv[,1] == approx_ids[1], 2],
    last  = pv[pv[,1] == tail(approx_ids, 1), 2],
    union = pv[,2]
  )
  idx <- as.integer(idx)
  idx <- idx[idx >= 1 & idx <= ncol(x_mat)]
  feats <- unique(colnames(X_tr)[idx])

  # hard-cap if using union and kmx > 0
  if (approx_strategy == "union" && kmx > 0 && length(feats) > kmx) {
    ord <- order(pv[,1], seq_len(nrow(pv)))
    keep_names <- unique(colnames(X_tr)[pv[ord, 2]])
    feats <- keep_names[seq_len(kmx)]
  }
  feats
}



# Make outer folds; optionally stratify by a precomputed label (e.g., CancerLabel)
.make_outer_folds <- function(y, k_outer, seed = 123,
                              strata_label = NULL,   # e.g., uu$CancerLabel (factor/character)
                              returnTrain = TRUE) {
  set.seed(seed)

  if (is.factor(y)) {
    strata <- y
  } else {
    bins <- .make_strata_bins(y, k = k_outer, min_per_fold_bin = 3)
    if (!is.null(strata_label)) {
      lab <- if (is.factor(strata_label)) strata_label else factor(strata_label)
      strata <- interaction(lab, bins, drop = TRUE)
    } else {
      strata <- bins
    }
  }

  caret::createFolds(strata, k = k_outer, returnTrain = returnTrain)
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
    gausscov_lm = 25,
    gausscov_p0 = 0.01,
    gausscov_kmn = 0,
    gausscov_kmx = 0,
    gausscov_qq = -1,
    gausscov_approx_strategy = "last", # choose from c("last", "union", "first")
    gausscov_vc = 0.01,
    model_grids = list()
) {
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

  # ---- handle target as formula or string ----
  if (inherits(target, "formula")) {
    target_var <- all.vars(target)[1]
    form <- target
  } else if (is.character(target) && length(target) == 1) {
    target_var <- target
    form <- as.formula(paste(target_var, "~ ."))  # simple form if user gave a name
  } else {
    stop("`target` must be a formula (e.g., y ~ (.)^2) or a single character column name.")
  }
  if (!(target_var %in% colnames(data))) stop("Target variable not found in `data`: ", target_var)

  # =========================
  # EXPAND DESIGN MATRIX UP FRONT (your request)
  # =========================
  # Expand RHS using the provided form, with data-aware '.' handling, then drop intercept.
  label_col <- "CancerLabel"
  if (!label_col %in% names(data)) stop("'", label_col, "' not found in data.")

  # 1) Get RHS term labels from the original formula
  tt   <- stats::terms(form, data = data)         # uses your data to resolve .
  rhs  <- attr(stats::delete.response(tt), "term.labels")

  # 2) Drop CancerLabel (and optionally interactions containing it)
  rhs_keep <- rhs[!grepl(paste0("^", label_col, "(\\b|:)"), rhs)]  # drops 'CancerLabel' and 'CancerLabel:...'

  # 3) Build the new formula: target ~ <kept terms>  (or ~ 1 if none left)
  rhs_str <- if (length(rhs_keep)) paste(rhs_keep, collapse = " + ") else "1"
  form_nolabel <- stats::as.formula(paste(target_var, "~", rhs_str))

  # 4) Now build the design matrix WITHOUT CancerLabel
  rhs_terms <- stats::terms(form_nolabel, data = data) |> stats::delete.response()
  mm_all <- stats::model.matrix(rhs_terms, data = data)
  if (ncol(mm_all) > 0 && colnames(mm_all)[1] == "(Intercept)") {
    mm_all <- mm_all[, -1, drop = FALSE]
  }

  # 5) Rebuild the modeling data: target + expanded numeric features + original CancerLabel column
  data_mm <- data.frame(
    setNames(list(data[[target_var]]), target_var),
    mm_all,
    setNames(list(data[[label_col]]), label_col),
    check.names = FALSE
  )

  # 6) Make syntactic names & keep target name consistent
  nm <- make.names(names(data_mm), unique = TRUE)
  names(data_mm) <- nm
  target_var_clean <- make.names(target_var)
  names(data_mm)[1] <- target_var_clean

  data  <- data_mm
  form2 <- as.formula(paste(target_var_clean, "~ ."))



  message("form2 is: ", paste(deparse(form2), collapse = " "))

  readr::write_csv(data,   file.path(outdir, paste0(target_var, "_input_data.csv")))

  # ---- task & splits ----
  task <- if (is.factor(data[[target_var]])) "Classification" else "Regression"
  X_all <- data %>% dplyr::select(-all_of(target_var))
  y_all <- data[[target_var]]

  # Optional CancerLabel stratification (assumed precomputed outside)
  cat("is.data.frame(data):", is.data.frame(data), "\n")
  cat("type of data:", class(data), "\n")

  cat("Has CancerLabel? ",
      "CancerLabel" %in% names(data), "\n")

  cat("Exact match positions: ",
      paste(which(names(data) == "CancerLabel"), collapse = ","), "\n")

  cat("Grepped positions: ",
      paste(grep("^\\s*CancerLabel\\s*$", names(data)), collapse = ","), "\n")

  cat("Name with possible whitespace trimmed? ",
      any(trimws(names(data)) == "CancerLabel"), "\n")

  print(tail(names(data), 10))  # eyeball the end


  strata_col <- if ("CancerLabel" %in% names(data)) data[["CancerLabel"]] else NULL

  cat("strata_col is:",
      if (is.null(strata_col)) "NULL" else
        sprintf("(%s) length %d; head: %s",
                paste(class(strata_col), collapse="/"),
                length(strata_col),
                paste(head(as.character(strata_col), 6), collapse=", ")),
      "\n")


  outer_folds <- .make_outer_folds(
    y           = y_all,   # target vector
    k_outer     = k_outer,
    seed        = seed,
    strata_label= strata_col,   # <- your precomputed label
    returnTrain = TRUE              # FALSE if you want test indices
  )


  # ---- inner CV controls ----
  ctrl_class <- trainControl(
    method = "cv", number = k_inner,
    classProbs = TRUE, summaryFunction = twoClassSummary,
    savePredictions = "final", allowParallel = TRUE,
    selectionFunction = "best"
  )
  ctrl_reg <- trainControl(
    method = "cv", number = k_inner,
    savePredictions = "final", allowParallel = TRUE,
    selectionFunction = "best"
  )

  # ---- default tuning grids ----
  p <- ncol(X_all)
  default_grids <- list(
    rf = data.frame(mtry = unique(pmax(1L, round(c(sqrt(p)/2, sqrt(p), sqrt(p)*2, p/3, p/2))))),
    ranger = expand.grid(
      mtry = unique(pmax(1L, round(c(sqrt(p)/2, sqrt(p), sqrt(p)*2, p/3, p/2)))),
      splitrule = c("variance", "extratrees", "gini"),
      min.node.size = c(1, 3, 5, 7, 10)
    ),
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
  perf_rows <- list(); imp_rows <- list(); shap_rows <- list(); shap_int_rows <- list()

  # ---- outer CV loop ----
  for (i in seq_along(outer_folds)) {
    message(sprintf("=== Outer fold %d / %d ===", i, length(outer_folds)))
    tr_idx <- outer_folds[[i]]
    te_idx <- setdiff(seq_len(nrow(data)), tr_idx)

    X_tr <- X_all[tr_idx, , drop = FALSE]; y_tr <- y_all[tr_idx]
    X_te <- X_all[te_idx, , drop = FALSE]; y_te <- y_all[te_idx]

    feats <- .select_features_gausscov(
      y_tr = y_tr,
      X_tr = X_tr,
      lm   = gausscov_lm,
      p0   = gausscov_p0,
      vc   = gausscov_vc,
      qq = gausscov_qq,
      approx_strategy = gausscov_approx_strategy,
      kmn = gausscov_kmn,
      kmx = gausscov_kmx
    )

    feat_df <- tibble::tibble(feature = feats)
    feat_path <- file.path(outdir, sprintf("%s_Fold%d_gausscov_feature.csv", target_var, i))
    readr::write_csv(feat_df, feat_path)

    # subset train/test to selected features
    X_tr_sel <- X_tr[, feats, drop = FALSE]
    X_te_sel <- X_te[, feats, drop = FALSE]

    train_df <- data.frame(y = y_tr, X_tr_sel)
    names(train_df)[1] <- target_var
    form2 <- as.formula(
      paste(target_var, "~", paste(make.names(colnames(X_tr_sel)), collapse = " + "))
    )


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
      if (method == "nnet") {
        extra$linout  <- (task == "Regression")
        extra$trace   <- FALSE
        extra$MaxNWts <- 250000
      }
      if (method == "rf") {
        extra$ntree    <- 500
        extra$nodesize <- if (task == "Classification") 1 else 5
      }
      if (method == "ranger") {
        extra$num.trees  <- 500
        extra$importance <- "impurity"
        # DO NOT set nodesize here; ranger uses min.node.size (already tuned via grid)
        if (task == "Classification") extra$probability <- TRUE  # optional, helps type="prob"
      }

      extra_name <- paste0(method, "_extra")
      if (extra_name %in% names(model_grids)) {
        #extra <- c(extra, model_grids[[extra_name]])
        extra <- utils::modifyList(extra, model_grids[[extra_name]])
      }


      # train on expanded columns
      # ---- model training ----
      if (method == "glmnet") {
        train_df_glmnet <- data.frame(y = y_tr, X_tr_sel, check.names = TRUE)
        names(train_df_glmnet)[1] <- target_var

        # Drop non-numeric predictor columns (e.g., CancerLabel factors) to avoid glmnet errors
        keep_pred <- c(TRUE, vapply(train_df_glmnet[-1], is.numeric, logical(1)))
        train_df_glmnet  <- train_df_glmnet[, keep_pred, drop = FALSE]

        # Build a formula that matches the current columns
        form_glmnet <- as.formula(paste(target_var, "~ ."))

        tuned <- try(
          do.call(caret::train, c(
            list(
              form = form_glmnet,
              data = train_df_glmnet,
              method = "glmnet",
              trControl = tr_ctrl,
              metric = metric,
              preProcess = c("center", "scale"),   # glmnet usually benefits from this
              family = if (is.factor(y_tr)) "binomial" else "gaussian"
            ),
            if (!is.null(tg)) list(tuneGrid = tg) else list(tuneLength = 50),
            extra
          )),
          silent = TRUE
        )

      } else {
        # all other methods: use formula + data
        train_df <- data.frame(y = y_tr, X_tr)
        names(train_df)[1] <- target_var

        tuned <- try(
          do.call(caret::train, c(
            list(form = form2, data = train_df, method = method, trControl = tr_ctrl, metric = metric),
            if (!is.null(tg)) list(tuneGrid = tg) else list(tuneLength = 5),
            extra
          )),
          silent = TRUE
        )
      }

      # ---- error handling ----
      if (inherits(tuned, "try-error")) {
        warning(sprintf("[Fold %d] %s failed: %s", i, method, as.character(tuned)))
        next
      }

      # save model
      saveRDS(tuned, file = file.path(outdir, sprintf("%s_%s_Fold%d.rds", target_var, method, i)))

      # inner-CV metrics
      inner_metrics <- .inner_fold_validation_metrics(
        tuned_model = tuned,
        task = if (is.factor(y_tr)) "Classification" else "Regression"
      )

      # -----------------------
      # training metrics
      # -----------------------
      print("computing training metrics")
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
        # ---- Regression ----
        if (inherits(tuned$finalModel, "glmnet")) {
          # Bypass caret’s formula parsing for glmnet
          mm_tr <- as.matrix(X_tr)
          train_df_glmnet <- data.frame(X_tr_sel, check.names = TRUE)
          test_df_glmnet  <- data.frame(X_te_sel, check.names = TRUE)

          # drop non-numeric predictors (glmnet can’t use raw factors/chars)
          is_num_tr <- vapply(train_df_glmnet, is.numeric, logical(1))
          is_num_te <- vapply(test_df_glmnet,  is.numeric, logical(1))
          keep_cols <- intersect(names(train_df_glmnet)[is_num_tr],
                                 names(test_df_glmnet)[is_num_te])

          train_df_glmnet <- train_df_glmnet[, keep_cols, drop = FALSE]
          test_df_glmnet  <- test_df_glmnet[,  keep_cols, drop = FALSE]

          pred_tr <- as.numeric(predict(tuned, newdata = train_df_glmnet))
        } else {
          pred_tr <- as.numeric(predict(tuned, newdata = as.data.frame(X_tr_sel)))
        }
        m_tr <- .compute_metrics(task, y_tr, pred_tr)
      }

      # -----------------------
      # testing metrics
      # -----------------------
      print("computing testing metrics")
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
        # ---- Regression ----
        if (inherits(tuned$finalModel, "glmnet")) {
          # same bypass for test data
          mm_te <- as.matrix(X_te)
          pred_te <- as.numeric(predict(tuned, newdata = test_df_glmnet))
        } else {
          pred_te <- as.numeric(predict(tuned, newdata = as.data.frame(X_te_sel)))
        }
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
      print("computing varImp")
      vi <- try(varImp(tuned)$importance, silent = TRUE)
      if (!inherits(vi, "try-error")) {
        vi_df <- vi %>% tibble::rownames_to_column("Feature") %>% mutate(Algorithm = method, Fold = i)
        imp_rows[[length(imp_rows) + 1]] <- vi_df
      }

      # After saving tuned and before SHAP
      message("Class of tuned object: ", class(tuned)[1])

      # ---- SHAP (now on expanded features; no model.matrix needed) ----
      print("computing shap")
      if (shap) {
        bg_n  <- min(shap_bg_max, nrow(X_tr))
        bg_ix <- sample.int(nrow(X_tr), size = bg_n, replace = FALSE)

        # keep as data.frames (preserve factors if any)
        bg_df        <- as.data.frame(X_tr[bg_ix, , drop = FALSE])
        X_explain_df <- as.data.frame(
          if (is.finite(shap_pred_max)) X_te[seq_len(min(nrow(X_te), shap_pred_max)), , drop = FALSE] else X_te
        )

        is_xgb    <- identical(method, "xgbTree")
        is_ranger <- identical(method, "ranger")

        if (is_xgb || is_ranger) {
          # ============================
          # TreeSHAP (main + interactions)
          # ============================
          suppressPackageStartupMessages({
            require(treeshap)
            require(shapviz)
          })

          # Unify model for treeshap
          X_tr_mat <- as.matrix(X_tr)
          X_ex_mat <- as.matrix(X_explain_df)

          uni <- NULL
          if (is_xgb) {
            # caret::train(method="xgbTree") stores the xgboost model in tuned$finalModel
            # Unify to a treeshap-compatible structure
            uni <- treeshap::xgboost.unify(tuned$finalModel, X_tr_mat)
          } else if (is_ranger) {
            # Works for caret::train(method="ranger") objects
            uni <- treeshap::ranger.unify(tuned$finalModel, X_tr_mat)
          }

          # Compute TreeSHAP with interactions
          ts <- treeshap::treeshap(uni, X_ex_mat, interactions = TRUE)
          saveRDS(ts, file = file.path(outdir, sprintf("%s_%s_Fold%d.treeshap.rds", target_var, method, i)))

          # ---- Main effect importance (mean |SHAP| per feature) ----
          sv <- shapviz::shapviz(ts)  # shapviz understands treeshap objects
          imp_tbl <- shapviz::sv_importance(sv, kind = "no", show_numbers = TRUE) |>
            as.data.frame() |>
            tibble::rownames_to_column("Feature") |>
            dplyr::mutate(Algorithm = method, Fold = i)
          colnames(imp_tbl)[2] <- "MeanAbsSHAP"
          shap_rows[[length(shap_rows) + 1]] <- imp_tbl

          # ---- Pairwise interaction strengths (mean |interaction SHAP|) ----
          # ts$interactions: array [n_samples, p, p]
          SI <- ts$interactions
          # mean absolute interaction across samples
          M <- apply(abs(SI), c(2, 3), mean, na.rm = TRUE)
          # tidy upper triangle (no diagonal)
          p <- ncol(M)
          ut <- which(upper.tri(M), arr.ind = TRUE)
          interaction_df <- tibble::tibble(
            FeatureA = colnames(X_ex_mat)[ut[, 1]],
            FeatureB = colnames(X_ex_mat)[ut[, 2]],
            MeanAbsInteractionSHAP = M[ut],
            Algorithm = method,
            Fold = i
          ) |>
            dplyr::arrange(dplyr::desc(MeanAbsInteractionSHAP))

          # Write interactions table per fold (optional but handy)
          readr::write_csv(
            interaction_df,
            file.path(outdir, sprintf("%s_%s_Fold%d_SHAP_interactions.csv", target_var, method, i))
          )

          # stash for after-loop aggregation
          shap_int_rows[[length(shap_int_rows) + 1]] <- interaction_df

          # ---- Main effect SHAP importances ----
          main_df <- shapviz::sv_importance(sv, kind = "no", show_numbers = TRUE) |>
            as.data.frame() |>
            tibble::rownames_to_column("Feature")

          # Identify the numeric importance column automatically
          imp_col <- names(main_df)[sapply(main_df, is.numeric)][1]

          # Rename it to a standard name
          main_df <- main_df |>
            dplyr::rename(MeanAbsMainEffectSHAP = !!imp_col) |>
            dplyr::mutate(Algorithm = method, Fold = i)

          # Save to CSV
          readr::write_csv(
            main_df,
            file.path(outdir, sprintf("%s_%s_Fold%d_SHAP_mainEffects.csv", target_var, method, i))
          )


          if (save_plots) {
            pdf(file.path(outdir, sprintf("Fold%d_%s_SHAP_importance.pdf", i, method)))
            shapviz::sv_importance(sv, kind = "both", show_numbers = TRUE, max_display = 20)
            dev.off()

            # simple heatmap of interaction strengths (optional)
            hm_file <- file.path(outdir, sprintf("Fold%d_%s_SHAP_interactions_heatmap.pdf", i, method))
            try({
              pdf(hm_file, width = 7, height = 7)
              op <- par(mar = c(6, 6, 2, 1))
              image(
                t(M[nrow(M):1, ]), axes = FALSE, main = "Mean |SHAP interaction|"
              )
              axis(1, at = seq(0, 1, length.out = p), labels = colnames(M), las = 2, cex.axis = 0.6)
              axis(2, at = seq(0, 1, length.out = p), labels = rev(colnames(M)), las = 2, cex.axis = 0.6)
              par(op); dev.off()
            }, silent = TRUE)
          }

        } else {
          # ============================
          # Fallback: kernel SHAP (no explicit interaction matrix)
          # ============================
          if (method == "glmnet") {
            # IMPORTANT: bypass caret's formula; use glmnet directly on matrices
            pred_fun <- function(object, newdata) {
              mm <- as.matrix(newdata)
              as.numeric(predict(object$finalModel, newx = mm, s = object$bestTune$lambda))
            }
          } else if (task == "Classification" && .supports_prob(method)) {
            pos <- levels(y_tr)[2]
            pred_fun <- function(object, newdata) {
              probs <- predict(object, newdata = as.data.frame(newdata), type = "prob")
              as.numeric(probs[[pos]])
            }
          } else {
            pred_fun <- function(object, newdata) {
              as.numeric(predict(object, newdata = as.data.frame(newdata)))
            }
          }

          # Optional: neutralize caret's symbolic formula so predict() won't rebuild interactions
          print("triggering kernelshap")

          ks <- kernelshap::kernelshap(
            tuned,
            X    = X_explain_df,
            bg_X = bg_df,
            pred_fun = pred_fun
          )
          saveRDS(ks, file = file.path(outdir, sprintf("%s_%s_Fold%d.kernelshap.rds", target_var, method, i)))

          sv <- shapviz::shapviz(ks, X = X_explain_df, X_pred = as.matrix(X_explain_df))
          imp_tbl <- shapviz::sv_importance(sv, kind = "no", show_numbers = TRUE) |>
            as.data.frame() |>
            tibble::rownames_to_column("Feature") |>
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
  }

  # -------- aggregate & write outputs --------
  perf_df <- dplyr::bind_rows(perf_rows)
  imp_df  <- dplyr::bind_rows(imp_rows)

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
  if (length(shap_int_rows)) {
    all_interactions <- dplyr::bind_rows(shap_int_rows)

    # canonicalize pairs so A–B and B–A collapse together
    all_interactions <- all_interactions |>
      dplyr::mutate(
        FA = pmin(FeatureA, FeatureB),
        FB = pmax(FeatureA, FeatureB)
      )

    # per-pair summary across folds
    interaction_summary <- all_interactions |>
      dplyr::group_by(FA, FB) |>
      dplyr::summarise(MeanAbsInteractionSHAP = mean(MeanAbsInteractionSHAP, na.rm = TRUE),
                       .groups = "drop") |>
      dplyr::arrange(dplyr::desc(MeanAbsInteractionSHAP)) |>
      dplyr::rename(FeatureA = FA, FeatureB = FB) |>
      dplyr::mutate(target = target_var)

    # write cross-fold tables
    readr::write_csv(
      all_interactions,
      file.path(outdir, sprintf("%s_%s_SHAP_interactions_allFolds.csv", target_var, method))
    )
    readr::write_csv(
      interaction_summary,
      file.path(outdir, sprintf("%s_%s_SHAP_interactions_summary.csv", target_var, method))
    )
  }

  list(performance = perf_df, importance = imp_df, shap = shap_df, summary = summary_df)
}

