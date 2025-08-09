library(adabag)
library(caret)
library(data.table)
library(e1071)
#library(ggbeeswarm)
#library(ggpubr)
library(glmnet)
library(iml)
library(kernelshap)
library(neuralnet)
library(pROC)
library(randomForest)
library(xgboost)
#library(shapley)
#library(shapviz)
library(tidyverse)
library(doParallel)
library(foreach)
library(rlang)

# The purpose of this script is to determine which machine learning algorithm fit best to our datasets
# Workflow:
# 1. Select a few  gene (methylation)- gene (dependency) pair
# 2. For each pair, train the model ~1,000 times
# 3. Check the consistency of the Hyperparameter of the best tune.
# 4. Compare the accuracy, precision, F1 score, ROAUC
# 5. Model selection

# Define Gloal Variables ===============================================================================================================================

# Setting input gene (methylation)- gene (dependency) pair
model_benchmark_V4 <- function(Features,
                               Target,
                               Input_Data,
                               max_tuning_iteration = 100,
                               fold = 10, # k-fold cross validation
                               model = c("Random Forest", "Naïve Bayes", "Elastic Net", "SVM",
                                         "XGBoost", "AdaBoost", "Neural Network", "KNN", "Decision Tree"),
                               model_type,
                               dependency_threshold,
                               gene_hits_percentage_cutoff_Lower = 0.2,
                               gene_hits_percentage_cutoff_Upper = 0.8,
                               XBoost_tuning_grid = "Simple",
                               Finding_Optimal_Threshold = TRUE,
                               testing_percentage = NA,
                               SHAP = FALSE) {

  Features <- Features
  Dependency_gene <- Target
  threshold <- dependency_threshold
  cutoff_Lower <- gene_hits_percentage_cutoff_Lower
  cutoff_Upper <- gene_hits_percentage_cutoff_Upper
  XBoost_tuning_grid <- XBoost_tuning_grid
  Finding_Optimal_Threshold <- Finding_Optimal_Threshold
  # Setting Machine learning algorithm for benchmarking
  ML_model <- model
  model_type <- model_type
  SHAP <- SHAP
  testing_percentage <- testing_percentage
  training_percentage <- 1 - testing_percentage

  if (!(model_type %in% c("Classification", "Regression"))) {
    stop("Invalid model type: must be 'Classification' or 'Regression'")
  }


  # Setting iteration time
  max_tuning_iteration <- max_tuning_iteration # Defined in the function

  # Setting Global training parameters
  ctrlspecs <- trainControl(method = "repeatedcv", number = fold, repeats = max_tuning_iteration,
                            savePredictions = "all", allowParallel = TRUE) # fold is defined in the function

  ctrl_svm <- trainControl(
    method = "repeatedcv", number = 10, repeats = 10,
    classProbs = TRUE,
    summaryFunction = twoClassSummary,
    savePredictions = "final",
    allowParallel = TRUE
  )


  # Setting parallel core number
  num_cores <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK"))
  if (is.na(num_cores)) {
    num_cores <- detectCores() - 1  # Use all cores except one
  }

  print(paste0("Will use ", num_cores, " cores for parallel processing"))

  # Setting input data
  merge_data <- Input_Data  # Create a copy of the input data frame
  print(merge_data$TP53_snv)
  #merge_data[] <- lapply(merge_data, as.numeric)
  print(str(merge_data))
  merge_data <- na.omit(merge_data)
  print(merge_data$TP53_snv)

  if (nrow(merge_data) == 0 || all(is.na(merge_data[[Dependency_gene]]))) {
    cat(Dependency_gene, "failed to run\n",
        file = "low coverage gene.txt",
        append = TRUE)
    stop(paste(Dependency_gene, "has no usable data"))
  }

  print("First test")
  print(colnames(merge_data))

  # Setting final output file
  final_benchmark_result <- data.frame()
  final_benchmark_result_write_out_filename <- paste0(Features,"_",Target,"_benchmarking_result.csv")

  # Function to calculate and rank feature importance ---
  # If using RandomForest package
  rank_feature_importance_RF <- function(model) {
    # Convert importance matrix to data frame
    imp <- as.data.frame(importance(model))

    # Try to use MeanDecreaseGini if it exists
    if ("MeanDecreaseGini" %in% colnames(imp)) {
      imp$score <- imp$MeanDecreaseGini
    } else if ("IncNodePurity" %in% colnames(imp)) {
      imp$score <- imp$IncNodePurity  # for regression models
    } else {
      stop("No suitable importance score column found in the model.")
    }

    # Normalize score
    imp <- imp %>%
      mutate(normalized = (score - min(score) + 1e-4) / (max(score) - min(score)))

    # Sort by importance
    ordered_imp <- imp[order(imp$score, decreasing = TRUE), ]
    ordered_imp <- ordered_imp %>%
      mutate(comb = paste0(rownames(ordered_imp), "_", normalized))

    cg_string <- paste(ordered_imp$comb, collapse = "|")
    return(cg_string)
  }


  # If using caret package
  rank_feature_importance_caret <- function(model) {
    ordered_imp <- tryCatch({
      imp <- as.data.frame(varImp(model)$importance)

      # Use the first column if only one is present, otherwise compute rowMeans
      if (ncol(imp) == 1) {
        imp$Score <- imp[, 1]
      } else {
        imp$Score <- rowMeans(imp, na.rm = TRUE)
      }

      imp %>%
        dplyr::select(Score) %>%
        rownames_to_column("Feature") %>%
        arrange(desc(Score)) %>%
        mutate(comb = paste0(Feature, "_", round(Score / max(Score + 1e-6), 3)))
    }, error = function(e) {
      warning("Feature importance could not be computed: ", e$message)
      return(NULL)
    })

    # Return string only if importance was successfully computed
    if (!is.null(ordered_imp)) {
      cg_string <- paste(ordered_imp$comb, collapse = "|")
    } else {
      cg_string <- NA
    }

    return(cg_string)
  }



  # If using xgboost package
  rank_feature_importance_xgboost <- function(model) {
    ordered_imp <- tryCatch({
      imp <- xgb.importance(model = model)

      # Check if importance object is valid
      if (is.null(imp) || nrow(imp) == 0) return(NA)

      # Rename columns explicitly if needed (safety for weird names)
      colnames(imp)[colnames(imp) == "Gain"] <- "Gain"
      colnames(imp)[colnames(imp) == "Feature"] <- "Feature"

      # Normalize Gain
      gain_min <- min(imp$Gain, na.rm = TRUE)
      gain_max <- max(imp$Gain, na.rm = TRUE)

      if (gain_max != gain_min) {
        imp$normalized <- (imp$Gain - gain_min + 1e-4) / (gain_max - gain_min)
      } else {
        imp$normalized <- 1
      }

      # Format feature_comb string
      imp$comb <- paste0(imp$Feature, "_", round(imp$normalized, 6))

      return(paste(imp$comb, collapse = "|"))

    }, error = function(e) {
      warning("Failed to extract xgboost feature importance: ", e$message)
      return(NA)
    })

    return(ordered_imp)
  }


  # If using e1071::svm package
  rank_feature_importance_e1071 <- function(model, train_df, Dependency_gene) {
    # Step 1: Get the predictor variable names (excluding the target)
    predictor_vars <- setdiff(colnames(train_df), Dependency_gene)

    # Step 2: Calculate importance from SVM coefficients
    coefficients <- t(model$coefs) %*% model$SV
    importance <- abs(coefficients)
    importance <- as.vector(importance)
    importance <- importance / max(importance)  # Normalize

    # Step 3: Check for mismatch between features and importance scores
    if (length(importance) != length(predictor_vars)) {
      # Adjust if off by 1 (common with SVM dropping 1 constant col)
      min_len <- min(length(importance), length(predictor_vars))
      importance <- importance[1:min_len]
      predictor_vars <- predictor_vars[1:min_len]
      warning("Feature and importance lengths mismatched; truncated to match.")
    }

    # Step 4: Create and rank importance dataframe
    importance_df <- data.frame(
      Variable = predictor_vars,
      Importance = importance
    )

    ordered_imp <- importance_df %>%
      arrange(desc(Importance)) %>%
      mutate(comb = paste0(Variable, "_", round(Importance, 4)))

    # Step 5: Combine into a single string
    cg_string <- paste(ordered_imp$comb, collapse = "|")

    return(cg_string)
  }




  # Function to calculate optimal_threshold ---
  evaluate_with_optimal_threshold <- function(training_pred,
                                              train_labels,
                                              positive_class = "1",
                                              negative_class = "0",
                                              model_type,
                                              Finding_Optimal_Threshold = TRUE) {
    # Convert labels to factor for classification
    #if (model_type == "Classification") {
    #  train_labels <- factor(train_labels)
    #}

    # Initialize outputs
    roc_curve <- NULL
    auroc <- NULL
    threshold_value <- 0.5
    conf_matrix_train <- NULL

    if (model_type == "Classification") {
      # Check input
      if (is.null(training_pred)) {
        stop("Training predicted probabilities are required.")
      }

      # Compute ROC/threshold from training data
      tryCatch({
        roc_curve <- roc(train_labels, training_pred)
        auroc <- auc(roc_curve)
      }, error = function(e) {
        message("ROC calculation failed: ", e$message)
      })

      # Determine threshold
      if (!is.null(roc_curve) & !is.null(auroc)) {
        if (Finding_Optimal_Threshold) {
          threshold_value <- coords(roc_curve, "best", ret = "threshold")[[1]]
        }

        training_preds <- ifelse(training_pred > threshold_value, positive_class, negative_class)
        #training_preds <- factor(training_preds, levels = levels(train_labels))
        train_ref  <- factor(as.character(train_labels), levels = c("0","1"))
        train_pred <- factor(as.character(training_preds), levels = levels(train_ref))

        print(train_pred)
        print(train_ref)
        print(positive_class)
        #conf_matrix_train <- confusionMatrix(training_preds, train_labels, positive = positive_class)
        conf_matrix_train <- confusionMatrix(train_pred, train_ref, positive = positive_class)
        print("Done conf_matrix")


      } else {
        conf_matrix_train <- list()
        auroc <- -1
      }

      return(list(
        model_type = model_type,
        roc = roc_curve,
        training_auroc = round(auroc, 3),
        optimal_threshold = round(threshold_value, 3),
        training_accuracy = round(conf_matrix_train$overall["Accuracy"], 3),
        training_accuracyPValue = round(conf_matrix_train$overall["AccuracyPValue"], 3),
        training_precision = round(conf_matrix_train$byClass["Precision"], 3),
        training_recall = round(conf_matrix_train$byClass["Recall"], 3),
        training_F1 = round(conf_matrix_train$byClass["F1"], 3),
        training_Kappa = round(conf_matrix_train$overall["Kappa"], 3),
        training_McnemarPValue = round(conf_matrix_train$overall["McnemarPValue"], 3)
      ))

    } else if (model_type == "Regression") {
      # For regression, compute RMSE, MAE, and R^2
      rmse_train <- sqrt(mean((train_labels - training_pred)^2))
      mae_train <- mean(abs(train_labels - training_pred))
      r2_train <- cor(train_labels, training_pred)^2

      return(list(
        model_type = model_type,
        training_rmse = round(rmse_train, 3),
        training_mae = round(mae_train, 3),
        training_r2 = round(r2_train, 3)
      ))

    } else {
      stop("Invalid model type: must be 'Classification' or 'Regression'")
    }
  }


  calculate_SHAP <- function(model, train_df, test_df, output_pdf) {
    s <- kernelshap(model,
                    X = test_df[,-1], # Remove the predictor column
                    bg_X = train_df,
                    type = "prob")  # type = "prob" for binomial prediction
    sv <- shapviz(s)

    pdf(output_pdf)
    sv_importance(sv, kind = "bee", show_numbers = TRUE, max_display = Inf) # set max_display = Inf to plot all features
    dev.off()
    sv_importance(sv, kind = "no", show_numbers = TRUE) # No plot, just print the mean(abs(SHAP)) value

  }



  # Function to safely extract value from null datasets (Not use in RF)
  safe_extract <- function(x, field) {
    if (!is.null(x) && !is.null(x[[field]])) x[[field]] else NA
  }


  # Define function to register and stop CPU cluster for parallel processing ---
  start_cluster <- function(num_cores, max_retries = 3, delay_secs = 1) {
    attempt <- 1
    repeat {
      message("Starting cluster (attempt ", attempt, ")...")
      cl <- tryCatch({
        makeCluster(num_cores)
      }, error = function(e) {
        message("makeCluster failed: ", e$message)
        return(NULL)
      })

      if (!is.null(cl)) {
        # Test worker connectivity
        test_result <- tryCatch({
          parallel::clusterCall(cl, function() TRUE)
        }, error = function(e) {
          message("ClusterCall failed: ", e$message)
          return(NULL)
        })

        if (!is.null(test_result) && all(unlist(test_result))) {
          doParallel::registerDoParallel(cl)
          message("✅ All workers connected.")
          return(invisible(cl))
        } else {
          parallel::stopCluster(cl)
          message("❌ Some workers failed to connect.")
        }
      }

      # Retry logic
      if (attempt >= max_retries) {
        stop("❗ Failed to start and connect all workers after ", max_retries, " attempts.")
      } else {
        Sys.sleep(delay_secs)
        attempt <- attempt + 1
      }
    }
  }

  stop_cluster <- function(cl) {
    if (!is.null(cl)) {
      parallel::stopCluster(cl)
      doParallel::registerDoSEQ()
      message("Cluster stopped and reverted to sequential processing.")
    }
  }


  # Calculate the proportion of hits that are less than threshold
  fraction_below <- mean(merge_data[[Dependency_gene]] < threshold, na.rm = TRUE)

  print(paste0("fraction_below:", fraction_below))
  print(Dependency_gene)
  print(merge_data[[Dependency_gene]])

  if (fraction_below < cutoff_Lower || fraction_below > cutoff_Upper) {
    print(paste0("gene hits percentage for ", Dependency_gene, " is ", mean(merge_data[[Dependency_gene]] < threshold, na.rm = TRUE),
                 " which is less than ", cutoff_Lower, ", thus skip model benchmarking"))
    final_benchmark_result <- rbind(final_benchmark_result,
                                    data.frame(Algorithm = NA,
                                               Hyperparameter = NA,
                                               Tuned_Value = NA,

                                               # Classification metrics
                                               Optimal_Threshold = NA,
                                               Training_Accuracy = NA,
                                               Training_Precision = NA,
                                               Training_Recall = NA,
                                               Training_F1 = NA,
                                               Training_Kappa = NA,
                                               Training_AccuracyPValue = NA,
                                               Training_McnemarPValue = NA,
                                               Training_AUROC = NA,
                                               Validation_Accuracy = NA,
                                               Validation_Kappa = NA,

                                               # Regression metrics
                                               Training_RMSE = NA,
                                               Training_MAE = NA,
                                               Training_R2 = NA,
                                               Validation_RMSE = NA,
                                               Validation_Rsq = NA,

                                               # Shared metrics
                                               time_taken = NA,
                                               feature_importance = NA,
                                               max_tuning_iteration = NA,
                                               gene_hits_percentage_cutoff_Lower = NA,
                                               gene_hits_percentage_cutoff_Upper = NA,
                                               model_type = NA))

    assign("final_benchmark_result", final_benchmark_result, envir = .GlobalEnv)  # Save in global env

  } else {
    # Script Start ===============================================================================================================================

    merge_data <- merge_data

    # Create training and test datasets
    if (model_type == "Classification") {
      merge_data <- merge_data %>%
        mutate(!!sym(Dependency_gene) := case_when(
          !!sym(Dependency_gene) <= threshold ~ 1,
          TRUE ~ 0
        )) %>%
        mutate(!!sym(Dependency_gene) := as.factor(!!sym(Dependency_gene)))

    } else if (model_type == "Regression") {
      # No changes needed for regression
      merge_data <- merge_data

    } else {
      stop("Invalid model type: must be 'Classification' or 'Regression'")
    }


    merge_data <- merge_data[, colMeans(is.na(merge_data)) < 0.5] # filter out columns with more than 50% missing values
    #merge_data <- na.omit(merge_data)
    set.seed(123)

    train_df <- merge_data # no train test split

    # Train each model ~1,000 times
    for (MLmodel in ML_model) {
      if (MLmodel == "Random Forest") {
        # To capture timestamps and compute the duration
        start_time <- Sys.time()

        # Benchmarking Random Forest ---------------------------------------------------------------
        print("Benchmarking Random Forest Start")

        # Initialize empty result storage
        RF_benchmark <- data.frame()
        ntree_to_try <- seq(100, 1000, by = 100)
        index_of_target <- which(colnames(train_df) == Dependency_gene)

        for (i in 1:max_tuning_iteration) {
          for (ntree in ntree_to_try) {
            tmp <- tryCatch({
              tuneRF(
                x = train_df[, -index_of_target],
                y = train_df[, index_of_target],
                ntreeTry = ntree,
                stepFactor = 1.5,
                improve = 0.01, # Only continue tuning if the OOB error decreases by more than 1%.
                doBest = FALSE,
                trace = FALSE
              )
            }, error = function(e) {
              message(paste("Error at iteration", i, "with ntree =", ntree, ":", e$message))
              return(NULL)
            })

            if (!is.null(tmp)) {
              tmp_df <- as.data.frame(tmp)
              tmp_df$iteration <- i
              tmp_df$ntree <- ntree
              RF_benchmark <- rbind(RF_benchmark, tmp_df)
            }
          }
        }
        print("Iteration finished")


        # Re-train the model using the best tuned hyper-parameters ---
        if (all(c("mtry", "ntree") %in% colnames(RF_benchmark))) {
          # Get benchmark summary
          RF_benchmark <- RF_benchmark %>%
            mutate(Hyperpar_comb = paste0(mtry, "-", ntree)) %>%
            group_by(Hyperpar_comb) %>%
            mutate(median_OOBError = median(OOBError))

          RF_benchmark_summary <- as.data.frame(table(RF_benchmark$Hyperpar_comb), stringsAsFactors = FALSE) %>%
            mutate(Var1 = as.character(Var1)) %>%  # Convert Var1 to integer for matching
            left_join(
              RF_benchmark %>% dplyr::select(Hyperpar_comb, median_OOBError) %>% unique(),
              by = c("Var1" = "Hyperpar_comb")
            )

          # find the best mtry-ntree combo
          RF_summary_parsed <- RF_benchmark_summary %>%
            separate(Var1, into = c("mtry", "ntree"), sep = "-", convert = TRUE) %>%
            mutate(median_OOBError = round(median_OOBError, 7))
          # Step 2: Get minimum OOB error
          min_oob <- min(RF_summary_parsed$median_OOBError)
          # Step 3: Filter for lowest OOB
          best_combos <- RF_summary_parsed %>%
            filter(median_OOBError == min_oob)
          # Step 4: Filter for highest frequency
          max_freq <- max(best_combos$Freq)
          best_combos <- best_combos %>%
            filter(Freq == max_freq)
          # Step 5: Filter for lowest mtry
          min_mtry <- min(best_combos$mtry)
          best_combos <- best_combos %>%
            filter(mtry == min_mtry)
          # Step 6: Filter for lowest ntree
          min_ntree <- min(best_combos$ntree)
          best_combos <- best_combos %>%
            filter(ntree == min_ntree)

          RF_best_tunned_mtry <- best_combos$mtry
          RF_best_tunned_ntree <- best_combos$ntree
          #Validation_Accuracy <- 1 - best_combos$median_OOBError

          # Model retrain
          RF.model <- randomForest(
            as.formula(paste(Dependency_gene, "~ .")),
            data = train_df,
            mtry = RF_best_tunned_mtry,
            ntree = RF_best_tunned_ntree,
            na.action = na.omit
          )

          #RF.model.accuracy <- sum(diag(RF.model$confusion)) / sum(RF.model$confusion)
          #Validation_Accuracy <- RF.model.accuracy
          if (model_type == "Classification") {
            #Validation_Accuracy <- 1 - RF.model$err.rate[nrow(RF.model$err.rate), "OOB"]
            RF_train_pred <- predict(RF.model, type = "response")
            # Kappa and Accuracy
            conf_mat <- confusionMatrix(RF_train_pred, RF.model$y)
            Validation_Accuracy <- conf_mat$overall["Accuracy"]
            Validation_Kappa <- conf_mat$overall["Kappa"]

          } else if (model_type == "Regression") {
            # Mean squared error at final tree
            Validation_RMSE <- RF.model$mse[length(RF.model$mse)]

            # R-squared at final tree
            Validation_Rsq <- RF.model$rsq[length(RF.model$rsq)]

          }

          # Predict on training datasets
          #RF.model.class <- predict(RF.model, train_df)
          #RF.model.class.confusionMatrix <- confusionMatrix(RF.model.class, train_df[[Dependency_gene]])

          if (model_type == "Classification") {
            RF.model.train.prob <- predict(RF.model, train_df, type = "prob")[, 2] # Probabilities for class

          } else if (model_type == "Regression") {
            RF.model.train.prob <- predict(RF.model, train_df)
          }

          # Calculate optimal threshold from AUC
          AUC_evaluation_results <- evaluate_with_optimal_threshold(
            training_pred = RF.model.train.prob,
            train_labels = train_df[[Dependency_gene]],
            positive_class = "1",
            model_type = model_type,
            Finding_Optimal_Threshold = Finding_Optimal_Threshold
          )

          # Calculate and rank feature importance
          feature_importance <- rank_feature_importance_RF(RF.model)
        } else {
          print("mtry or ntree column is missing.")
          AUC_evaluation_results <- NULL
          RF_best_tunned_mtry <- NULL
          RF_best_tunned_ntree <- NULL
          Validation_RMSE <- NULL
          Validation_Rsq <- NULL
          Validation_Accuracy <- 0
          feature_importance <- NULL
        }

        end_time <- Sys.time()

        time_taken <- end_time - start_time

        # Print final result
        final_benchmark_result <- rbind(
          final_benchmark_result,
          data.frame(
            Algorithm = "Random Forest",
            Hyperparameter = "mtry-ntree",
            Tuned_Value = paste0(
              ifelse(is.null(RF_best_tunned_mtry), NA, RF_best_tunned_mtry), "-",
              ifelse(is.null(RF_best_tunned_ntree), NA, RF_best_tunned_ntree)
            ),

            # Classification metrics
            Optimal_Threshold = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "optimal_threshold") else NA,
            Training_Accuracy = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "training_accuracy") else NA,
            Training_Precision = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "training_precision") else NA,
            Training_Recall = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "training_recall") else NA,
            Training_F1 = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "training_F1") else NA,
            Training_Kappa = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "training_Kappa") else NA,
            Training_AccuracyPValue = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "training_accuracyPValue") else NA,
            Training_McnemarPValue = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "training_McnemarPValue") else NA,
            Training_AUROC = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "training_auroc") else NA,
            Validation_Accuracy = if (model_type == "Classification") round(as.numeric(Validation_Accuracy), 5) else NA,
            Validation_Kappa = if (model_type == "Classification") round(as.numeric(Validation_Kappa), 5) else NA,

            # Regression metrics
            Training_RMSE = if (model_type == "Regression") safe_extract(AUC_evaluation_results, "training_rmse") else NA,
            Training_MAE = if (model_type == "Regression") safe_extract(AUC_evaluation_results, "training_mae") else NA,
            Training_R2 = if (model_type == "Regression") safe_extract(AUC_evaluation_results, "training_r2") else NA,
            Validation_RMSE = if (model_type == "Regression") Validation_RMSE else NA,
            Validation_Rsq = if (model_type == "Regression") Validation_Rsq else NA,

            # Shared
            time_taken = round(as.numeric(time_taken, units = "secs"), 10),
            feature_importance = ifelse(is.null(feature_importance), NA, feature_importance)
          )
        )

        saveRDS(RF.model, file = paste0(Dependency_gene, ".RandomForest.rds"))

        print("Benchmarking Random Forest END")

        # End of Benchmarking Random Forest ---

      } else if (MLmodel == "Naïve Bayes") {

        start_time <- Sys.time()

        # Benchmarking Naïve Bayes ---------------------------------------------------------------
        if (model_type == "Classification") {

          print("Benchmarking Naïve Bayes Start")

          # Define the tuning grid
          tune_grid <- expand.grid(
            usekernel = c(TRUE, FALSE),
            fL = seq(0, 2, by = 0.5),  # Laplace smoothing
            adjust = seq(0.5, 3, by = 0.5)  # Adjust parameter
          )

          # Train Naïve Bayes model
          NB.model <- train(as.formula(paste(Dependency_gene, "~ .")),
                            data = train_df,
                            method = "nb",
                            trControl = ctrlspecs,
                            tuneGrid = tune_grid)


          # Extract best hyperparameters
          best_row_index <- as.numeric(rownames(NB.model$bestTune))
          Validation_Accuracy <- NB.model$results[best_row_index, ]$Accuracy
          Validation_Kappa <- NB.model$results[best_row_index, ]$Kappa

          NB_best_tunned_usekernel <- NB.model$bestTune$usekernel # usekernel is either TRUE or FALSE
          NB_best_tunned_fL <- NB.model$bestTune$fL
          NB_best_tunned_adjust <- NB.model$bestTune$adjust

          NB.model.train.prob <- predict(NB.model, train_df, type = "prob")[, 2] # Probabilities for class 1


          AUC_evaluation_results <- evaluate_with_optimal_threshold(
            training_pred = NB.model.train.prob,
            train_labels = train_df[[Dependency_gene]],
            positive_class = "1",
            model_type = model_type,
            Finding_Optimal_Threshold = Finding_Optimal_Threshold
          )

          # Calculate and rank feature importance
          feature_importance <- rank_feature_importance_caret(NB.model)

          end_time <- Sys.time()

          time_taken <- end_time - start_time

          # Write final benchmark result
          final_benchmark_result <- rbind(
            final_benchmark_result,
            data.frame(
              Algorithm = "Naïve Bayes",
              Hyperparameter = "fL-usekernel-adjust",
              Tuned_Value = paste0(NB_best_tunned_fL,"-",NB_best_tunned_usekernel,"-",NB_best_tunned_adjust),

              # Classification metrics
              Optimal_Threshold = if (model_type == "Classification") AUC_evaluation_results$optimal_threshold else NA,
              Training_Accuracy = if (model_type == "Classification") AUC_evaluation_results$training_accuracy else NA,
              Training_Precision = if (model_type == "Classification") AUC_evaluation_results$training_precision else NA,
              Training_Recall = if (model_type == "Classification") AUC_evaluation_results$training_recall else NA,
              Training_F1 = if (model_type == "Classification") AUC_evaluation_results$training_F1 else NA,
              Training_Kappa = if (model_type == "Classification") AUC_evaluation_results$training_Kappa else NA,
              Training_AccuracyPValue = if (model_type == "Classification") AUC_evaluation_results$training_accuracyPValue else NA,
              Training_McnemarPValue = if (model_type == "Classification") AUC_evaluation_results$training_McnemarPValue else NA,
              Training_AUROC = if (model_type == "Classification") AUC_evaluation_results$training_auroc else NA,
              Validation_Accuracy = if (model_type == "Classification") round(as.numeric(Validation_Accuracy), 5) else NA,
              Validation_Kappa = if (model_type == "Classification") round(as.numeric(Validation_Kappa), 5) else NA,

              # Regression metrics
              Training_RMSE = if (model_type == "Regression") AUC_evaluation_results$training_rmse else NA,
              Training_MAE = if (model_type == "Regression") AUC_evaluation_results$training_mae else NA,
              Training_R2 = if (model_type == "Regression") AUC_evaluation_results$training_r2 else NA,
              Validation_RMSE = if (model_type == "Regression") Validation_RMSE else NA,
              Validation_Rsq = if (model_type == "Regression") Validation_Rsq else NA,

              # Shared metrics
              time_taken = round(as.numeric(time_taken, units = "secs"), 10),
              feature_importance = feature_importance
            )
          )

          print("Benchmarking Naïve Bayes END")
          saveRDS(NB.model, file = paste0(Dependency_gene, ".NaiveBayes.rds"))
        } else {
          print("Skip Naïve Bayes for Regression")
        }

        # End of Benchmarking Naïve Bayes ---


      } else if (MLmodel == "SVM") {

        start_time <- Sys.time()

        # Benchmarking SVM ---------------------------------------------------------------
        print("Benchmarking SVM Start")

        tune_and_evaluate_svm <- function(train_df, Dependency_gene,
                                          model_type = c("Classification", "Regression"),
                                          positive_class = "1",
                                          need_probabilities = FALSE,
                                          number = fold, repeats = max_tuning_iteration) {

          library(caret)

          model_type <- match.arg(model_type)

          # ===== 1) Prepare outcome =====
          if (model_type == "Classification") {
            if (!is.factor(train_df[[Dependency_gene]])) {
              train_df[[Dependency_gene]] <- factor(train_df[[Dependency_gene]])
            }
            # Make level names valid and set positive class as reference
            levels(train_df[[Dependency_gene]]) <- make.names(levels(train_df[[Dependency_gene]]))
            pos <- make.names(positive_class)   # "1" -> "X1"
            if (!pos %in% levels(train_df[[Dependency_gene]])) {
              stop("positive_class not found in outcome levels after make.names(): ", pos)
            }
            train_df[[Dependency_gene]] <- relevel(train_df[[Dependency_gene]], ref = pos)
          }

          # ===== 2) trainControl by task =====
          if (model_type == "Classification" && need_probabilities) {
            ctrlspecs <- trainControl(
              method = "repeatedcv", number = number, repeats = repeats,
              classProbs = TRUE,
              summaryFunction = twoClassSummary,   # enables metric = "ROC"
              savePredictions = "final",
              allowParallel = TRUE
            )
            metric_name <- "ROC"
          } else if (model_type == "Classification") {
            ctrlspecs <- trainControl(
              method = "repeatedcv", number = number, repeats = repeats,
              savePredictions = "final",
              allowParallel = TRUE
            )
            metric_name <- "Accuracy"
          } else { # Regression
            ctrlspecs <- trainControl(
              method = "repeatedcv", number = number, repeats = repeats,
              savePredictions = "final",
              allowParallel = TRUE
            )
            metric_name <- "RMSE"
          }

          # ===== 3) Kernels and grids =====
          kernel_methods <- c(
            linear = "svmLinear",
            radial = "svmRadial",
            poly   = "svmPoly"
          )

          results_list <- list()

          for (kernel_name in names(kernel_methods)) {
            if (kernel_name == "linear") {
              tune_grid <- expand.grid(C = 10^seq(-3, 2, by = 1))
            } else if (kernel_name == "radial") {
              tune_grid <- expand.grid(
                C = 10^seq(-3, 2, by = 1),
                sigma = 10^seq(-3, 2, by = 1)
              )
            } else { # poly
              tune_grid <- expand.grid(
                degree = c(2, 3, 4),
                scale  = 10^seq(-3, 2, by = 1),  # gamma equivalent
                C      = 10^seq(-3, 2, by = 1)
              )
            }

            # Add prob.model only when we need probabilities for classification
            extra_args <- list()
            if (model_type == "Classification" && need_probabilities) {
              extra_args$prob.model <- TRUE
            }

            model <- do.call(train, c(list(
              form = as.formula(paste(Dependency_gene, "~ .")),
              data = train_df,
              method = kernel_methods[[kernel_name]],
              trControl = ctrlspecs,
              tuneGrid  = tune_grid,
              metric    = metric_name
            ), extra_args))

            results_list[[kernel_name]] <- model
          }

          # ===== 4) Select best across kernels =====
          pick_best <- function(m, metric) {
            if (metric == "ROC")      return(max(m$results$ROC))
            if (metric == "Accuracy") return(max(m$results$Accuracy))
            if (metric == "RMSE")     return(-min(m$results$RMSE))  # negate for min
            stop("Unknown metric")
          }
          best_model <- results_list[[ which.max(sapply(results_list, pick_best, metric = metric_name)) ]]

          # ===== 5) CV predictions filtered to bestTune =====
          pred_df <- best_model$pred
          if (!is.null(pred_df) && nrow(pred_df)) {
            for (p in names(best_model$bestTune)) {
              pred_df <- pred_df[pred_df[[p]] == best_model$bestTune[[p]], , drop = FALSE]
            }
          }

          # ===== 6) Metrics and CV outputs =====
          Validation_Accuracy <- NA_real_
          Validation_Kappa    <- NA_real_
          Validation_RMSE     <- NA_real_
          Validation_R2       <- NA_real_
          cv_prob             <- NULL
          cv_labels           <- NULL
          cv_rowIndex         <- NULL
          positive_level      <- if (model_type == "Classification") levels(train_df[[Dependency_gene]])[1] else NA_character_

          if (model_type == "Classification") {
            if (!is.null(pred_df) && nrow(pred_df)) {
              # CV Accuracy/Kappa from out-of-fold predictions
              cm <- confusionMatrix(pred_df$pred, pred_df$obs, positive = positive_level)
              Validation_Accuracy <- unname(cm$overall["Accuracy"])
              Validation_Kappa    <- unname(cm$overall["Kappa"])

              # If probs requested, extract the positive-class prob column + labels + row indices
              if (need_probabilities) {
                stopifnot(positive_level %in% colnames(pred_df))
                cv_prob     <- pred_df[[positive_level]]
                cv_labels   <- pred_df$obs
                cv_rowIndex <- pred_df$rowIndex
              }
            }
          } else {
            # Regression metrics from best-tune row
            res <- best_model$results
            idx <- rep(TRUE, nrow(res))
            for (p in names(best_model$bestTune)) idx <- idx & res[[p]] == best_model$bestTune[[p]]
            Validation_RMSE <- res$RMSE[idx][1]
            Validation_R2   <- res$Rsquared[idx][1]
          }

          # Readable tuned value string
          tuned_value <- paste(best_model$method, paste(unlist(best_model$bestTune), collapse = "-"), sep = "-")

          return(list(
            model = best_model,
            best_kernel = best_model$method,
            best_params = best_model$bestTune,
            tuned_value = tuned_value,
            Validation_Accuracy = Validation_Accuracy,
            Validation_Kappa = Validation_Kappa,
            Validation_RMSE = Validation_RMSE,
            Validation_R2 = Validation_R2,
            cv_prob = cv_prob,                 # NULL unless need_probabilities = TRUE
            cv_labels = cv_labels,             # factor labels aligned to cv_prob
            cv_rowIndex = cv_rowIndex,         # row indices back to train_df
            positive_level = positive_level    # e.g., "X1"
          ))
        }


        if (model_type == "Classification") {
          # Triggers the function
          result <- tune_and_evaluate_svm(
            train_df = train_df,
            Dependency_gene = Dependency_gene,
            model_type = model_type,
            positive_class = "1",
            need_probabilities = TRUE
          )

          # Best Accuracy from results
          Validation_Accuracy <- result$Validation_Accuracy
          Validation_Kappa <- result$Validation_Kappa

          print(result$cv_prob)
          print(result$cv_labels)

        } else if (model_type == "Regression") {
          result <- tune_and_evaluate_svm(
            train_df = train_df,
            Dependency_gene = Dependency_gene,
            model_type = model_type
          )

          # Extract RMSE and Rsquared at best tuning parameter
          Validation_RMSE <- result$Validation_RMSE
          Validation_Rsq <- result$Validation_R2
        }

        AUC_evaluation_results <- evaluate_with_optimal_threshold(
          training_pred = result$cv_prob,
          train_labels = result$cv_labels,
          positive_class = "X1",
          negative_class = "X0",
          model_type = model_type,
          Finding_Optimal_Threshold = Finding_Optimal_Threshold
        )

        # Calculate and rank feature importance
        feature_importance <- rank_feature_importance_caret(result$model)
        # Calculate and rank feature importance
        #feature_importance <- rank_feature_importance_e1071(result$model, train_df, Dependency_gene)

        end_time <- Sys.time()

        time_taken <- end_time - start_time

        # Write final benchmark result
        # Store results

        # Append to benchmark
        final_benchmark_result <- rbind(
          final_benchmark_result,
          data.frame(
            Algorithm = "SVM",
            Hyperparameter = "kernel-cost-gamma",
            Tuned_Value = result$tuned_value,


            # Classification metrics
            Optimal_Threshold = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "optimal_threshold") else NA,
            Training_Accuracy = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "training_accuracy") else NA,
            Training_Precision = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "training_precision") else NA,
            Training_Recall = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "training_recall") else NA,
            Training_F1 = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "training_F1") else NA,
            Training_Kappa = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "training_Kappa") else NA,
            Training_AccuracyPValue = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "training_accuracyPValue") else NA,
            Training_McnemarPValue = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "training_McnemarPValue") else NA,
            Training_AUROC = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "training_auroc") else NA,
            Validation_Accuracy = if (model_type == "Classification") round(as.numeric(Validation_Accuracy), 5) else NA,
            Validation_Kappa = if (model_type == "Classification") round(as.numeric(Validation_Kappa), 5) else NA,

            # Regression metrics
            Training_RMSE = if (model_type == "Regression") safe_extract(AUC_evaluation_results, "training_rmse") else NA,
            Training_MAE = if (model_type == "Regression") safe_extract(AUC_evaluation_results, "training_mae") else NA,
            Training_R2 = if (model_type == "Regression") safe_extract(AUC_evaluation_results, "training_r2") else NA,
            Validation_RMSE = if (model_type == "Regression") Validation_RMSE else NA,
            Validation_Rsq = if (model_type == "Regression") Validation_Rsq else NA,

            # Shared metrics
            time_taken = round(as.numeric(time_taken, units = "secs"), 10),
            feature_importance = feature_importance,
            stringsAsFactors = FALSE
          )
        )

        saveRDS(result$model, file = paste0(Dependency_gene, ".SVM.rds"))

        print("Benchmarking SVM END")
        # End of Benchmarking SVM ---


      } else if (MLmodel == "Elastic Net") {

        start_time <- Sys.time()

        # Benchmarking Elastic Net ---------------------------------------------------------------
        print("Benchmarking Elastic Net Start")
        set.seed(101)
        alpha_vector <- seq(0, 1, length=10)
        lambda_vector <- 10^seq(5, -5, length=100)
        tune_grid <- expand.grid(alpha = alpha_vector, lambda = lambda_vector)

        # Train Elastic Net model
        family_type <- if (model_type == "Classification") "binomial" else "gaussian"

        ECN.model <- train(
          as.formula(paste(Dependency_gene, "~ .")),
          data = train_df,
          preProcess = c("center", "scale"),
          method = "glmnet",
          tuneGrid = tune_grid,
          trControl = ctrlspecs,
          family = family_type
        )


        # Extract best hyperparameters

        if (model_type == "Classification") {
          # Best Accuracy from results
          best_row_index <- as.numeric(rownames(ECN.model$bestTune))
          Validation_Accuracy <- ECN.model$results[best_row_index, "Accuracy"]
          Validation_Kappa <- ECN.model$results[best_row_index, "Kappa"]

          # Probabilities for class 1
          ECN.model.train.prob <- predict(ECN.model, train_df, type = "prob")[, 2]

        } else if (model_type == "Regression") {
          # Extract RMSE and Rsquared at best tuning parameter
          best_row_index <- as.numeric(rownames(ECN.model$bestTune))
          Validation_RMSE <- ECN.model$results[best_row_index, "RMSE"]
          Validation_Rsq <- ECN.model$results[best_row_index, "Rsquared"]

          # Predicted values (continuous)
          ECN.model.train.prob <- predict(ECN.model, train_df)
        }


        AUC_evaluation_results <- evaluate_with_optimal_threshold(
          training_pred = ECN.model.train.prob,
          train_labels = train_df[[Dependency_gene]],
          positive_class = "1",
          model_type = model_type,
          Finding_Optimal_Threshold = Finding_Optimal_Threshold
        )

        # Calculate and rank feature importance
        feature_importance <- rank_feature_importance_caret(ECN.model)

        end_time <- Sys.time()

        time_taken <- end_time - start_time

        # Write final benchmark result
        final_benchmark_result <- rbind(
          final_benchmark_result,
          data.frame(
            Algorithm = "Elastic-Net",
            Hyperparameter = "alpha-lamda",
            Tuned_Value = paste0(ECN.model$bestTune$alpha,"-",ECN.model$bestTune$lambda),

            # Classification metrics
            Optimal_Threshold = if (model_type == "Classification") AUC_evaluation_results$optimal_threshold else NA,
            Training_Accuracy = if (model_type == "Classification") AUC_evaluation_results$training_accuracy else NA,
            Training_Precision = if (model_type == "Classification") AUC_evaluation_results$training_precision else NA,
            Training_Recall = if (model_type == "Classification") AUC_evaluation_results$training_recall else NA,
            Training_F1 = if (model_type == "Classification") AUC_evaluation_results$training_F1 else NA,
            Training_Kappa = if (model_type == "Classification") AUC_evaluation_results$training_Kappa else NA,
            Training_AccuracyPValue = if (model_type == "Classification") AUC_evaluation_results$training_accuracyPValue else NA,
            Training_McnemarPValue = if (model_type == "Classification") AUC_evaluation_results$training_McnemarPValue else NA,
            Training_AUROC = if (model_type == "Classification") AUC_evaluation_results$training_auroc else NA,
            Validation_Accuracy = if (model_type == "Classification") round(as.numeric(Validation_Accuracy), 5) else NA,
            Validation_Kappa = if (model_type == "Classification") round(as.numeric(Validation_Kappa), 5) else NA,

            # Regression metrics
            Training_RMSE = if (model_type == "Regression") AUC_evaluation_results$training_rmse else NA,
            Training_MAE = if (model_type == "Regression") AUC_evaluation_results$training_mae else NA,
            Training_R2 = if (model_type == "Regression") AUC_evaluation_results$training_r2 else NA,
            Validation_RMSE = if (model_type == "Regression") Validation_RMSE else NA,
            Validation_Rsq = if (model_type == "Regression") Validation_Rsq else NA,

            # Shared metrics
            time_taken = round(as.numeric(time_taken, units = "secs"), 10),
            feature_importance = feature_importance
          )
        )

        saveRDS(ECN.model, file = paste0(Dependency_gene, ".ElasticNet.rds"))

        print("Benchmarking ECN END")
        # End of Benchmarking ECN ---


      } else if (MLmodel == "KNN") {

        start_time <- Sys.time()

        # Benchmarking KNN ---------------------------------------------------------------
        print("Benchmarking KNN Start")

        metric <- if (model_type == "Classification") "Accuracy" else "RMSE"

        grid <- expand.grid(.k=seq(1,50,by=1))

        KNN.model <- train(as.formula(paste(Dependency_gene, "~ .")),
                           data= train_df,
                           method="knn",
                           metric=metric,
                           tuneGrid=grid,
                           trControl=ctrlspecs,
                           na.action = na.omit)

        if (model_type == "Classification") {
          # Best Accuracy from results
          best_row_index <- as.numeric(rownames(KNN.model$bestTune))
          Validation_Accuracy <- KNN.model$results[best_row_index, "Accuracy"]
          Validation_Kappa <- KNN.model$results[best_row_index, "Kappa"]

          # Probabilities for class 1
          KNN.model.train.prob <- predict(KNN.model, train_df, type = "prob")[, 2]

        } else if (model_type == "Regression") {
          # Extract RMSE and Rsquared at best tuning parameter
          best_row_index <- as.numeric(rownames(KNN.model$bestTune))
          Validation_RMSE <- KNN.model$results[best_row_index, "RMSE"]
          Validation_Rsq <- KNN.model$results[best_row_index, "Rsquared"]

          # Predicted values (continuous)
          KNN.model.train.prob <- predict(KNN.model, train_df)
        }


        AUC_evaluation_results <- evaluate_with_optimal_threshold(
          training_pred = KNN.model.train.prob,
          train_labels = train_df[[Dependency_gene]],
          positive_class = "1",
          model_type = model_type,
          Finding_Optimal_Threshold = Finding_Optimal_Threshold
        )

        # Calculate and rank feature importance
        feature_importance <- rank_feature_importance_caret(KNN.model)

        end_time <- Sys.time()

        time_taken <- end_time - start_time

        # Write final benchmark result
        final_benchmark_result <- rbind(
          final_benchmark_result,
          data.frame(
            Algorithm = "KNN",
            Hyperparameter = "k",
            Tuned_Value = paste0(KNN.model$bestTune$k),

            # Classification metrics
            Optimal_Threshold = if (model_type == "Classification") AUC_evaluation_results$optimal_threshold else NA,
            Training_Accuracy = if (model_type == "Classification") AUC_evaluation_results$training_accuracy else NA,
            Training_Precision = if (model_type == "Classification") AUC_evaluation_results$training_precision else NA,
            Training_Recall = if (model_type == "Classification") AUC_evaluation_results$training_recall else NA,
            Training_F1 = if (model_type == "Classification") AUC_evaluation_results$training_F1 else NA,
            Training_Kappa = if (model_type == "Classification") AUC_evaluation_results$training_Kappa else NA,
            Training_AccuracyPValue = if (model_type == "Classification") AUC_evaluation_results$training_accuracyPValue else NA,
            Training_McnemarPValue = if (model_type == "Classification") AUC_evaluation_results$training_McnemarPValue else NA,
            Training_AUROC = if (model_type == "Classification") AUC_evaluation_results$training_auroc else NA,
            Validation_Accuracy = if (model_type == "Classification") round(as.numeric(Validation_Accuracy), 5) else NA,
            Validation_Kappa = if (model_type == "Classification") round(as.numeric(Validation_Kappa), 5) else NA,

            # Regression metrics
            Training_RMSE = if (model_type == "Regression") AUC_evaluation_results$training_rmse else NA,
            Training_MAE = if (model_type == "Regression") AUC_evaluation_results$training_mae else NA,
            Training_R2 = if (model_type == "Regression") AUC_evaluation_results$training_r2 else NA,
            Validation_RMSE = if (model_type == "Regression") Validation_RMSE else NA,
            Validation_Rsq = if (model_type == "Regression") Validation_Rsq else NA,

            # Shared metrics
            time_taken = round(as.numeric(time_taken, units = "secs"), 10),
            feature_importance = feature_importance
          )
        )

        saveRDS(KNN.model, file = paste0(Dependency_gene, ".KNN.rds"))

        print("Benchmarking KNN END")
        # End of Benchmarking KNN ---


      } else if (MLmodel == "Neural Network") {

        start_time <- Sys.time()

        # Benchmarking Neural Network ---------------------------------------------------------------
        print("Benchmarking Neural Network Start")

        linout <- if (model_type == "Classification") FALSE else TRUE

        grid_tune <- expand.grid(
          size = c(1:10),       # Number of hidden neurons
          decay = c(0.001, 0.01, 0.1)  # Regularization values
        )

        # Train Neural Network model
        NeurNet.model <- tryCatch({
          train(as.formula(paste(Dependency_gene, "~ .")),
                data = train_df,
                method = "nnet",
                trControl = ctrlspecs,
                tuneGrid = grid_tune,
                linout = linout,          # linout = TRUE for regression
                na.action = na.omit,
                trace = FALSE)
        }, error = function(e) {
          message("Model training failed: ", e$message)
          return(NULL)
        })


        if (!is.null(NeurNet.model)) {
          if (model_type == "Classification") {
            # Best Accuracy from results
            best_row_index <- as.numeric(rownames(NeurNet.model$bestTune))
            Validation_Accuracy <- NeurNet.model$results[best_row_index, "Accuracy"]
            Validation_Accuracy <- NeurNet.model$results[best_row_index, "Kappa"]

            # Probabilities for class 1
            NeurNet.model.train.prob <- predict(NeurNet.model, train_df, type = "prob")[, 2]
          } else if (model_type == "Regression") {
            # Extract RMSE and Rsquared at best tuning parameter
            best_row_index <- as.numeric(rownames(NeurNet.model$bestTune))
            Validation_RMSE <- NeurNet.model$results[best_row_index, "RMSE"]
            Validation_Rsq <- NeurNet.model$results[best_row_index, "Rsquared"]

            # Predicted values (continuous)
            NeurNet.model.train.prob <- predict(NeurNet.model, train_df)
          }


          AUC_evaluation_results <- evaluate_with_optimal_threshold(
            training_pred = NeurNet.model.train.prob,
            train_labels = train_df[[Dependency_gene]],
            positive_class = "1",
            model_type = model_type,
            Finding_Optimal_Threshold = Finding_Optimal_Threshold
          )

          # Calculate and rank feature importance
          feature_importance <- rank_feature_importance_caret(NeurNet.model)

          end_time <- Sys.time()

          time_taken <- end_time - start_time

          # Write final benchmark result
          final_benchmark_result <- rbind(
            final_benchmark_result,
            data.frame(
              Algorithm = "Neural Network",
              Hyperparameter = "size-decay",
              Tuned_Value = paste0(NeurNet.model$bestTune$size,"-",NeurNet.model$bestTune$decay),

              # Classification metrics
              Optimal_Threshold = if (model_type == "Classification") AUC_evaluation_results$optimal_threshold else NA,
              Training_Accuracy = if (model_type == "Classification") AUC_evaluation_results$training_accuracy else NA,
              Training_Precision = if (model_type == "Classification") AUC_evaluation_results$training_precision else NA,
              Training_Recall = if (model_type == "Classification") AUC_evaluation_results$training_recall else NA,
              Training_F1 = if (model_type == "Classification") AUC_evaluation_results$training_F1 else NA,
              Training_Kappa = if (model_type == "Classification") AUC_evaluation_results$training_Kappa else NA,
              Training_AccuracyPValue = if (model_type == "Classification") AUC_evaluation_results$training_accuracyPValue else NA,
              Training_McnemarPValue = if (model_type == "Classification") AUC_evaluation_results$training_McnemarPValue else NA,
              Training_AUROC = if (model_type == "Classification") AUC_evaluation_results$training_auroc else NA,
              Validation_Accuracy = if (model_type == "Classification") round(as.numeric(Validation_Accuracy), 5) else NA,
              Validation_Kappa = if (model_type == "Classification") round(as.numeric(Validation_Kappa), 5) else NA,

              # Regression metrics
              Training_RMSE = if (model_type == "Regression") AUC_evaluation_results$training_rmse else NA,
              Training_MAE = if (model_type == "Regression") AUC_evaluation_results$training_mae else NA,
              Training_R2 = if (model_type == "Regression") AUC_evaluation_results$training_r2 else NA,
              Validation_RMSE = if (model_type == "Regression") Validation_RMSE else NA,
              Validation_Rsq = if (model_type == "Regression") Validation_Rsq else NA,

              # Shared metrics
              time_taken = round(as.numeric(time_taken, units = "secs"), 10),
              feature_importance = feature_importance
            )
          )

          saveRDS(NeurNet.model, file = paste0(Dependency_gene, ".NeuroNetwork.rds"))

        } else {
          warning("Neural network model did not train successfully.")
          final_benchmark_result <- rbind(
            final_benchmark_result,
            data.frame(
              Algorithm = "Neural Network",
              Hyperparameter = "size-decay",
              Tuned_Value = "Neural network model did not train successfully",
              # Classification metrics
              Optimal_Threshold = NA,
              Training_Accuracy = NA,
              Training_Precision = NA,
              Training_Recall = NA,
              Training_F1 = NA,
              Training_Kappa = NA,
              Training_AccuracyPValue = NA,
              Training_McnemarPValue = NA,
              Training_AUROC = NA,
              Validation_Accuracy = NA,
              Validation_Kappa = NA,
              # Regression metrics
              Training_RMSE = NA,
              Training_MAE = NA,
              Training_R2 = NA,
              Validation_RMSE = NA,
              Validation_Rsq = NA,
              # Shared metrics
              time_taken = NA,
              feature_importance = NA
            )
          )
        }

        print("Benchmarking Neural Network END")
        # End of Benchmarking Neural Network ---


      } else if (MLmodel == "AdaBoost") {

        start_time <- Sys.time()

        # Benchmarking AdaBoost ---------------------------------------------------------------
        print("Benchmarking AdaBoost Start")

        grid_tune <- expand.grid(
          mfinal = seq(50, 150, by = 10),
          maxdepth = c(1,2,3,4),
          coeflearn = c("Breiman", "Freund", "Zhu")
        )

        # Train the Adaboost model
        AdaBoost.model <- train(as.formula(paste(Dependency_gene, "~ .")),
                                data = train_df,
                                method = "AdaBoost.M1",
                                tuneGrid = grid_tune,
                                trControl = ctrlspecs)

        if (model_type == "Classification") {
          # Best Accuracy from results
          best_row_index <- as.numeric(rownames(AdaBoost.model$bestTune))
          Validation_Accuracy <- AdaBoost.model$results[best_row_index, "Accuracy"]
          Validation_Kappa <- AdaBoost.model$results[best_row_index, "Kappa"]

          # Probabilities for class 1
          AdaBoost.model.train.prob <- predict(AdaBoost.model, train_df, type = "prob")[, 2]

        } else if (model_type == "Regression") {
          # Extract RMSE and Rsquared at best tuning parameter
          best_row_index <- as.numeric(rownames(AdaBoost.model$bestTune))
          Validation_RMSE <- AdaBoost.model$results[best_row_index, "RMSE"]
          Validation_Rsq <- AdaBoost.model$results[best_row_index, "Rsquared"]

          # Predicted values (continuous)
          AdaBoost.model.train.prob <- predict(AdaBoost.model, train_df)
        }


        AUC_evaluation_results <- evaluate_with_optimal_threshold(
          training_pred = AdaBoost.model.train.prob,
          train_labels = train_df[[Dependency_gene]],
          positive_class = "1",
          model_type = model_type,
          Finding_Optimal_Threshold = Finding_Optimal_Threshold
        )

        # Calculate and rank feature importance
        feature_importance <- rank_feature_importance_caret(AdaBoost.model)


        end_time <- Sys.time()

        time_taken <- end_time - start_time

        # Write final benchmark result
        final_benchmark_result <- rbind(
          final_benchmark_result,
          data.frame(
            Algorithm = "AdaBoost",
            Hyperparameter = "coeflearn-maxdepth-mfinal",
            Tuned_Value = paste0(AdaBoost.model$bestTune$coeflearn,"-",AdaBoost.model$bestTune$maxdepth,"-",AdaBoost.model$bestTune$mfinal),

            # Classification metrics
            Optimal_Threshold = if (model_type == "Classification") AUC_evaluation_results$optimal_threshold else NA,
            Training_Accuracy = if (model_type == "Classification") AUC_evaluation_results$training_accuracy else NA,
            Training_Precision = if (model_type == "Classification") AUC_evaluation_results$training_precision else NA,
            Training_Recall = if (model_type == "Classification") AUC_evaluation_results$training_recall else NA,
            Training_F1 = if (model_type == "Classification") AUC_evaluation_results$training_F1 else NA,
            Training_Kappa = if (model_type == "Classification") AUC_evaluation_results$training_Kappa else NA,
            Training_AccuracyPValue = if (model_type == "Classification") AUC_evaluation_results$training_accuracyPValue else NA,
            Training_McnemarPValue = if (model_type == "Classification") AUC_evaluation_results$training_McnemarPValue else NA,
            Training_AUROC = if (model_type == "Classification") AUC_evaluation_results$training_auroc else NA,
            Validation_Accuracy = if (model_type == "Classification") round(as.numeric(Validation_Accuracy), 5) else NA,
            Validation_Kappa = if (model_type == "Classification") round(as.numeric(Validation_Kappa), 5) else NA,

            # Regression metrics
            Training_RMSE = if (model_type == "Regression") AUC_evaluation_results$training_rmse else NA,
            Training_MAE = if (model_type == "Regression") AUC_evaluation_results$training_mae else NA,
            Training_R2 = if (model_type == "Regression") AUC_evaluation_results$training_r2 else NA,
            Validation_RMSE = if (model_type == "Regression") Validation_RMSE else NA,
            Validation_Rsq = if (model_type == "Regression") Validation_Rsq else NA,

            # Shared metrics
            time_taken = round(as.numeric(time_taken, units = "secs"), 10),
            feature_importance = feature_importance
          )
        )

        saveRDS(AdaBoost.model, file = paste0(Dependency_gene, ".AdaBoost.rds"))

        print("Benchmarking AdaBoost END")
        # End of Benchmarking AdaBoost ---



      } else if (MLmodel == "XGBoost") {

        start_time <- Sys.time()

        # Benchmarking XGBoost ---------------------------------------------------------------
        print("Benchmarking XGBoost start")
        tune_and_evaluate_xgboost <- function(train_df,
                                              Dependency_gene,
                                              model_type = c("Classification", "Regression"),
                                              XBoost_tuning_grid = "Simple",
                                              positive_class = 1) {

          library(xgboost)
          library(caret)
          library(dplyr)

          model_type <- match.arg(model_type)

          # Prepare labels
          if (model_type == "Classification") {
            y <- as.numeric(as.character(train_df[[Dependency_gene]]))
            y[!is.na(y) & y != 0] <- 1
            y[is.na(y)] <- 0
            train_df[[Dependency_gene]] <- y
          } else {
            y <- as.numeric(as.character(train_df[[Dependency_gene]]))
            train_df[[Dependency_gene]] <- y
          }

          # Predictors (numeric)
          X_train <- train_df[, !names(train_df) %in% Dependency_gene] %>%
            mutate(across(everything(), ~ as.numeric(as.character(.))))

          dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = train_df[[Dependency_gene]])

          # Hyperparameter grid
          search_grid <- if (XBoost_tuning_grid == "Simple") {
            expand.grid(
              max_depth = c(3, 6),
              eta = c(0.05, 0.3),
              gamma = c(0, 0.5, 1.0),
              colsample_bytree = c(0.6, 1.0),
              min_child_weight = c(1, 3),
              subsample = c(0.75, 1.0)
            )
          } else {
            expand.grid(
              max_depth = c(2, 4, 6),
              eta = c(0.025, 0.05, 0.1, 0.3),
              gamma = c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0),
              colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
              min_child_weight = c(1, 2, 3),
              subsample = c(0.5, 0.75, 1.0)
            )
          }

          best_score <- if (model_type == "Classification") 0 else Inf
          best_params <- list()
          best_nrounds <- 50
          best_cv_pred <- NULL  # store OOF predictions for Kappa

          for (i in 1:nrow(search_grid)) {
            params <- list(
              objective = if (model_type == "Classification") "binary:logistic" else "reg:squarederror",
              eval_metric = if (model_type == "Classification") "error" else "rmse",
              max_depth = search_grid$max_depth[i],
              eta = search_grid$eta[i],
              gamma = search_grid$gamma[i],
              colsample_bytree = search_grid$colsample_bytree[i],
              min_child_weight = search_grid$min_child_weight[i],
              subsample = search_grid$subsample[i]
            )

            cv_results <- xgb.cv(
              params = params,
              data = dtrain,
              nfold = 10,
              nrounds = 100,
              early_stopping_rounds = 20,
              stratified = (model_type == "Classification"),
              verbose = 0,
              prediction = TRUE   # <-- get OOF predictions
            )

            if (!is.null(cv_results)) {
              best_iter <- cv_results$best_iteration
              score <- if (model_type == "Classification") {
                1 - cv_results$evaluation_log$test_error_mean[best_iter]
              } else {
                cv_results$evaluation_log$test_rmse_mean[best_iter]
              }

              improve <- if (model_type == "Classification") score > best_score else score < best_score
              if (!is.na(score) && improve) {
                best_score <- score
                best_params <- params
                best_nrounds <- best_iter
                # xgb.cv$pred are OOF probabilities at the final iteration used
                best_cv_pred <- cv_results$pred
              }
            }
          }

          # Train final model on all data with best params
          final_model <- xgb.train(
            params = best_params,
            data = dtrain,
            nrounds = best_nrounds,
            verbose = 0
          )

          if (model_type == "Classification") {
            # Training predictions (for reference)
            train_prob <- predict(final_model, dtrain)

            # Use OOF preds from best CV (best_cv_pred) to compute Validation Kappa by optimal threshold
            # Fallback to 0.5 if your helper doesn't return a threshold
            print(train_df[[Dependency_gene]])

            print("Maybe there")

            val_eval <- evaluate_with_optimal_threshold(
              training_pred = best_cv_pred,
              train_labels = train_df[[Dependency_gene]],
              model_type = model_type,
              positive_class = "1",
              negative_class = "0",
              Finding_Optimal_Threshold = TRUE
            )

            thr <- val_eval$optimal_threshold %||% val_eval$best_threshold %||% 0.5
            val_pred <- ifelse(best_cv_pred >= thr, 1, 0)

            print("Helloe there")

            cm <- caret::confusionMatrix(
              factor(val_pred, levels = c(0, 1)),
              factor(train_df[[Dependency_gene]], levels = c(0, 1)),
              positive = as.character(positive_class)
            )

            Validation_Accuracy <- unname(cm$overall["Accuracy"])
            Validation_Kappa    <- unname(cm$overall["Kappa"])

            Validation_Metric <- best_score  # CV accuracy from xgb.cv

            AUC_evaluation_results <- modifyList(val_eval, list(
              validation_accuracy = Validation_Accuracy,
              validation_kappa = Validation_Kappa
            ))

          } else {
            # Regression metrics
            train_pred <- predict(final_model, dtrain)
            rmse_train <- sqrt(mean((train_df[[Dependency_gene]] - train_pred)^2))
            mae_train  <- mean(abs(train_df[[Dependency_gene]] - train_pred))
            r2_train   <- cor(train_df[[Dependency_gene]], train_pred)^2

            Validation_Metric <- best_score  # CV RMSE from xgb.cv

            AUC_evaluation_results <- list(
              training_rmse = rmse_train,
              training_mae = mae_train,
              training_r2  = r2_train,
              validation_rmse = best_score,
              # pseudo R2 from CV RMSE (uses var(y) from all data)
              validation_r2 = 1 - (best_score^2 / var(train_df[[Dependency_gene]]))
            )

            Validation_Kappa <- NA
          }

          feature_importance <- rank_feature_importance_xgboost(final_model)

          param_string <- paste0(best_params$max_depth, "-", best_params$eta, "-", best_params$gamma, "-",
                                 best_params$colsample_bytree, "-", best_params$min_child_weight, "-",
                                 best_params$subsample, "-", best_nrounds)

          return(list(
            model = final_model,
            evaluation = AUC_evaluation_results,
            validation_metric = Validation_Metric,
            validation_kappa = if (model_type == "Classification") Validation_Kappa else NA,
            tuned_value = param_string,
            feature_importance = feature_importance
          ))
        }



        # triggers the function ---
        result <- tune_and_evaluate_xgboost(
          train_df = train_df,
          Dependency_gene = Dependency_gene,
          model_type = model_type,
          XBoost_tuning_grid = XBoost_tuning_grid
        )


        end_time <- Sys.time()

        time_taken <- end_time - start_time

        AUC_evaluation_results <- result$evaluation

        # Extract validation metrics from result if regression
        Validation_RMSE <- if (model_type == "Regression") AUC_evaluation_results$validation_rmse else NA
        Validation_Rsq  <- if (model_type == "Regression") AUC_evaluation_results$validation_r2  else NA
        feature_importance <- result$feature_importance

        # Append to benchmark
        final_benchmark_result <- rbind(
          final_benchmark_result,
          data.frame(
            Algorithm = "XGBoost",
            Hyperparameter = "max_depth-eta-gamma-colsample_bytree-min_child_weight-subsample-nrounds",
            Tuned_Value = result$tuned_value,

            # Classification metrics
            Optimal_Threshold       = if (model_type == "Classification") AUC_evaluation_results$optimal_threshold else NA,
            Training_Accuracy       = if (model_type == "Classification") AUC_evaluation_results$training_accuracy else NA,
            Training_Precision      = if (model_type == "Classification") AUC_evaluation_results$training_precision else NA,
            Training_Recall         = if (model_type == "Classification") AUC_evaluation_results$training_recall else NA,
            Training_F1             = if (model_type == "Classification") AUC_evaluation_results$training_F1 else NA,
            Training_Kappa          = if (model_type == "Classification") AUC_evaluation_results$training_Kappa else NA,
            Training_AccuracyPValue = if (model_type == "Classification") AUC_evaluation_results$training_accuracyPValue else NA,
            Training_McnemarPValue  = if (model_type == "Classification") AUC_evaluation_results$training_McnemarPValue else NA,
            Training_AUROC          = if (model_type == "Classification") AUC_evaluation_results$training_auroc else NA,
            Validation_Accuracy     = if (model_type == "Classification") round(as.numeric(result$validation_metric), 5) else NA,
            Validation_Kappa     = if (model_type == "Classification") round(as.numeric(result$validation_kappa), 5) else NA,

            # Regression metrics
            Training_RMSE     = if (model_type == "Regression") AUC_evaluation_results$training_rmse else NA,
            Training_MAE      = if (model_type == "Regression") AUC_evaluation_results$training_mae  else NA,
            Training_R2       = if (model_type == "Regression") AUC_evaluation_results$training_r2   else NA,
            Validation_RMSE   = Validation_RMSE,
            Validation_Rsq    = Validation_Rsq,

            # Shared metrics
            time_taken = round(as.numeric(time_taken, units = "secs"), 10),
            feature_importance = feature_importance
          )
        )

        saveRDS(result$model, file = paste0(Dependency_gene, ".XGBoost.rds"))
        xgb.save(result$model, paste0(Dependency_gene, ".XGBoost.model")) # Not .rds


        print("Benchmarking XGBoost END")
        # End of Benchmarking XGBoost ---


      } else if (MLmodel == "Decision Tree") {

        start_time <- Sys.time()

        # Benchmarking Decision Tree ---------------------------------------------------------------
        print("Benchmarking Decision Tree Start")

        grid_tune <- expand.grid(
          cp = seq(0.01, 0.1, 0.01)
        )

        Decision_Tree.model <- train(
          as.formula(paste(Dependency_gene, "~ .")),
          data = train_df,
          method = "rpart",
          trControl = ctrlspecs,
          tuneGrid = grid_tune)

        if (model_type == "Classification") {
          # Best Accuracy from results
          best_row_index <- as.numeric(rownames(Decision_Tree.model$bestTune))
          Validation_Accuracy <- Decision_Tree.model$results[best_row_index, "Accuracy"]
          Validation_Kappa <- Decision_Tree.model$results[best_row_index, "Kappa"]

          # Probabilities for class 1
          Decision_Tree.model.train.prob <- predict(Decision_Tree.model, train_df, type = "prob")[, 2]
        } else if (model_type == "Regression") {
          # Extract RMSE and Rsquared at best tuning parameter
          best_row_index <- as.numeric(rownames(Decision_Tree.model$bestTune))
          Validation_RMSE <- Decision_Tree.model$results[best_row_index, "RMSE"]
          Validation_Rsq <- Decision_Tree.model$results[best_row_index, "Rsquared"]

          # Predicted values (continuous)
          Decision_Tree.model.train.prob <- predict(Decision_Tree.model, train_df)
        }


        AUC_evaluation_results <- evaluate_with_optimal_threshold(
          training_pred = Decision_Tree.model.train.prob,
          train_labels = train_df[[Dependency_gene]],
          positive_class = "1",
          model_type = model_type,
          Finding_Optimal_Threshold = Finding_Optimal_Threshold
        )

        # Calculate and rank feature importance
        feature_importance <- rank_feature_importance_caret(Decision_Tree.model)

        end_time <- Sys.time()

        time_taken <- end_time - start_time

        # Write final benchmark result
        final_benchmark_result <- rbind(
          final_benchmark_result,
          data.frame(
            Algorithm = "Decision Tree",
            Hyperparameter = "cp",
            Tuned_Value = Decision_Tree.model$bestTune$cp,

            # Classification metrics
            Optimal_Threshold = if (model_type == "Classification") AUC_evaluation_results$optimal_threshold else NA,
            Training_Accuracy = if (model_type == "Classification") AUC_evaluation_results$training_accuracy else NA,
            Training_Precision = if (model_type == "Classification") AUC_evaluation_results$training_precision else NA,
            Training_Recall = if (model_type == "Classification") AUC_evaluation_results$training_recall else NA,
            Training_F1 = if (model_type == "Classification") AUC_evaluation_results$training_F1 else NA,
            Training_Kappa = if (model_type == "Classification") AUC_evaluation_results$training_Kappa else NA,
            Training_AccuracyPValue = if (model_type == "Classification") AUC_evaluation_results$training_accuracyPValue else NA,
            Training_McnemarPValue = if (model_type == "Classification") AUC_evaluation_results$training_McnemarPValue else NA,
            Training_AUROC = if (model_type == "Classification") AUC_evaluation_results$training_auroc else NA,
            Validation_Accuracy = if (model_type == "Classification") round(as.numeric(Validation_Accuracy), 5) else NA,
            Validation_Kappa = if (model_type == "Classification") round(as.numeric(Validation_Kappa), 5) else NA,

            # Regression metrics
            Training_RMSE = if (model_type == "Regression") AUC_evaluation_results$training_rmse else NA,
            Training_MAE = if (model_type == "Regression") AUC_evaluation_results$training_mae else NA,
            Training_R2 = if (model_type == "Regression") AUC_evaluation_results$training_r2 else NA,
            Validation_RMSE = if (model_type == "Regression") Validation_RMSE else NA,
            Validation_Rsq = if (model_type == "Regression") Validation_Rsq else NA,

            # Shared metrics
            time_taken = round(as.numeric(time_taken, units = "secs"), 10),
            feature_importance = feature_importance
          )
        )

        saveRDS(Decision_Tree.model, file = paste0(Dependency_gene, ".DecisionTree.rds"))

        print("Benchmarking Decision Tree END")
        # End of Benchmarking Decision Tree ---


      } else {
        print("Not a avaible model")
      }
    }

    print("Writing out")

    final_benchmark_result <- final_benchmark_result %>%
      mutate(max_tuning_iteration = max_tuning_iteration,
             gene_hits_percentage_cutoff_Lower = cutoff_Lower,
             gene_hits_percentage_cutoff_Upper = cutoff_Upper,
             model_type = model_type)

    #write.csv(final_benchmark_result,
    #          file = final_benchmark_result_write_out_filename,
    #          row.names = F)
    assign("final_benchmark_result", final_benchmark_result, envir = .GlobalEnv)  # Save in global env
    assign("train_df", train_df, envir = .GlobalEnv)  # Save in global env
  }
}
