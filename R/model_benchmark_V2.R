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
model_benchmark_V2 <- function(Features,
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
                            Finding_Optimal_Threshold = TRUE) {

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

  if (!(model_type %in% c("Classification", "Regression"))) {
    stop("Invalid model type: must be 'Classification' or 'Regression'")
  }


  # Setting iteration time
  max_tuning_iteration <- max_tuning_iteration # Defined in the function

  # Setting Global training parameters
  ctrlspecs <- trainControl(method = "repeatedcv", number = fold, repeats = max_tuning_iteration,
                            savePredictions = "all", allowParallel = TRUE) # fold is defined in the function

  # Setting parallel core number
  num_cores <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK"))
  if (is.na(num_cores)) {
    num_cores <- detectCores() - 1  # Use all cores except one
  }

  print(paste0("Will use ", num_cores, " cores for parallel processing"))

  # Setting input data
  merge_data <- Input_Data  # Create a copy of the input data frame
  merge_data[] <- lapply(merge_data, as.numeric)

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
    # Calculate coefficients and normalize
    coefficients <- t(model$coefs) %*% model$SV
    importance <- abs(coefficients)
    importance <- importance / max(importance)

    # Prepare importance dataframe
    importance_df <- data.frame(
      Variable = names(train_df)[colnames(train_df) != Dependency_gene],
      Importance = as.vector(importance)
    )

    # Arrange and format output
    ordered_imp <- importance_df %>%
      arrange(desc(Importance)) %>%
      mutate(comb = paste0(Variable, "_", round(Importance, 4)))

    # Collapse into pipe-separated string
    cg_string <- paste(ordered_imp$comb, collapse = "|")
    return(cg_string)
  }


  # Function to calculate optimal_threshold ---
  evaluate_with_optimal_threshold <- function(training_pred, testing_pred,
                                              train_labels, test_labels,
                                              fallback_conf_matrix = NULL,
                                              positive_class = "1",
                                              model_type,
                                              Finding_Optimal_Threshold) {
    # Convert labels to factor for classification
    if (model_type == "Classification") {
      train_labels <- factor(train_labels)
      test_labels <- factor(test_labels)
    }

    # Initialize outputs
    roc_curve <- NULL
    auroc <- NULL
    threshold_value <- 0.5  # Default
    testing_auroc <- NA
    conf_matrix_train <- NULL
    conf_matrix_test <- NULL

    if (model_type == "Classification") {
      # Check input
      if (is.null(training_pred) || is.null(testing_pred)) {
        stop("Both training and testing predicted probabilities are required.")
      }

      # Compute ROC/threshold from training data
      tryCatch({
        roc_curve <- roc(train_labels, training_pred)
        auroc <- auc(roc_curve)
      }, error = function(e) {
        message("ROC calculation failed: ", e$message)
      })

      tryCatch({
        testing_roc_curve <- roc(test_labels, testing_pred)
        testing_auroc <- auc(testing_roc_curve)
      }, error = function(e) {
        message("Testing ROC calculation failed: ", e$message)
      })

      # Determine threshold
      if (!is.null(roc_curve) & !is.null(auroc)) {
        if (Finding_Optimal_Threshold) {
          threshold_value <- coords(roc_curve, "best", ret = "threshold")[[1]]
        }

        training_preds <- ifelse(training_pred > threshold_value, 1, 0)
        training_preds <- factor(training_preds, levels = levels(train_labels))
        conf_matrix_train <- confusionMatrix(training_preds, train_labels, positive = positive_class)
      } else {
        conf_matrix_train <- if (!is.null(fallback_conf_matrix)) fallback_conf_matrix else list()
        auroc <- -1
      }

      # Testing predictions
      testing_preds <- ifelse(testing_pred > threshold_value, 1, 0)
      testing_preds <- factor(testing_preds, levels = levels(test_labels))
      conf_matrix_test <- confusionMatrix(testing_preds, test_labels, positive = positive_class)

      return(list(
        model_type = model_type,
        roc = roc_curve,
        training_auroc = round(auroc, 3),
        testing_auroc = round(testing_auroc, 3),
        optimal_threshold = round(threshold_value, 3),
        training_accuracy = round(conf_matrix_train$overall["Accuracy"], 3),
        training_accuracyPValue = round(conf_matrix_train$overall["AccuracyPValue"], 3),
        training_precision = round(conf_matrix_train$byClass["Precision"], 3),
        training_recall = round(conf_matrix_train$byClass["Recall"], 3),
        training_F1 = round(conf_matrix_train$byClass["F1"], 3),
        training_Kappa = round(conf_matrix_train$overall["Kappa"], 3),
        training_McnemarPValue = round(conf_matrix_train$overall["McnemarPValue"], 3),
        testing_accuracy = round(conf_matrix_test$overall["Accuracy"], 3),
        testing_accuracyPValue = round(conf_matrix_test$overall["AccuracyPValue"], 3),
        testing_precision = round(conf_matrix_test$byClass["Precision"], 3),
        testing_recall = round(conf_matrix_test$byClass["Recall"], 3),
        testing_F1 = round(conf_matrix_test$byClass["F1"], 3),
        testing_Kappa = round(conf_matrix_test$overall["Kappa"], 3),
        testing_McnemarPValue = round(conf_matrix_test$overall["McnemarPValue"], 3)
      ))

    } else if (model_type == "Regression") {
      # For regression, compute RMSE, MAE, and R^2
      rmse_train <- sqrt(mean((train_labels - training_pred)^2))
      mae_train <- mean(abs(train_labels - training_pred))
      r2_train <- cor(train_labels, training_pred)^2

      rmse_test <- sqrt(mean((test_labels - testing_pred)^2))
      mae_test <- mean(abs(test_labels - testing_pred))
      r2_test <- cor(test_labels, testing_pred)^2

      return(list(
        model_type = model_type,
        training_rmse = round(rmse_train, 3),
        training_mae = round(mae_train, 3),
        training_r2 = round(r2_train, 3),
        testing_rmse = round(rmse_test, 3),
        testing_mae = round(mae_test, 3),
        testing_r2 = round(r2_test, 3)
      ))

    } else {
      stop("Invalid model type: must be 'Classification' or 'Regression'")
    }
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
                                               Prediction_Accuracy = NA,
                                               Prediction_Precision = NA,
                                               Prediction_Recall = NA,
                                               Prediction_F1 = NA,
                                               Prediction_Kappa = NA,
                                               Prediction_AccuracyPValue = NA,
                                               Prediction_McnemarPValue = NA,
                                               Prediction_AUROC = NA,
                                               Validation_Accuracy = NA,

                                               # Regression metrics
                                               Training_RMSE = NA,
                                               Training_MAE = NA,
                                               Training_R2 = NA,
                                               Prediction_RMSE = NA,
                                               Prediction_MAE = NA,
                                               Prediction_R2 = NA,
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

    # Remove column if all betascore are either greater than 0.8 or less than 0.2
    subset_indices <- !apply(merge_data, 2, function(col) {
      all((col > 0.8 | col < 0.2), na.rm = TRUE)
    }) | colnames(merge_data) == Dependency_gene

    merge_data <- merge_data[, subset_indices, drop = FALSE]

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

    # Partitioning the dataframe into training and testing datasets
    index <- createDataPartition(merge_data[[Dependency_gene]], p = 0.8, list = FALSE, times = 1)
    train_df <- merge_data[index, ]
    test_df <- merge_data[-index, ]


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
        index_of_target <- which(colnames(test_df) == Dependency_gene)

        for (i in 1:max_tuning_iteration) {
          for (ntree in ntree_to_try) {
            tmp <- tryCatch({
              tuneRF(
                x = test_df[, -index_of_target],
                y = test_df[, index_of_target],
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
            ntree = RF_best_tunned_ntree
          )

          #RF.model.accuracy <- sum(diag(RF.model$confusion)) / sum(RF.model$confusion)
          #Validation_Accuracy <- RF.model.accuracy
          if (model_type == "Classification") {
            Validation_Accuracy <- 1 - RF.model$err.rate[nrow(RF.model$err.rate), "OOB"]
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
            RF.model.train.prob <- predict(RF.model, train_df, type = "prob")[, 2] # Probabilities for class 1

            # Predict on testing datasets
            RF.model.predict.prob <- predict(RF.model, test_df, type = "prob")[, 2] # Probabilities for class 1

          } else if (model_type == "Regression") {
            RF.model.train.prob <- predict(RF.model, train_df)
            RF.model.predict.prob <- predict(RF.model, test_df)
          }

          # Calculate optimal threshold from AUC
          AUC_evaluation_results <- evaluate_with_optimal_threshold(
            training_pred = RF.model.train.prob,
            testing_pred = RF.model.predict.prob,
            train_labels = train_df[[Dependency_gene]],
            test_labels = test_df[[Dependency_gene]],
            #fallback_conf_matrix = RF.model.class.confusionMatrix, # in case if optimal threshold can not be calculate
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
            Prediction_Accuracy = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "testing_accuracy") else NA,
            Prediction_Precision = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "testing_precision") else NA,
            Prediction_Recall = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "testing_recall") else NA,
            Prediction_F1 = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "testing_F1") else NA,
            Prediction_Kappa = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "testing_Kappa") else NA,
            Prediction_AccuracyPValue = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "testing_accuracyPValue") else NA,
            Prediction_McnemarPValue = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "testing_McnemarPValue") else NA,
            Prediction_AUROC = if (model_type == "Classification") safe_extract(AUC_evaluation_results, "testing_auroc") else NA,
            Validation_Accuracy = if (model_type == "Classification") round(as.numeric(Validation_Accuracy), 2) else NA,

            # Regression metrics
            Training_RMSE = if (model_type == "Regression") safe_extract(AUC_evaluation_results, "training_rmse") else NA,
            Training_MAE = if (model_type == "Regression") safe_extract(AUC_evaluation_results, "training_mae") else NA,
            Training_R2 = if (model_type == "Regression") safe_extract(AUC_evaluation_results, "training_r2") else NA,
            Prediction_RMSE = if (model_type == "Regression") safe_extract(AUC_evaluation_results, "testing_rmse") else NA,
            Prediction_MAE = if (model_type == "Regression") safe_extract(AUC_evaluation_results, "testing_mae") else NA,
            Prediction_R2 = if (model_type == "Regression") safe_extract(AUC_evaluation_results, "testing_r2") else NA,
            Validation_RMSE = if (model_type == "Regression") Validation_RMSE else NA,
            Validation_Rsq = if (model_type == "Regression") Validation_Rsq else NA,

            # Shared
            time_taken = round(as.numeric(time_taken, units = "secs"), 10),
            feature_importance = ifelse(is.null(feature_importance), NA, feature_importance)
          )
        )



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

          NB_best_tunned_usekernel <- NB.model$bestTune$usekernel # usekernel is either TRUE or FALSE
          NB_best_tunned_fL <- NB.model$bestTune$fL
          NB_best_tunned_adjust <- NB.model$bestTune$adjust

          # Predict using the best tuned hyper-parameters
          NB.model.predict <- predict(NB.model, test_df)

          NB.model.train.prob <- predict(NB.model, train_df, type = "prob")[, 2] # Probabilities for class 1
          NB.model.predict.prob <- predict(NB.model, test_df, type = "prob")[, 2] # Probabilities for class 1

          AUC_evaluation_results <- evaluate_with_optimal_threshold(
            training_pred = NB.model.train.prob,
            testing_pred = NB.model.predict.prob,
            train_labels = train_df[[Dependency_gene]],
            test_labels = test_df[[Dependency_gene]],
            fallback_conf_matrix = NB.model.predict.confusionMatrix, # in case if optimal threshold can not be calculate
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
              Prediction_Accuracy = if (model_type == "Classification") AUC_evaluation_results$testing_accuracy else NA,
              Prediction_Precision = if (model_type == "Classification") AUC_evaluation_results$testing_precision else NA,
              Prediction_Recall = if (model_type == "Classification") AUC_evaluation_results$testing_recall else NA,
              Prediction_F1 = if (model_type == "Classification") AUC_evaluation_results$testing_F1 else NA,
              Prediction_Kappa = if (model_type == "Classification") AUC_evaluation_results$testing_Kappa else NA,
              Prediction_AccuracyPValue = if (model_type == "Classification") AUC_evaluation_results$testing_accuracyPValue else NA,
              Prediction_McnemarPValue = if (model_type == "Classification") AUC_evaluation_results$testing_McnemarPValue else NA,
              Prediction_AUROC = if (model_type == "Classification") AUC_evaluation_results$testing_auroc else NA,
              Validation_Accuracy = if (model_type == "Classification") round(as.numeric(Validation_Accuracy), 2) else NA,

              # Regression metrics
              Training_RMSE = if (model_type == "Regression") AUC_evaluation_results$training_rmse else NA,
              Training_MAE = if (model_type == "Regression") AUC_evaluation_results$training_mae else NA,
              Training_R2 = if (model_type == "Regression") AUC_evaluation_results$training_r2 else NA,
              Prediction_RMSE = if (model_type == "Regression") AUC_evaluation_results$testing_rmse else NA,
              Prediction_MAE = if (model_type == "Regression") AUC_evaluation_results$testing_mae else NA,
              Prediction_R2 = if (model_type == "Regression") AUC_evaluation_results$testing_r2 else NA,
              Validation_RMSE = if (model_type == "Regression") Validation_RMSE else NA,
              Validation_Rsq = if (model_type == "Regression") Validation_Rsq else NA,

              # Shared metrics
              time_taken = round(as.numeric(time_taken, units = "secs"), 10),
              feature_importance = feature_importance
            )
          )

          print("Benchmarking Naïve Bayes END")
        } else {
          print("Skip Naïve Bayes for Regression")
        }
        # End of Benchmarking Naïve Bayes ---


      } else if (MLmodel == "SVM") {

        start_time <- Sys.time()

        # Benchmarking SVM ---------------------------------------------------------------
        print("Benchmarking SVM Start")

        tune_and_evaluate_svm <- function(train_df, test_df, Dependency_gene,
                                          model_type = c("Classification", "Regression"),
                                          max_tuning_iteration,
                                          positive_class = "1") {

          model_type <- match.arg(model_type)
          kernel_list <- c("linear", "polynomial", "radial", "sigmoid")
          SVM_benchmark <- data.frame()

          for (i in 1:max_tuning_iteration) {
            iteration_results <- data.frame()

            for (k in kernel_list) {
              try_result <- tryCatch({
                if (k == "linear") {
                  tune_out <- tune(
                    svm,
                    as.formula(paste(Dependency_gene, "~ .")),
                    data = train_df,
                    kernel = k,
                    ranges = list(cost = 10^seq(-3, 2, by = 1)),
                    tunecontrol = tune.control(cross = 10),
                    type = ifelse(model_type == "Classification", "C-classification", "eps-regression"),
                    probability = (model_type == "Classification")
                  )
                } else {
                  tune_out <- tune(
                    svm,
                    as.formula(paste(Dependency_gene, "~ .")),
                    data = train_df,
                    kernel = k,
                    ranges = list(cost = 10^seq(-3, 2, by = 1),
                                  gamma = 10^seq(-3, 2, by = 1)),
                    tunecontrol = tune.control(cross = 10),
                    type = ifelse(model_type == "Classification", "C-classification", "eps-regression"),
                    probability = (model_type == "Classification")
                  )
                }

                result <- data.frame(
                  iteration = i,
                  kernel = k,
                  best_cost = tune_out$best.parameters$cost,
                  best_gamma = if ("gamma" %in% names(tune_out$best.parameters)) tune_out$best.parameters$gamma else NA,
                  best_error = tune_out$best.performance,
                  Validation_RMSE = if (model_type == "Regression") sqrt(tune_out$best.performance) else NA,
                  Validation_R2 = NA
                )
              }, error = function(e) {
                message("Tuning failed for kernel: ", k, " | Error: ", e$message)
                return(NULL)
              })

              if (!is.null(try_result)) {
                iteration_results <- rbind(iteration_results, try_result)
              }
            }

            if (nrow(iteration_results) > 0) {
              best_result <- iteration_results[which.min(iteration_results$best_error), ]
              SVM_benchmark <- rbind(SVM_benchmark, best_result)
            }
          }

          SVM_benchmark <- SVM_benchmark %>%
            mutate(Hyperparam = paste0(kernel, "-", best_cost, "-", best_gamma)) %>%
            group_by(Hyperparam) %>%
            mutate(mean_error_rate = median(best_error)) %>%
            ungroup()

          SVM_benchmark_summary <- as.data.frame(table(SVM_benchmark$Hyperparam)) %>%
            left_join(
              SVM_benchmark %>% dplyr::select(Hyperparam, mean_error_rate, Validation_RMSE) %>% unique(),
              by = c("Var1" = "Hyperparam")
            )

          best_index <- which.max(SVM_benchmark_summary$Freq)
          SVM_best_tunned <- SVM_benchmark_summary$Var1[best_index]
          SVM_best_tunned_kernel <- str_split_i(SVM_best_tunned, "-", 1)
          SVM_best_tunned_cost <- as.numeric(str_split_i(SVM_best_tunned, "-", 2))
          SVM_best_tunned_gamma <- as.numeric(str_split_i(SVM_best_tunned, "-", 3))
          Validation_RMSE <- SVM_benchmark_summary$Validation_RMSE[best_index]

          # Estimate validation R2 based on training set variance
          Validation_R2 <- if (model_type == "Regression") {
            1 - (Validation_RMSE^2 / var(train_df[[Dependency_gene]]))
          } else { NA }

          # Final model
          if (SVM_best_tunned_kernel == "linear") {
            SVM.model <- svm(
              as.formula(paste(Dependency_gene, "~ .")),
              data = train_df,
              type = ifelse(model_type == "Classification", "C-classification", "eps-regression"),
              kernel = SVM_best_tunned_kernel,
              cost = SVM_best_tunned_cost,
              scale = FALSE,
              probability = (model_type == "Classification")
            )
          } else {
            SVM.model <- svm(
              as.formula(paste(Dependency_gene, "~ .")),
              data = train_df,
              type = ifelse(model_type == "Classification", "C-classification", "eps-regression"),
              kernel = SVM_best_tunned_kernel,
              cost = SVM_best_tunned_cost,
              gamma = SVM_best_tunned_gamma,
              scale = FALSE,
              probability = (model_type == "Classification")
            )
          }

          if (model_type == "Classification") {
            train_prob <- attr(predict(SVM.model, train_df, probability = TRUE), "probabilities")[, 2]
            test_prob <- attr(predict(SVM.model, test_df, probability = TRUE), "probabilities")[, 2]
            test_pred <- predict(SVM.model, test_df)

            cm <- confusionMatrix(test_pred, test_df[[Dependency_gene]])

            eval_results <- evaluate_with_optimal_threshold(
              training_pred = train_prob,
              testing_pred = test_prob,
              train_labels = train_df[[Dependency_gene]],
              test_labels = test_df[[Dependency_gene]],
              fallback_conf_matrix = cm,
              positive_class = positive_class,
              model_type = "Classification",
              Finding_Optimal_Threshold = Finding_Optimal_Threshold
            )

          } else {
            train_pred <- predict(SVM.model, train_df)
            test_pred <- predict(SVM.model, test_df)

            rmse_train <- sqrt(mean((train_df[[Dependency_gene]] - train_pred)^2))
            r2_train <- cor(train_df[[Dependency_gene]], train_pred)^2

            rmse_test <- sqrt(mean((test_df[[Dependency_gene]] - test_pred)^2))
            r2_test <- cor(test_df[[Dependency_gene]], test_pred)^2

            mae_train <- mean(abs(train_df[[Dependency_gene]] - train_pred))
            mae_test <- mean(abs(test_df[[Dependency_gene]] - test_pred))


            eval_results <- list(
              training_rmse = rmse_train,
              training_mae = mae_train,
              training_r2 = r2_train,
              testing_rmse = rmse_test,
              testing_mae = mae_test,
              testing_r2 = r2_test,
              validation_rmse = Validation_RMSE,
              validation_r2 = Validation_R2
            )

          }

          return(list(
            best_kernel = SVM_best_tunned_kernel,
            best_cost = SVM_best_tunned_cost,
            best_gamma = SVM_best_tunned_gamma,
            model = SVM.model,
            evaluation = eval_results,
            benchmark = SVM_benchmark_summary
          ))
        }

        # Triggers the function
        result <- tune_and_evaluate_svm(
          train_df = train_df,
          test_df = test_df,
          Dependency_gene = Dependency_gene,
          max_tuning_iteration = max_tuning_iteration,
          model_type = model_type  # or "Regression"
        )



        # Calculate and rank feature importance
        feature_importance <- rank_feature_importance_e1071(result$model, train_df, Dependency_gene)

        end_time <- Sys.time()

        time_taken <- end_time - start_time

        # Write final benchmark result
        # Store results
        AUC_evaluation_results <- result$evaluation

        # Extract validation metrics from result if regression
        Validation_RMSE <- if (model_type == "Regression") AUC_evaluation_results$validation_rmse else NA
        Validation_Rsq  <- if (model_type == "Regression") AUC_evaluation_results$validation_r2  else NA


        # Append to benchmark
        final_benchmark_result <- rbind(
          final_benchmark_result,
          data.frame(
            Algorithm = "SVM",
            Hyperparameter = "kernel-cost-gamma",
            Tuned_Value = paste0(result$best_kernel, "-", result$best_cost, "-", result$best_gamma),

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
            Prediction_Accuracy     = if (model_type == "Classification") AUC_evaluation_results$testing_accuracy else NA,
            Prediction_Precision    = if (model_type == "Classification") AUC_evaluation_results$testing_precision else NA,
            Prediction_Recall       = if (model_type == "Classification") AUC_evaluation_results$testing_recall else NA,
            Prediction_F1           = if (model_type == "Classification") AUC_evaluation_results$testing_F1 else NA,
            Prediction_Kappa        = if (model_type == "Classification") AUC_evaluation_results$testing_Kappa else NA,
            Prediction_AccuracyPValue = if (model_type == "Classification") AUC_evaluation_results$testing_accuracyPValue else NA,
            Prediction_McnemarPValue  = if (model_type == "Classification") AUC_evaluation_results$testing_McnemarPValue else NA,
            Prediction_AUROC        = if (model_type == "Classification") AUC_evaluation_results$testing_auroc else NA,
            Validation_Accuracy     = if (model_type == "Classification") round(as.numeric(Validation_Accuracy), 2) else NA,

            # Regression metrics
            Training_RMSE     = if (model_type == "Regression") AUC_evaluation_results$training_rmse else NA,
            Training_MAE      = if (model_type == "Regression") AUC_evaluation_results$training_mae  else NA,
            Training_R2       = if (model_type == "Regression") AUC_evaluation_results$training_r2   else NA,
            Prediction_RMSE   = if (model_type == "Regression") AUC_evaluation_results$testing_rmse  else NA,
            Prediction_MAE    = if (model_type == "Regression") AUC_evaluation_results$testing_mae   else NA,
            Prediction_R2     = if (model_type == "Regression") AUC_evaluation_results$testing_r2    else NA,
            Validation_RMSE   = Validation_RMSE,
            Validation_Rsq    = Validation_Rsq,

            # Shared metrics
            time_taken = round(as.numeric(time_taken, units = "secs"), 10),
            feature_importance = feature_importance
          )
        )

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

          # Probabilities for class 1
          ECN.model.train.prob <- predict(ECN.model, train_df, type = "prob")[, 2]
          ECN.model.predict.prob <- predict(ECN.model, test_df, type = "prob")[, 2]

        } else if (model_type == "Regression") {
          # Extract RMSE and Rsquared at best tuning parameter
          best_row_index <- as.numeric(rownames(ECN.model$bestTune))
          Validation_RMSE <- ECN.model$results[best_row_index, "RMSE"]
          Validation_Rsq <- ECN.model$results[best_row_index, "Rsquared"]

          # Predicted values (continuous)
          ECN.model.train.prob <- predict(ECN.model, train_df)
          ECN.model.predict.prob <- predict(ECN.model, test_df)
        }


        AUC_evaluation_results <- evaluate_with_optimal_threshold(
          training_pred = ECN.model.train.prob,
          testing_pred = ECN.model.predict.prob,
          train_labels = train_df[[Dependency_gene]],
          test_labels = test_df[[Dependency_gene]],
          #fallback_conf_matrix = ECN.model.predict.confusionMatrix, # in case if optimal threshold can not be calculate
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
            Prediction_Accuracy = if (model_type == "Classification") AUC_evaluation_results$testing_accuracy else NA,
            Prediction_Precision = if (model_type == "Classification") AUC_evaluation_results$testing_precision else NA,
            Prediction_Recall = if (model_type == "Classification") AUC_evaluation_results$testing_recall else NA,
            Prediction_F1 = if (model_type == "Classification") AUC_evaluation_results$testing_F1 else NA,
            Prediction_Kappa = if (model_type == "Classification") AUC_evaluation_results$testing_Kappa else NA,
            Prediction_AccuracyPValue = if (model_type == "Classification") AUC_evaluation_results$testing_accuracyPValue else NA,
            Prediction_McnemarPValue = if (model_type == "Classification") AUC_evaluation_results$testing_McnemarPValue else NA,
            Prediction_AUROC = if (model_type == "Classification") AUC_evaluation_results$testing_auroc else NA,
            Validation_Accuracy = if (model_type == "Classification") round(as.numeric(Validation_Accuracy), 2) else NA,

            # Regression metrics
            Training_RMSE = if (model_type == "Regression") AUC_evaluation_results$training_rmse else NA,
            Training_MAE = if (model_type == "Regression") AUC_evaluation_results$training_mae else NA,
            Training_R2 = if (model_type == "Regression") AUC_evaluation_results$training_r2 else NA,
            Prediction_RMSE = if (model_type == "Regression") AUC_evaluation_results$testing_rmse else NA,
            Prediction_MAE = if (model_type == "Regression") AUC_evaluation_results$testing_mae else NA,
            Prediction_R2 = if (model_type == "Regression") AUC_evaluation_results$testing_r2 else NA,
            Validation_RMSE = if (model_type == "Regression") Validation_RMSE else NA,
            Validation_Rsq = if (model_type == "Regression") Validation_Rsq else NA,

            # Shared metrics
            time_taken = round(as.numeric(time_taken, units = "secs"), 10),
            feature_importance = feature_importance
          )
        )

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

          # Probabilities for class 1
          KNN.model.train.prob <- predict(KNN.model, train_df, type = "prob")[, 2]
          KNN.model.predict.prob <- predict(KNN.model, test_df, type = "prob")[, 2]

        } else if (model_type == "Regression") {
          # Extract RMSE and Rsquared at best tuning parameter
          best_row_index <- as.numeric(rownames(KNN.model$bestTune))
          Validation_RMSE <- KNN.model$results[best_row_index, "RMSE"]
          Validation_Rsq <- KNN.model$results[best_row_index, "Rsquared"]

          # Predicted values (continuous)
          KNN.model.train.prob <- predict(KNN.model, train_df)
          KNN.model.predict.prob <- predict(KNN.model, test_df)
        }


        AUC_evaluation_results <- evaluate_with_optimal_threshold(
          training_pred = KNN.model.train.prob,
          testing_pred = KNN.model.predict.prob,
          train_labels = train_df[[Dependency_gene]],
          test_labels = test_df[[Dependency_gene]],
          #fallback_conf_matrix = KNN.model.predict.confusionMatrix, # in case if optimal threshold can not be calculate
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
            Prediction_Accuracy = if (model_type == "Classification") AUC_evaluation_results$testing_accuracy else NA,
            Prediction_Precision = if (model_type == "Classification") AUC_evaluation_results$testing_precision else NA,
            Prediction_Recall = if (model_type == "Classification") AUC_evaluation_results$testing_recall else NA,
            Prediction_F1 = if (model_type == "Classification") AUC_evaluation_results$testing_F1 else NA,
            Prediction_Kappa = if (model_type == "Classification") AUC_evaluation_results$testing_Kappa else NA,
            Prediction_AccuracyPValue = if (model_type == "Classification") AUC_evaluation_results$testing_accuracyPValue else NA,
            Prediction_McnemarPValue = if (model_type == "Classification") AUC_evaluation_results$testing_McnemarPValue else NA,
            Prediction_AUROC = if (model_type == "Classification") AUC_evaluation_results$testing_auroc else NA,
            Validation_Accuracy = if (model_type == "Classification") round(as.numeric(Validation_Accuracy), 2) else NA,

            # Regression metrics
            Training_RMSE = if (model_type == "Regression") AUC_evaluation_results$training_rmse else NA,
            Training_MAE = if (model_type == "Regression") AUC_evaluation_results$training_mae else NA,
            Training_R2 = if (model_type == "Regression") AUC_evaluation_results$training_r2 else NA,
            Prediction_RMSE = if (model_type == "Regression") AUC_evaluation_results$testing_rmse else NA,
            Prediction_MAE = if (model_type == "Regression") AUC_evaluation_results$testing_mae else NA,
            Prediction_R2 = if (model_type == "Regression") AUC_evaluation_results$testing_r2 else NA,
            Validation_RMSE = if (model_type == "Regression") Validation_RMSE else NA,
            Validation_Rsq = if (model_type == "Regression") Validation_Rsq else NA,

            # Shared metrics
            time_taken = round(as.numeric(time_taken, units = "secs"), 10),
            feature_importance = feature_importance
          )
        )
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
        NeurNet.model <- train(as.formula(paste(Dependency_gene, "~ .")),
                            data = train_df,
                            method = "nnet",
                            trControl = ctrlspecs,
                            tuneGrid = grid_tune,
                            linout = linout,  # Use linout = TRUE for regression
                            na.action = na.omit,
                            trace = FALSE)  # Suppress training output

        if (model_type == "Classification") {
          # Best Accuracy from results
          best_row_index <- as.numeric(rownames(NeurNet.model$bestTune))
          Validation_Accuracy <- NeurNet.model$results[best_row_index, "Accuracy"]

          # Probabilities for class 1
          NeurNet.model.train.prob <- predict(NeurNet.model, train_df, type = "prob")[, 2]
          NeurNet.model.predict.prob <- predict(NeurNet.model, test_df, type = "prob")[, 2]

        } else if (model_type == "Regression") {
          # Extract RMSE and Rsquared at best tuning parameter
          best_row_index <- as.numeric(rownames(NeurNet.model$bestTune))
          Validation_RMSE <- NeurNet.model$results[best_row_index, "RMSE"]
          Validation_Rsq <- NeurNet.model$results[best_row_index, "Rsquared"]

          # Predicted values (continuous)
          NeurNet.model.train.prob <- predict(NeurNet.model, train_df)
          NeurNet.model.predict.prob <- predict(NeurNet.model, test_df)
        }


        AUC_evaluation_results <- evaluate_with_optimal_threshold(
          training_pred = NeurNet.model.train.prob,
          testing_pred = NeurNet.model.predict.prob,
          train_labels = train_df[[Dependency_gene]],
          test_labels = test_df[[Dependency_gene]],
          #fallback_conf_matrix = NeurNet.model.predict.confusionMatrix, # in case if optimal threshold can not be calculate
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
            Prediction_Accuracy = if (model_type == "Classification") AUC_evaluation_results$testing_accuracy else NA,
            Prediction_Precision = if (model_type == "Classification") AUC_evaluation_results$testing_precision else NA,
            Prediction_Recall = if (model_type == "Classification") AUC_evaluation_results$testing_recall else NA,
            Prediction_F1 = if (model_type == "Classification") AUC_evaluation_results$testing_F1 else NA,
            Prediction_Kappa = if (model_type == "Classification") AUC_evaluation_results$testing_Kappa else NA,
            Prediction_AccuracyPValue = if (model_type == "Classification") AUC_evaluation_results$testing_accuracyPValue else NA,
            Prediction_McnemarPValue = if (model_type == "Classification") AUC_evaluation_results$testing_McnemarPValue else NA,
            Prediction_AUROC = if (model_type == "Classification") AUC_evaluation_results$testing_auroc else NA,
            Validation_Accuracy = if (model_type == "Classification") round(as.numeric(Validation_Accuracy), 2) else NA,

            # Regression metrics
            Training_RMSE = if (model_type == "Regression") AUC_evaluation_results$training_rmse else NA,
            Training_MAE = if (model_type == "Regression") AUC_evaluation_results$training_mae else NA,
            Training_R2 = if (model_type == "Regression") AUC_evaluation_results$training_r2 else NA,
            Prediction_RMSE = if (model_type == "Regression") AUC_evaluation_results$testing_rmse else NA,
            Prediction_MAE = if (model_type == "Regression") AUC_evaluation_results$testing_mae else NA,
            Prediction_R2 = if (model_type == "Regression") AUC_evaluation_results$testing_r2 else NA,
            Validation_RMSE = if (model_type == "Regression") Validation_RMSE else NA,
            Validation_Rsq = if (model_type == "Regression") Validation_Rsq else NA,

            # Shared metrics
            time_taken = round(as.numeric(time_taken, units = "secs"), 10),
            feature_importance = feature_importance
          )
        )

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

          # Probabilities for class 1
          AdaBoost.model.train.prob <- predict(AdaBoost.model, train_df, type = "prob")[, 2]
          AdaBoost.model.predict.prob <- predict(AdaBoost.model, test_df, type = "prob")[, 2]

        } else if (model_type == "Regression") {
          # Extract RMSE and Rsquared at best tuning parameter
          best_row_index <- as.numeric(rownames(AdaBoost.model$bestTune))
          Validation_RMSE <- AdaBoost.model$results[best_row_index, "RMSE"]
          Validation_Rsq <- AdaBoost.model$results[best_row_index, "Rsquared"]

          # Predicted values (continuous)
          AdaBoost.model.train.prob <- predict(AdaBoost.model, train_df)
          AdaBoost.model.predict.prob <- predict(AdaBoost.model, test_df)
        }


        AUC_evaluation_results <- evaluate_with_optimal_threshold(
          training_pred = AdaBoost.model.train.prob,
          testing_pred = AdaBoost.model.predict.prob,
          train_labels = train_df[[Dependency_gene]],
          test_labels = test_df[[Dependency_gene]],
          #fallback_conf_matrix = AdaBoost.model.predict.confusionMatrix, # in case if optimal threshold can not be calculate
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
            Prediction_Accuracy = if (model_type == "Classification") AUC_evaluation_results$testing_accuracy else NA,
            Prediction_Precision = if (model_type == "Classification") AUC_evaluation_results$testing_precision else NA,
            Prediction_Recall = if (model_type == "Classification") AUC_evaluation_results$testing_recall else NA,
            Prediction_F1 = if (model_type == "Classification") AUC_evaluation_results$testing_F1 else NA,
            Prediction_Kappa = if (model_type == "Classification") AUC_evaluation_results$testing_Kappa else NA,
            Prediction_AccuracyPValue = if (model_type == "Classification") AUC_evaluation_results$testing_accuracyPValue else NA,
            Prediction_McnemarPValue = if (model_type == "Classification") AUC_evaluation_results$testing_McnemarPValue else NA,
            Prediction_AUROC = if (model_type == "Classification") AUC_evaluation_results$testing_auroc else NA,
            Validation_Accuracy = if (model_type == "Classification") round(as.numeric(Validation_Accuracy), 2) else NA,

            # Regression metrics
            Training_RMSE = if (model_type == "Regression") AUC_evaluation_results$training_rmse else NA,
            Training_MAE = if (model_type == "Regression") AUC_evaluation_results$training_mae else NA,
            Training_R2 = if (model_type == "Regression") AUC_evaluation_results$training_r2 else NA,
            Prediction_RMSE = if (model_type == "Regression") AUC_evaluation_results$testing_rmse else NA,
            Prediction_MAE = if (model_type == "Regression") AUC_evaluation_results$testing_mae else NA,
            Prediction_R2 = if (model_type == "Regression") AUC_evaluation_results$testing_r2 else NA,
            Validation_RMSE = if (model_type == "Regression") Validation_RMSE else NA,
            Validation_Rsq = if (model_type == "Regression") Validation_Rsq else NA,

            # Shared metrics
            time_taken = round(as.numeric(time_taken, units = "secs"), 10),
            feature_importance = feature_importance
          )
        )


        print("Benchmarking AdaBoost END")
        # End of Benchmarking AdaBoost ---



      } else if (MLmodel == "XGBoost") {

        start_time <- Sys.time()

        # Benchmarking XGBoost ---------------------------------------------------------------
        print("Benchmarking XGBoost start")
        tune_and_evaluate_xgboost <- function(train_df, test_df, Dependency_gene,
                                              model_type = c("Classification", "Regression"),
                                              XBoost_tuning_grid = "Simple") {

          model_type <- match.arg(model_type)

          # Convert label for regression or binary classification
          if (model_type == "Classification") {
            train_df[[Dependency_gene]] <- as.numeric(as.character(train_df[[Dependency_gene]]))
            train_df[[Dependency_gene]][train_df[[Dependency_gene]] != 0] <- 1

            test_df[[Dependency_gene]] <- as.numeric(as.character(test_df[[Dependency_gene]]))
            test_df[[Dependency_gene]][test_df[[Dependency_gene]] != 0] <- 1
          }

          dtrain <- xgb.DMatrix(data = as.matrix(train_df[, !names(train_df) %in% Dependency_gene]),
                                label = train_df[[Dependency_gene]])
          dtest <- xgb.DMatrix(data = as.matrix(test_df[, !names(test_df) %in% Dependency_gene]),
                               label = test_df[[Dependency_gene]])

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
              verbose = 0
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
              }
            }
          }

          final_model <- xgb.train(
            params = best_params,
            data = dtrain,
            nrounds = best_nrounds
          )

          if (model_type == "Classification") {
            train_prob <- predict(final_model, dtrain)
            test_prob <- predict(final_model, dtest)

            AUC_evaluation_results <- evaluate_with_optimal_threshold(
              training_pred = train_prob,
              testing_pred = test_prob,
              train_labels = train_df[[Dependency_gene]],
              test_labels = test_df[[Dependency_gene]],
              fallback_conf_matrix = NULL,
              positive_class = "1",
              model_type = model_type,
              Finding_Optimal_Threshold = Finding_Optimal_Threshold
            )
            Validation_Metric <- best_score  # validation accuracy

          } else {
            train_pred <- predict(final_model, dtrain)
            test_pred <- predict(final_model, dtest)

            rmse_train <- sqrt(mean((train_df[[Dependency_gene]] - train_pred)^2))
            mae_train <- mean(abs(train_df[[Dependency_gene]] - train_pred))
            r2_train <- cor(train_df[[Dependency_gene]], train_pred)^2

            rmse_test <- sqrt(mean((test_df[[Dependency_gene]] - test_pred)^2))
            mae_test <- mean(abs(test_df[[Dependency_gene]] - test_pred))
            r2_test <- cor(test_df[[Dependency_gene]], test_pred)^2

            Validation_Metric <- best_score  # validation RMSE

            AUC_evaluation_results <- list(
              training_rmse = rmse_train,
              training_mae = mae_train,
              training_r2 = r2_train,
              testing_rmse = rmse_test,
              testing_mae = mae_test,
              testing_r2 = r2_test,
              validation_rmse = best_score,
              validation_r2 = 1 - (best_score^2 / var(train_df[[Dependency_gene]]))
            )
          }

          feature_importance <- rank_feature_importance_xgboost(final_model)
          param_string <- paste0(best_params$max_depth, "-", best_params$eta, "-", best_params$gamma, "-",
                                 best_params$colsample_bytree, "-", best_params$min_child_weight, "-",
                                 best_params$subsample, "-", best_nrounds)

          return(list(
            model = final_model,
            evaluation = AUC_evaluation_results,
            validation_metric = Validation_Metric,
            tuned_value = param_string,
            feature_importance = feature_importance
          ))
        }

        # triggers the function ---
        result <- tune_and_evaluate_xgboost(
          train_df = train_df,
          test_df = test_df,
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
            Prediction_Accuracy     = if (model_type == "Classification") AUC_evaluation_results$testing_accuracy else NA,
            Prediction_Precision    = if (model_type == "Classification") AUC_evaluation_results$testing_precision else NA,
            Prediction_Recall       = if (model_type == "Classification") AUC_evaluation_results$testing_recall else NA,
            Prediction_F1           = if (model_type == "Classification") AUC_evaluation_results$testing_F1 else NA,
            Prediction_Kappa        = if (model_type == "Classification") AUC_evaluation_results$testing_Kappa else NA,
            Prediction_AccuracyPValue = if (model_type == "Classification") AUC_evaluation_results$testing_accuracyPValue else NA,
            Prediction_McnemarPValue  = if (model_type == "Classification") AUC_evaluation_results$testing_McnemarPValue else NA,
            Prediction_AUROC        = if (model_type == "Classification") AUC_evaluation_results$testing_auroc else NA,
            Validation_Accuracy     = if (model_type == "Classification") round(as.numeric(result$validation_metric), 2) else NA,

            # Regression metrics
            Training_RMSE     = if (model_type == "Regression") AUC_evaluation_results$training_rmse else NA,
            Training_MAE      = if (model_type == "Regression") AUC_evaluation_results$training_mae  else NA,
            Training_R2       = if (model_type == "Regression") AUC_evaluation_results$training_r2   else NA,
            Prediction_RMSE   = if (model_type == "Regression") AUC_evaluation_results$testing_rmse  else NA,
            Prediction_MAE    = if (model_type == "Regression") AUC_evaluation_results$testing_mae   else NA,
            Prediction_R2     = if (model_type == "Regression") AUC_evaluation_results$testing_r2    else NA,
            Validation_RMSE   = Validation_RMSE,
            Validation_Rsq    = Validation_Rsq,

            # Shared metrics
            time_taken = round(as.numeric(time_taken, units = "secs"), 10),
            feature_importance = feature_importance
          )
        )

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

          # Probabilities for class 1
          Decision_Tree.model.train.prob <- predict(Decision_Tree.model, train_df, type = "prob")[, 2]
          Decision_Tree.model.predict.prob <- predict(Decision_Tree.model, test_df, type = "prob")[, 2]

        } else if (model_type == "Regression") {
          # Extract RMSE and Rsquared at best tuning parameter
          best_row_index <- as.numeric(rownames(Decision_Tree.model$bestTune))
          Validation_RMSE <- Decision_Tree.model$results[best_row_index, "RMSE"]
          Validation_Rsq <- Decision_Tree.model$results[best_row_index, "Rsquared"]

          # Predicted values (continuous)
          Decision_Tree.model.train.prob <- predict(Decision_Tree.model, train_df)
          Decision_Tree.model.predict.prob <- predict(Decision_Tree.model, test_df)
        }


        AUC_evaluation_results <- evaluate_with_optimal_threshold(
          training_pred = Decision_Tree.model.train.prob,
          testing_pred = Decision_Tree.model.predict.prob,
          train_labels = train_df[[Dependency_gene]],
          test_labels = test_df[[Dependency_gene]],
          #fallback_conf_matrix = Decision_Tree.model.predict.confusionMatrix, # in case if optimal threshold can not be calculate
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
            Prediction_Accuracy = if (model_type == "Classification") AUC_evaluation_results$testing_accuracy else NA,
            Prediction_Precision = if (model_type == "Classification") AUC_evaluation_results$testing_precision else NA,
            Prediction_Recall = if (model_type == "Classification") AUC_evaluation_results$testing_recall else NA,
            Prediction_F1 = if (model_type == "Classification") AUC_evaluation_results$testing_F1 else NA,
            Prediction_Kappa = if (model_type == "Classification") AUC_evaluation_results$testing_Kappa else NA,
            Prediction_AccuracyPValue = if (model_type == "Classification") AUC_evaluation_results$testing_accuracyPValue else NA,
            Prediction_McnemarPValue = if (model_type == "Classification") AUC_evaluation_results$testing_McnemarPValue else NA,
            Prediction_AUROC = if (model_type == "Classification") AUC_evaluation_results$testing_auroc else NA,
            Validation_Accuracy = if (model_type == "Classification") round(as.numeric(Validation_Accuracy), 2) else NA,

            # Regression metrics
            Training_RMSE = if (model_type == "Regression") AUC_evaluation_results$training_rmse else NA,
            Training_MAE = if (model_type == "Regression") AUC_evaluation_results$training_mae else NA,
            Training_R2 = if (model_type == "Regression") AUC_evaluation_results$training_r2 else NA,
            Prediction_RMSE = if (model_type == "Regression") AUC_evaluation_results$testing_rmse else NA,
            Prediction_MAE = if (model_type == "Regression") AUC_evaluation_results$testing_mae else NA,
            Prediction_R2 = if (model_type == "Regression") AUC_evaluation_results$testing_r2 else NA,
            Validation_RMSE = if (model_type == "Regression") Validation_RMSE else NA,
            Validation_Rsq = if (model_type == "Regression") Validation_Rsq else NA,

            # Shared metrics
            time_taken = round(as.numeric(time_taken, units = "secs"), 10),
            feature_importance = feature_importance
          )
        )

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
  }
}
