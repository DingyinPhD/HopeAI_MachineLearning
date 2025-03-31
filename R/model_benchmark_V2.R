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
                            dependency_threshold,
                            gene_hits_percentage_cutoff_Lower = 0.2,
                            gene_hits_percentage_cutoff_Upper = 0.8,
                            XBoost_tuning_grid = "Simple") {

  Features <- Features
  Dependency_gene <- Target
  threshold <- dependency_threshold
  cutoff_Lower <- gene_hits_percentage_cutoff_Lower
  cutoff_Upper <- gene_hits_percentage_cutoff_Upper
  XBoost_tuning_grid <- XBoost_tuning_grid
  # Setting Machine learning algorithm for benchmarking
  ML_model <- model

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

  # Function to calculate and rank feature importance
  # If using RandomForest package
  rank_feature_importance_RF <- function(model) {
    imp <- importance(model)
    # Sort by MeanDecreaseGini in decreasing order
    ordered_imp <- imp[order(imp[, "MeanDecreaseGini"], decreasing = TRUE), ]
    cg_list <- names(ordered_imp)
    cg_string <- paste(cg_list, collapse = "|")
    return(cg_string)
  }

  # If using caret package
  #rank_feature_importance_caret <- function(model) {
  #  imp <- varImp(model)
  #  imp_df <- imp$importance
  #  ordered_imp <- imp_df[order(imp_df[, 1], decreasing = TRUE), ]
  #  cg_list <- rownames(ordered_imp)
  #  cg_string <- paste(cg_list, collapse = "|")
  #  return(cg_string)
  #}

  rank_feature_importance_caret <- function(model) {
    # Try varImp safely
    imp <- tryCatch({
      varImp(model)$importance
    }, error = function(e) {
      warning("Feature importance could not be computed: ", e$message)
      return(NULL)
    })

    # If importance was returned
    if (!is.null(imp)) {
      ordered_imp <- imp[order(imp[, 1], decreasing = TRUE), , drop = FALSE]
      cg_list <- rownames(ordered_imp)
      cg_string <- paste(cg_list, collapse = "|")
      return(cg_string)
    } else {
      return("Empty imp")
    }
  }


  # If using xgboost package
  rank_feature_importance_xgboost <- function(model) {
    imp <- xgb.importance(model = model)
    ordered_imp <- imp[order(imp$Gain, decreasing = TRUE), ]
    cg_list <- ordered_imp$Feature
    cg_string <- paste(cg_list, collapse = "|")
    return(cg_string)
  }

  # If using e1071::svm package
  rank_feature_importance_e1071 <- function(model) {
    coefficients <- t(model$coefs) %*% model$SV
    importance <- abs(coefficients)
    importance <- importance / max(importance)
    importance_df <- data.frame(Variable = names(train_df)[colnames(train_df) != Dependency_gene], Importance = as.vector(importance))
    importance_df <- importance_df[order(importance_df$Importance, decreasing = TRUE), ]
    importance_df <- importance_df %>% mutate(
      comb = paste0(Variable, "-", round(Importance,2))
    ) %>%
      pull(Variable)
    cg_string <- paste(importance_df, collapse = "|")
    return(cg_string)
  }

  # Define function to register and stop CPU cluster for parallel processing
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
                                               Optimal_Threshold = NA,
                                               Prediction_Accuracy = NA,
                                               Prediction_Precision = NA,
                                               Prediction_Recall = NA,
                                               Prediction_F1 = NA,
                                               Prediction_Kappa = NA,
                                               AccuracyPValue = NA,
                                               McnemarPValue = NA,
                                               AUROC = NA,
                                               time_taken = NA,
                                               feature_importance = NA,
                                               max_tuning_iteration = NA,
                                               gene_hits_percentage_cutoff_Lower = NA,
                                               gene_hits_percentage_cutoff_Upper = NA))

    assign("final_benchmark_result", final_benchmark_result, envir = .GlobalEnv)  # Save in global env

  } else {
    # Script Start ===============================================================================================================================

    # Remove column if all betascore are either greater than 0.7 or less than 0.3
    subset_indices <- !apply(merge_data, 2, function(col) {
      all((col > 0.7 | col < 0.3), na.rm = TRUE)
    }) | colnames(merge_data) == Dependency_gene

    merge_data <- merge_data[, subset_indices, drop = FALSE]

    # Create training and test datasets
    merge_data <- merge_data %>%
      mutate(!!sym(Dependency_gene) := case_when(
        !!sym(Dependency_gene) <= -1.5 ~ 1,
        TRUE ~ 0
      )) %>%
      mutate(!!sym(Dependency_gene) := as.factor(!!sym(Dependency_gene)))

    merge_data <- merge_data[, colMeans(is.na(merge_data)) < 0.5]
    merge_data <- na.omit(merge_data)
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
        ntree_to_try <- seq(500,1000, by = 250)
        index_of_target <- which(colnames(test_df) == Dependency_gene)

        for (i in 1:max_tuning_iteration) {
          for (ntree in ntree_to_try) {
            tmp <- tuneRF(x = test_df[, -index_of_target],
                          y = test_df[, index_of_target],
                          ntreeTry = ntree,
                          stepFactor = 1.5,
                          doBest = FALSE,
                          trace = FALSE)

            tmp_df <- as.data.frame(tmp)
            tmp_df$iteration <- i
            tmp_df$ntree <- ntree

            RF_benchmark <- rbind(RF_benchmark, tmp_df)
          }
        }

        # Re-train the model using the best tuned hyper-parameters ---
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
        Validation_accuracy <- 1 - best_combos$median_OOBError

        # Model retrain
        RF.model <- randomForest(
          as.formula(paste(Dependency_gene, "~ .")),
          data = train_df,
          mtry = RF_best_tunned_mtry,
          ntree = RF_best_tunned_ntree
        )

        RF.model.accuracy <- sum(diag(RF.model$confusion)) / sum(RF.model$confusion)
        TP <- RF.model$confusion["1", "1"]  # True Positives
        FP <- RF.model$confusion["0", "1"]  # False Positives
        RF.model.precision <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)

        # Predict using the best tuned hyper-parameters
        RF.model.class <- predict(RF.model, test_df, type = "class")
        RF.model.class.confusionMatrix <- confusionMatrix(RF.model.class, test_df[[Dependency_gene]])
        RF.model.class.confusionMatrix$overall["Accuracy"] # prediction_accuracy
        RF.model.class.confusionMatrix$byClass["Precision"] # prediction_precision

        RF.model.predict.prob <- predict(RF.model, test_df, type = "prob")[, 2] # Probabilities for class 1

        roc_curve <- NULL
        auroc <- NULL

        tryCatch({
          roc_curve <- roc(test_df[[Dependency_gene]], RF.model.predict.prob)  # May fail if no positive class
          auroc <- auc(roc_curve)
        }, error = function(e) {
          message("ROC calculation failed: ", e$message)
        })

        if (!is.null(roc_curve) & !is.null(auroc)) {
          # Get optimal threshold
          optimal_threshold <- coords(roc_curve, "best", ret = "threshold")

          # If 'coords' returns a single value (vector), no $threshold extraction needed
          threshold_value <- as.numeric(optimal_threshold[[1]])

          # Generate new predictions
          new_predictions <- ifelse(RF.model.predict.prob > threshold_value, 1, 0)
          new_predictions <- factor(new_predictions, levels = levels(factor(test_df[[Dependency_gene]])))
          test_labels <- factor(test_df[[Dependency_gene]])

          # Compute confusion matrix using optimal threshold
          new_conf_matrix <- confusionMatrix(new_predictions, test_labels, positive = "1")

        } else {
          # Fallback to default confusion matrix if ROC fails
          new_conf_matrix <- RF.model.class.confusionMatrix
          threshold_value <- 0.5 # Default Threshold
          auroc <- -1 # as a place holder to indicate failure
        }

        # Calculate and rank feature importance
        feature_importance <- rank_feature_importance_RF(RF.model)

        end_time <- Sys.time()

        time_taken <- end_time - start_time
        # Print final result
        final_benchmark_result <- rbind(final_benchmark_result,
                                        data.frame(Algorithm = "Random Forest",
                                                   Hyperparameter = "mtry-ntree",
                                                   Tuned_Value = paste0(RF_best_tunned_mtry,"-",RF_best_tunned_ntree),
                                                   Optimal_Threshold = round(threshold_value,2),
                                                   Prediction_Accuracy = round(new_conf_matrix$overall["Accuracy"],2),
                                                   Prediction_Precision = round(new_conf_matrix$byClass["Precision"],2),
                                                   Prediction_Recall = round(new_conf_matrix$byClass["Recall"],2),
                                                   Prediction_F1 = round(new_conf_matrix$byClass["F1"],2),
                                                   Prediction_Kappa = round(new_conf_matrix$overall["Kappa"],2),
                                                   AccuracyPValue = round(new_conf_matrix$overall["AccuracyPValue"],2),
                                                   McnemarPValue = round(new_conf_matrix$overall["McnemarPValue"],2),
                                                   AUROC = round(auroc,2),
                                                   time_taken = round(as.numeric(time_taken, units = "secs"), 10),
                                                   feature_importance = feature_importance,
                                                   Validation_accuracy = round(as.numeric(Validation_accuracy), 2)))

        print("Benchmarking Random Forest END")

        # End of Benchmarking Random Forest ---

      } else if (MLmodel == "Naïve Bayes") {

        start_time <- Sys.time()

        # Benchmarking Naïve Bayes ---------------------------------------------------------------
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
        Validation_accuracy <- NB.model$results[best_row_index, ]$Accuracy

        NB_best_tunned_usekernel <- NB.model$bestTune$usekernel # usekernel is either TRUE or FALSE
        NB_best_tunned_fL <- NB.model$bestTune$fL
        NB_best_tunned_adjust <- NB.model$bestTune$adjust

        NB.model.confusionMatrix <- confusionMatrix(NB.model)
        TP <- NB.model.confusionMatrix$table["1", "1"]  # True Positives
        FP <- NB.model.confusionMatrix$table["0", "1"]  # False Positives
        NB.model.precision <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)

        # Predict using the best tuned hyper-parameters
        NB.model.predict <- predict(NB.model, test_df)
        NB.model.predict.confusionMatrix <- confusionMatrix(NB.model.predict, test_df[[Dependency_gene]])

        NB.model.predict.prob <- predict(NB.model, test_df, type = "prob")[, 2] # Probabilities for class 1

        roc_curve <- NULL
        auroc <- NULL

        tryCatch({
          roc_curve <- roc(test_df[[Dependency_gene]], NB.model.predict.prob)  # May fail if no positive class
          auroc <- auc(roc_curve)
        }, error = function(e) {
          message("ROC calculation failed: ", e$message)
        })

        if (!is.null(roc_curve) & !is.null(auroc)) {
          # Get optimal threshold
          optimal_threshold <- coords(roc_curve, "best", ret = "threshold")
          # If 'coords' returns a single value (vector), no $threshold extraction needed
          #threshold_value <- as.numeric(optimal_threshold)
          threshold_value <- as.numeric(optimal_threshold[[1]])
          # Generate new predictions
          new_predictions <- ifelse(NB.model.predict.prob > threshold_value, 1, 0)
          new_predictions <- factor(new_predictions, levels = levels(factor(test_df[[Dependency_gene]])))
          test_labels <- factor(test_df[[Dependency_gene]])
          # Compute confusion matrix using optimal threshold
          new_conf_matrix <- confusionMatrix(new_predictions, test_labels, positive = "1")

        } else {
          # Fallback to default confusion matrix if ROC fails
          new_conf_matrix <- NB.model.predict.confusionMatrix
          threshold_value <- 0.5 # Default Threshold
          auroc <- -1 # as a place holder to indicate failure
        }

        # Calculate and rank feature importance
        feature_importance <- rank_feature_importance_caret(NB.model)

        end_time <- Sys.time()

        time_taken <- end_time - start_time

        # Write final benchmark result
        final_benchmark_result <- rbind(final_benchmark_result,
                                        data.frame(Algorithm = "Naïve Bayes",
                                                   Hyperparameter = "fL-usekernel-adjust",
                                                   Tuned_Value = paste0(NB_best_tunned_fL,"-",NB_best_tunned_usekernel,"-",NB_best_tunned_adjust),
                                                   Optimal_Threshold = round(threshold_value,2),
                                                   Prediction_Accuracy = round(new_conf_matrix$overall["Accuracy"],2),
                                                   Prediction_Precision = round(new_conf_matrix$byClass["Precision"],2),
                                                   Prediction_Recall = round(new_conf_matrix$byClass["Recall"],2),
                                                   Prediction_F1 = round(new_conf_matrix$byClass["F1"],2),
                                                   Prediction_Kappa = round(new_conf_matrix$overall["Kappa"],2),
                                                   AccuracyPValue = round(new_conf_matrix$overall["AccuracyPValue"],2),
                                                   McnemarPValue = round(new_conf_matrix$overall["McnemarPValue"],2),
                                                   AUROC = round(auroc,2),
                                                   time_taken = round(as.numeric(time_taken, units = "secs"), 10),
                                                   feature_importance = feature_importance,
                                                   Validation_accuracy = round(Validation_accuracy,2)))

        print("Benchmarking Naïve Bayes END")
        # End of Benchmarking Naïve Bayes ---


      } else if (MLmodel == "SVM") {

        start_time <- Sys.time()

        # Benchmarking SVM ---------------------------------------------------------------
        print("Benchmarking SVM Start")

        kernel_list <- c("linear", "polynomial", "radial", "sigmoid")
        # Initialize result storage
        SVM_benchmark <- data.frame()

        for (i in 1:max_tuning_iteration) {

          iteration_results <- data.frame()

          for (k in kernel_list) {

            # Function to tune SVM for a given kernel
            if (k == "linear") {
              # Linear kernel: tune cost only
              tune_out <- tryCatch({
                e1071::tune(
                  e1071::svm,
                  as.formula(paste(Dependency_gene, "~ .")),
                  data = train_df,
                  kernel = k,
                  ranges = list(cost = 10^seq(-3, 2, by = 1)),
                  tunecontrol = tune.control(cross = 10)
                )
              }, error = function(e) {
                message("Tuning failed for kernel: ", k, " | Error: ", e$message)
                return(NULL)
              })

              if (is.null(tune_out)) {
                result <- data.frame(iteration = i, kernel = k,
                                     best_cost = NA,
                                     best_gamma = NA,
                                     best_error = Inf)
              } else {
                result <- data.frame(iteration = i, kernel = k,
                                     best_cost = tune_out$best.parameters$cost,
                                     best_gamma = NA,
                                     best_error = tune_out$best.performance)
              }

            } else {
              # Non-linear kernel: tune cost and gamma
              tune_out <- tryCatch({
                e1071::tune(
                  e1071::svm,
                  as.formula(paste(Dependency_gene, "~ .")),
                  data = train_df,
                  kernel = k,
                  ranges = list(cost = 10^seq(-3, 2, by = 1),
                                gamma = 10^seq(-3, 2, by = 1)),
                  tunecontrol = tune.control(cross = 10)
                )
              }, error = function(e) {
                message("Tuning failed for kernel: ", k, " | Error: ", e$message)
                return(NULL)
              })

              if (is.null(tune_out)) {
                result <- data.frame(iteration = i, kernel = k,
                                     best_cost = NA,
                                     best_gamma = NA,
                                     best_error = Inf)
              } else {
                result <- data.frame(iteration = i, kernel = k,
                                     best_cost = tune_out$best.parameters$cost,
                                     best_gamma = tune_out$best.parameters$gamma,
                                     best_error = tune_out$best.performance)
              }
            }

            iteration_results <- rbind(iteration_results, result)
          }

          # Select the best result from this iteration
          best_result <- iteration_results[which.min(iteration_results$best_error), ]

          # Append to benchmark results
          SVM_benchmark <- rbind(SVM_benchmark, best_result)
        } # end of 1:max_tuning_iteration loop

        # Re-train the model using the best tuned hyper-parameters ---
        # Get benchmark summary
        SVM_benchmark <- SVM_benchmark %>%
          mutate(Hyperparam = paste0(kernel,"-",best_cost, "-",best_gamma)) %>%
          group_by(Hyperparam) %>%
          mutate(mean_error_rate = median(best_error))

        SVM_benchmark_summary <- as.data.frame(table(SVM_benchmark$Hyperparam)) %>%
          left_join(.,
                    SVM_benchmark %>% dplyr::select(Hyperparam,mean_error_rate) %>% unique(),
                    by = c("Var1" = "Hyperparam")) %>%
          mutate(Cross_validated_accuracy = 1 - mean_error_rate)


        SVM_best_tunned <- SVM_benchmark_summary$Var1[which.max(SVM_benchmark_summary$Freq)]
        SVM_best_tunned_kernel <- as.character(str_split_i(SVM_best_tunned, "-", 1))
        SVM_best_tunned_cost <- as.numeric(str_split_i(SVM_best_tunned, "-", 2))
        SVM_best_tunned_gamma <- as.numeric(str_split_i(SVM_best_tunned, "-", 3))
        Validation_accuracy <- SVM_benchmark_summary$Cross_validated_accuracy[which.max(SVM_benchmark_summary$Freq)]


        if (SVM_best_tunned_kernel == "linear") {
        SVM.model <- svm(as.formula(paste(Dependency_gene, "~ .")),
                         data = train_df,
                         type = 'C-classification',
                         kernel = SVM_best_tunned_kernel,
                         cost = SVM_best_tunned_cost,
                         scale = FALSE,
                         probability = TRUE)
        } else {
          SVM.model <- svm(as.formula(paste(Dependency_gene, "~ .")),
                           data = train_df,
                           type = 'C-classification',
                           kernel = SVM_best_tunned_kernel,
                           cost = SVM_best_tunned_cost,
                           gamma = SVM_best_tunned_gamma,
                           scale = FALSE,
                           probability = TRUE)
        }

        # Predict using the best tuned hyper-parameters
        SVM.model.predict <- predict(SVM.model, test_df) # The model directly assigns class labels based on the hyperplane decision boundary.
        SVM.model.predict.confusionMatrix <- confusionMatrix(SVM.model.predict, test_df[[Dependency_gene]])
        SVM.model.predict.confusionMatrix$overall["Accuracy"] # prediction_accuracy
        SVM.model.predict.confusionMatrix$byClass["Precision"] # prediction_precision

        SVM.model.predict.prob <- attr(predict(SVM.model, test_df, probability = TRUE), "probabilities")[,2] # Returns class probabilities for threshold tuning.

        roc_curve <- NULL
        auroc <- NULL

        tryCatch({
          roc_curve <- roc(test_df[[Dependency_gene]], SVM.model.predict.prob)  # May fail if no positive class
          auroc <- auc(roc_curve)
        }, error = function(e) {
          message("ROC calculation failed: ", e$message)
        })

        if (!is.null(roc_curve) & !is.null(auroc)) {
          # Get optimal threshold
          optimal_threshold <- coords(roc_curve, "best", ret = "threshold")

          # If 'coords' returns a single value (vector), no $threshold extraction needed
          threshold_value <- as.numeric(optimal_threshold[[1]])

          # Generate new predictions
          new_predictions <- ifelse(SVM.model.predict.prob > threshold_value, 1, 0)
          new_predictions <- factor(new_predictions, levels = levels(factor(test_df[[Dependency_gene]])))
          test_labels <- factor(test_df[[Dependency_gene]])

          # Compute confusion matrix using optimal threshold
          new_conf_matrix <- confusionMatrix(new_predictions, test_labels, positive = "1")

        } else {
          # Fallback to default confusion matrix if ROC fails
          new_conf_matrix <- SVM.model.predict.confusionMatrix
          threshold_value <- 0.5 # Default Threshold
          auroc <- -1 # as a place holder to indicate failure
        }

        # Calculate and rank feature importance
        feature_importance <- rank_feature_importance_e1071(SVM.model)

        end_time <- Sys.time()

        time_taken <- end_time - start_time

        # Write final benchmark result
        final_benchmark_result <- rbind(final_benchmark_result,
                                        data.frame(Algorithm = "SVM",
                                                   Hyperparameter = "kernel-cost-gamma",
                                                   Tuned_Value = SVM_best_tunned,
                                                   Optimal_Threshold = round(threshold_value,2),
                                                   Prediction_Accuracy = round(new_conf_matrix$overall["Accuracy"],2),
                                                   Prediction_Precision = round(new_conf_matrix$byClass["Precision"],2),
                                                   Prediction_Recall = round(new_conf_matrix$byClass["Recall"],2),
                                                   Prediction_F1 = round(new_conf_matrix$byClass["F1"],2),
                                                   Prediction_Kappa = round(new_conf_matrix$overall["Kappa"],2),
                                                   AccuracyPValue = round(new_conf_matrix$overall["AccuracyPValue"],2),
                                                   McnemarPValue = round(new_conf_matrix$overall["McnemarPValue"],2),
                                                   AUROC = round(auroc,2),
                                                   time_taken = round(as.numeric(time_taken, units = "secs"), 10),
                                                   feature_importance = feature_importance,
                                                   Validation_accuracy = round(Validation_accuracy,2)))

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
        ECN.model <- train(as.formula(paste(Dependency_gene, "~ .")),
                           data = train_df,
                           preProcess = c("center", "scale"),
                           method = "glmnet",
                           tuneGrid = tune_grid,
                           trControl = ctrlspecs,
                           family = "binomial")

        # Extract best hyperparameters
        best_row_index <- as.numeric(rownames(ECN.model$bestTune))
        Validation_accuracy <- ECN.model$results[best_row_index, ]$Accuracy

        # Predict using the best tuned hyper-parameters
        ECN.model.predict <- predict(ECN.model, test_df)
        ECN.model.predict.confusionMatrix <- confusionMatrix(ECN.model.predict, test_df[[Dependency_gene]])
        ECN.model.predict.confusionMatrix$overall["Accuracy"] # prediction_accuracy
        ECN.model.predict.confusionMatrix$byClass["Precision"] # prediction_precision

        ECN.model.predict.prob <- predict(ECN.model, test_df, type = "prob")[, 2] # Probabilities for class 1

        roc_curve <- NULL
        auroc <- NULL

        tryCatch({
          roc_curve <- roc(test_df[[Dependency_gene]], ECN.model.predict.prob)  # May fail if no positive class
          auroc <- auc(roc_curve)
        }, error = function(e) {
          message("ROC calculation failed: ", e$message)
        })

        if (!is.null(roc_curve) & !is.null(auroc)) {
          # Get optimal threshold
          optimal_threshold <- coords(roc_curve, "best", ret = "threshold")

          # If 'coords' returns a single value (vector), no $threshold extraction needed
          threshold_value <- as.numeric(optimal_threshold[[1]])

          # Generate new predictions
          new_predictions <- ifelse(ECN.model.predict.prob > threshold_value, 1, 0)
          new_predictions <- factor(new_predictions, levels = levels(factor(test_df[[Dependency_gene]])))
          test_labels <- factor(test_df[[Dependency_gene]])

          # Compute confusion matrix using optimal threshold
          new_conf_matrix <- confusionMatrix(new_predictions, test_labels, positive = "1")

        } else {
          # Fallback to default confusion matrix if ROC fails
          new_conf_matrix <- ECN.model.predict.confusionMatrix
          threshold_value <- 0.5 # Default Threshold
          auroc <- -1 # as a place holder to indicate failure
        }

        # Calculate and rank feature importance
        feature_importance <- rank_feature_importance_caret(ECN.model)

        end_time <- Sys.time()

        time_taken <- end_time - start_time

        # Write final benchmark result
        final_benchmark_result <- rbind(final_benchmark_result,
                                        data.frame(Algorithm = "Elastic-Net",
                                                   Hyperparameter = "alpha-lamda",
                                                   Tuned_Value = paste0(ECN.model$bestTune$alpha,"-",ECN.model$bestTune$lambda),
                                                   Optimal_Threshold = round(threshold_value,2),
                                                   Prediction_Accuracy = round(new_conf_matrix$overall["Accuracy"],2),
                                                   Prediction_Precision = round(new_conf_matrix$byClass["Precision"],2),
                                                   Prediction_Recall = round(new_conf_matrix$byClass["Recall"],2),
                                                   Prediction_F1 = round(new_conf_matrix$byClass["F1"],2),
                                                   Prediction_Kappa = round(new_conf_matrix$overall["Kappa"],2),
                                                   AccuracyPValue = round(new_conf_matrix$overall["AccuracyPValue"],2),
                                                   McnemarPValue = round(new_conf_matrix$overall["McnemarPValue"],2),
                                                   AUROC = round(auroc,2),
                                                   time_taken = round(as.numeric(time_taken, units = "secs"), 10),
                                                   feature_importance = feature_importance,
                                                   Validation_accuracy = round(Validation_accuracy,2)))

        print("Benchmarking ECN END")
        # End of Benchmarking ECN ---


      } else if (MLmodel == "KNN") {

        start_time <- Sys.time()

        # Benchmarking KNN ---------------------------------------------------------------
        print("Benchmarking KNN Start")

        metric <- "Accuracy"
        grid <- expand.grid(.k=seq(1,50,by=1))

        KNN.model <- train(as.formula(paste(Dependency_gene, "~ .")),
                           data= train_df,
                           method="knn",
                           metric=metric,
                           tuneGrid=grid,
                           trControl=ctrlspecs,
                           na.action = na.omit)

        best_row_index <- as.numeric(rownames(KNN.model$bestTune))
        Validation_accuracy <- KNN.model$results[best_row_index, ]$Accuracy


        # Predict using the best tuned hyper-parameters
        KNN.model.predict <- predict(KNN.model, test_df)
        KNN.model.predict.confusionMatrix <- confusionMatrix(KNN.model.predict, test_df[[Dependency_gene]])
        KNN.model.predict.confusionMatrix$overall["Accuracy"] # prediction_accuracy
        KNN.model.predict.confusionMatrix$byClass["Precision"] # prediction_precision

        KNN.model.predict.prob <- predict(KNN.model, test_df, type = "prob")[, 2] # Probabilities for class 1

        roc_curve <- NULL
        auroc <- NULL

        tryCatch({
          roc_curve <- roc(test_df[[Dependency_gene]], KNN.model.predict.prob)  # May fail if no positive class
          auroc <- auc(roc_curve)
        }, error = function(e) {
          message("ROC calculation failed: ", e$message)
        })

        if (!is.null(roc_curve) & !is.null(auroc)) {
          # Get optimal threshold
          optimal_threshold <- coords(roc_curve, "best", ret = "threshold")

          # If 'coords' returns a single value (vector), no $threshold extraction needed
          threshold_value <- as.numeric(optimal_threshold[[1]])

          # Generate new predictions
          new_predictions <- ifelse(KNN.model.predict.prob > threshold_value, 1, 0)
          new_predictions <- factor(new_predictions, levels = levels(factor(test_df[[Dependency_gene]])))
          test_labels <- factor(test_df[[Dependency_gene]])

          # Compute confusion matrix using optimal threshold
          new_conf_matrix <- confusionMatrix(new_predictions, test_labels, positive = "1")

        } else {
          # Fallback to default confusion matrix if ROC fails
          new_conf_matrix <- KNN.model.predict.confusionMatrix
          threshold_value <- 0.5 # Default Threshold
          auroc <- -1 # as a place holder to indicate failure
        }

        # Calculate and rank feature importance
        feature_importance <- rank_feature_importance_caret(KNN.model)

        end_time <- Sys.time()

        time_taken <- end_time - start_time

        # Write final benchmark result
        final_benchmark_result <- rbind(final_benchmark_result,
                                        data.frame(Algorithm = "KNN",
                                                   Hyperparameter = "k",
                                                   Tuned_Value = KNN.model$bestTune$k,
                                                   Optimal_Threshold = round(threshold_value,2),
                                                   Prediction_Accuracy = round(new_conf_matrix$overall["Accuracy"],2),
                                                   Prediction_Precision = round(new_conf_matrix$byClass["Precision"],2),
                                                   Prediction_Recall = round(new_conf_matrix$byClass["Recall"],2),
                                                   Prediction_F1 = round(new_conf_matrix$byClass["F1"],2),
                                                   Prediction_Kappa = round(new_conf_matrix$overall["Kappa"],2),
                                                   AccuracyPValue = round(new_conf_matrix$overall["AccuracyPValue"],2),
                                                   McnemarPValue = round(new_conf_matrix$overall["McnemarPValue"],2),
                                                   AUROC = round(auroc,2),
                                                   time_taken = round(as.numeric(time_taken, units = "secs"), 10),
                                                   feature_importance = feature_importance,
                                                   Validation_accuracy = round(Validation_accuracy,2)))



        print("Benchmarking KNN END")
        # End of Benchmarking KNN ---


      } else if (MLmodel == "Neural Network") {

        start_time <- Sys.time()

        # Benchmarking Neural Network ---------------------------------------------------------------
        print("Benchmarking Neural Network Start")

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
                            linout = FALSE,  # Use linout = TRUE for regression
                            na.action = na.omit,
                            trace = FALSE)  # Suppress training output

        best_row_index <- as.numeric(rownames(NeurNet.model$bestTune))
        Validation_accuracy <- NeurNet.model$results[best_row_index, ]$Accuracy


        # Predict using the best tuned hyper-parameters
        NeurNet.model.predict <- predict(NeurNet.model, test_df)
        NeurNet.model.predict.confusionMatrix <- confusionMatrix(NeurNet.model.predict, test_df[[Dependency_gene]])
        NeurNet.model.predict.confusionMatrix$overall["Accuracy"] # prediction_accuracy
        NeurNet.model.predict.confusionMatrix$byClass["Precision"] # prediction_precision

        NeurNet.model.predict.prob <- predict(NeurNet.model, test_df, type = "prob")[, 2] # Probabilities for class 1

        roc_curve <- NULL
        auroc <- NULL

        tryCatch({
          roc_curve <- roc(test_df[[Dependency_gene]], NeurNet.model.predict.prob)  # May fail if no positive class
          auroc <- auc(roc_curve)
        }, error = function(e) {
          message("ROC calculation failed: ", e$message)
        })

        if (!is.null(roc_curve) & !is.null(auroc)) {
          # Get optimal threshold
          optimal_threshold <- coords(roc_curve, "best", ret = "threshold")

          # If 'coords' returns a single value (vector), no $threshold extraction needed
          threshold_value <- as.numeric(optimal_threshold[[1]])

          # Generate new predictions
          new_predictions <- ifelse(NeurNet.model.predict.prob > threshold_value, 1, 0)
          new_predictions <- factor(new_predictions, levels = levels(factor(test_df[[Dependency_gene]])))
          test_labels <- factor(test_df[[Dependency_gene]])

          # Compute confusion matrix using optimal threshold
          new_conf_matrix <- confusionMatrix(new_predictions, test_labels, positive = "1")

        } else {
          # Fallback to default confusion matrix if ROC fails
          new_conf_matrix <- NeurNet.model.predict.confusionMatrix
          threshold_value <- 0.5 # Default Threshold
          auroc <- -1 # as a place holder to indicate failure
        }

        # Calculate and rank feature importance
        feature_importance <- rank_feature_importance_caret(NeurNet.model)

        end_time <- Sys.time()

        time_taken <- end_time - start_time

        # Write final benchmark result
        final_benchmark_result <- rbind(final_benchmark_result,
                                        data.frame(Algorithm = "Neural Network",
                                                   Hyperparameter = "size-decay",
                                                   Tuned_Value = paste0(NeurNet.model$bestTune$size,"-",NeurNet.model$bestTune$decay),
                                                   Optimal_Threshold = round(threshold_value,2),
                                                   Prediction_Accuracy = round(new_conf_matrix$overall["Accuracy"],2),
                                                   Prediction_Precision = round(new_conf_matrix$byClass["Precision"],2),
                                                   Prediction_Recall = round(new_conf_matrix$byClass["Recall"],2),
                                                   Prediction_F1 = round(new_conf_matrix$byClass["F1"],2),
                                                   Prediction_Kappa = round(new_conf_matrix$overall["Kappa"],2),
                                                   AccuracyPValue = round(new_conf_matrix$overall["AccuracyPValue"],2),
                                                   McnemarPValue = round(new_conf_matrix$overall["McnemarPValue"],2),
                                                   AUROC = round(auroc,2),
                                                   time_taken = round(as.numeric(time_taken, units = "secs"), 10),
                                                   feature_importance = feature_importance,
                                                   Validation_accuracy = round(Validation_accuracy,2)))

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

        best_row_index <- as.numeric(rownames(AdaBoost.model$bestTune))
        Validation_accuracy <- AdaBoost.model$results[best_row_index, ]$Accuracy


        # Predict using the best tuned hyper-parameters
        AdaBoost.model.predict <- predict(AdaBoost.model, test_df)
        AdaBoost.model.predict.confusionMatrix <- confusionMatrix(AdaBoost.model.predict, test_df[[Dependency_gene]])
        AdaBoost.model.predict.confusionMatrix$overall["Accuracy"] # prediction_accuracy
        AdaBoost.model.predict.confusionMatrix$byClass["Precision"] # prediction_precision

        AdaBoost.model.predict.prob <- predict(AdaBoost.model, test_df, type = "prob")[, 2] # Probabilities for class 1

        roc_curve <- NULL
        auroc <- NULL

        tryCatch({
          roc_curve <- roc(test_df[[Dependency_gene]], AdaBoost.model.predict.prob)  # May fail if no positive class
          auroc <- auc(roc_curve)
        }, error = function(e) {
          message("ROC calculation failed: ", e$message)
        })

        if (!is.null(roc_curve) & !is.null(auroc)) {
          # Get optimal threshold
          optimal_threshold <- coords(roc_curve, "best", ret = "threshold")

          # If 'coords' returns a single value (vector), no $threshold extraction needed
          threshold_value <- as.numeric(optimal_threshold[[1]])

          # Generate new predictions
          new_predictions <- ifelse(AdaBoost.model.predict.prob > threshold_value, 1, 0)
          new_predictions <- factor(new_predictions, levels = levels(factor(test_df[[Dependency_gene]])))
          test_labels <- factor(test_df[[Dependency_gene]])

          # Compute confusion matrix using optimal threshold
          new_conf_matrix <- confusionMatrix(new_predictions, test_labels, positive = "1")

        } else {
          # Fallback to default confusion matrix if ROC fails
          new_conf_matrix <- AdaBoost.model.predict.confusionMatrix
          threshold_value <- 0.5 # Default Threshold
          auroc <- -1 # as a place holder to indicate failure
        }

        # Calculate and rank feature importance
        feature_importance <- rank_feature_importance_caret(AdaBoost.model)

        end_time <- Sys.time()

        time_taken <- end_time - start_time

        # Write final benchmark result
        final_benchmark_result <- rbind(final_benchmark_result,
                                        data.frame(Algorithm = "AdaBoost",
                                                   Hyperparameter = "coeflearn-maxdepth-mfinal",
                                                   Tuned_Value = paste0(AdaBoost.model$bestTune$coeflearn,"-",AdaBoost.model$bestTune$maxdepth,"-",AdaBoost.model$bestTune$mfinal),
                                                   Optimal_Threshold = round(threshold_value,2),
                                                   Prediction_Accuracy = round(new_conf_matrix$overall["Accuracy"],2),
                                                   Prediction_Precision = round(new_conf_matrix$byClass["Precision"],2),
                                                   Prediction_Recall = round(new_conf_matrix$byClass["Recall"],2),
                                                   Prediction_F1 = round(new_conf_matrix$byClass["F1"],2),
                                                   Prediction_Kappa = round(new_conf_matrix$overall["Kappa"],2),
                                                   AccuracyPValue = round(new_conf_matrix$overall["AccuracyPValue"],2),
                                                   McnemarPValue = round(new_conf_matrix$overall["McnemarPValue"],2),
                                                   AUROC = round(auroc,2),
                                                   time_taken = round(as.numeric(time_taken, units = "secs"), 10),
                                                   feature_importance = feature_importance,
                                                   Validation_accuracy = round(Validation_accuracy,2)))

        print("Benchmarking AdaBoost END")
        # End of Benchmarking AdaBoost ---



      } else if (MLmodel == "XGBoost") {

        start_time <- Sys.time()

        # Benchmarking XGBoost ---------------------------------------------------------------
        print("Benchmarking XGBoost start")
        train_df_XGBoost <- train_df
        test_df_XGBoost <- test_df
        # Ensure the target variable is binary (0/1)
        train_df_XGBoost[[Dependency_gene]] <- as.numeric(as.character(train_df_XGBoost[[Dependency_gene]]))  # Convert factor to numeric
        train_df_XGBoost[[Dependency_gene]][train_df_XGBoost[[Dependency_gene]] != 0] <- 1  # Ensure only 0 and 1

        test_df_XGBoost[[Dependency_gene]] <- as.numeric(as.character(test_df_XGBoost[[Dependency_gene]]))
        test_df_XGBoost[[Dependency_gene]][test_df_XGBoost[[Dependency_gene]] != 0] <- 1

        # Convert data to XGBoost DMatrix format
        dtrain <- xgb.DMatrix(
          data = as.matrix(train_df_XGBoost[, -which(names(train_df_XGBoost) == Dependency_gene)]),
          label = train_df_XGBoost[[Dependency_gene]]
        )

        dtest <- xgb.DMatrix(
          data = as.matrix(test_df_XGBoost[, -which(names(test_df_XGBoost) == Dependency_gene)]),
          label = test_df_XGBoost[[Dependency_gene]]
        )



        # Grid search for hyperparameter tuning
        if (XBoost_tuning_grid == "Simple") {
          search_grid <- expand.grid(
            max_depth = c(3, 6),
            eta = c(0.05, 0.3),
            gamma = c(0, 0.5, 1.0),
            colsample_bytree = c(0.6, 1.0),
            min_child_weight = c(1, 3),
            subsample = c(0.75, 1.0)
          )
        } else if (XBoost_tuning_grid == "Fine") {
          search_grid <- expand.grid(
            max_depth = c(2, 4, 6),
            eta = c(0.025, 0.05, 0.1, 0.3),
            gamma = c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0),
            colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
            min_child_weight = c(1, 2, 3),
            subsample = c(0.5, 0.75, 1.0)
          )
        }

        best_auc <- 0
        best_accuracy <- 0
        best_params <- list()

        for (i in 1:nrow(search_grid)) {
          params <- list(
            objective = "binary:logistic",
            eval_metric = "error",
            #num_class = 2,
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
            stratified = TRUE,
            verbose = TRUE
          )

          if (!is.null(cv_results)) {
            # Extract CV accuracy at the best round
            best_iter <- cv_results$best_iteration
            test_error <- cv_results$evaluation_log$test_error_mean[best_iter]
            cv_accuracy <- 1 - test_error

            if (!is.na(cv_accuracy) && cv_accuracy > best_accuracy) {
              best_accuracy <- cv_accuracy
              best_params <- params
              best_nrounds <- best_iter
            }
          }
        }

        Validation_accuracy <- best_accuracy


        # Train the final model with the best tuned hyper-parameters
        XGBoost.model <- xgb.train(
          params = best_params,
          data = dtrain,
          nrounds = best_nrounds
        )

        # Make prediction
        pred <- predict(XGBoost.model, dtest)

        # Step 1: Convert numeric predictions to class labels using a default threshold of 0.5
        pred_labels <- ifelse(pred > 0.5, 1, 0)

        # Convert to factors to match `confusionMatrix` input requirements
        pred_labels <- factor(pred_labels, levels = c(0, 1))
        true_labels <- factor(test_df_XGBoost[[Dependency_gene]], levels = c(0, 1))

        # Step 2: Compute the confusion matrix
        conf_matrix <- confusionMatrix(pred_labels, true_labels)

        # Step 3: Compute ROC curve and AUC
        pred_prob <- pred  # Since `xgboost` returns probabilities by default

        roc_curve <- NULL
        auroc <- NULL

        tryCatch({
          roc_curve <- roc(test_df[[Dependency_gene]], pred_prob)  # May fail if no positive class
          auroc <- auc(roc_curve)
        }, error = function(e) {
          message("ROC calculation failed: ", e$message)
        })

        if (!is.null(roc_curve) & !is.null(auroc)) {
          # Get optimal threshold
          optimal_threshold <- coords(roc_curve, "best", ret = "threshold")

          # If 'coords' returns a single value (vector), no $threshold extraction needed
          threshold_value <- as.numeric(optimal_threshold[[1]])

          # Generate new predictions
          new_predictions <- ifelse(pred_prob > threshold_value, 1, 0)
          new_predictions <- factor(new_predictions, levels = levels(factor(test_df[[Dependency_gene]])))
          test_labels <- factor(test_df[[Dependency_gene]])

          # Compute confusion matrix using optimal threshold
          new_conf_matrix <- confusionMatrix(new_predictions, test_labels, positive = "1")

        } else {
          # Fallback to default confusion matrix if ROC fails
          new_conf_matrix <- conf_matrix
          threshold_value <- 0.5 # Default Threshold
          auroc <- -1 # as a place holder to indicate failure
        }

        # Write final benchmark result
        best_max_depth <- XGBoost.model$params$max_depth
        best_eta <- XGBoost.model$params$eta
        best_gamma <- XGBoost.model$params$gamma
        best_colsample_bytree <- XGBoost.model$params$colsample_bytree
        best_min_child_weight <- XGBoost.model$params$min_child_weight
        best_subsample <- XGBoost.model$params$subsample

        XGBoost_best_tunned <- paste0(best_max_depth,"-",best_eta,"-",best_gamma,"-",best_colsample_bytree,"-",best_min_child_weight,"-",best_subsample,"-",best_nrounds)

        # Calculate and rank feature importance
        feature_importance <- rank_feature_importance_xgboost(XGBoost.model)

        end_time <- Sys.time()

        time_taken <- end_time - start_time

        final_benchmark_result <- rbind(final_benchmark_result,
                                        data.frame(Algorithm = "XGBoost",
                                                   Hyperparameter = "max_depth-eta-gamma-colsample_bytree-min_child_weight-subsample-nrounds",
                                                   Tuned_Value = XGBoost_best_tunned,
                                                   Optimal_Threshold = round(threshold_value,2),
                                                   Prediction_Accuracy = round(new_conf_matrix$overall["Accuracy"],2),
                                                   Prediction_Precision = round(new_conf_matrix$byClass["Precision"],2),
                                                   Prediction_Recall = round(new_conf_matrix$byClass["Recall"],2),
                                                   Prediction_F1 = round(new_conf_matrix$byClass["F1"],2),
                                                   Prediction_Kappa = round(new_conf_matrix$overall["Kappa"],2),
                                                   AccuracyPValue = round(new_conf_matrix$overall["AccuracyPValue"],2),
                                                   McnemarPValue = round(new_conf_matrix$overall["McnemarPValue"],2),
                                                   AUROC = round(auroc,2),
                                                   time_taken = round(as.numeric(time_taken, units = "secs"), 10),
                                                   feature_importance = feature_importance,
                                                   Validation_accuracy = round(Validation_accuracy,2)))

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

        best_row_index <- as.numeric(rownames(Decision_Tree.model$bestTune))
        Validation_accuracy <- Decision_Tree.model$results[best_row_index, ]$Accuracy

        # Predict using the best tuned hyper-parameters
        Decision_Tree.model.predict <- predict(Decision_Tree.model, test_df)
        Decision_Tree.model.predict.confusionMatrix <- confusionMatrix(Decision_Tree.model.predict, test_df[[Dependency_gene]])
        Decision_Tree.model.predict.confusionMatrix$overall["Accuracy"] # prediction_accuracy
        Decision_Tree.model.predict.confusionMatrix$byClass["Precision"] # prediction_precision

        Decision_Tree.model.predict.prob <- predict(Decision_Tree.model, test_df, type = "prob")[, 2] # Probabilities for class 1

        roc_curve <- NULL
        auroc <- NULL

        tryCatch({
          roc_curve <- roc(test_df[[Dependency_gene]], Decision_Tree.model.predict.prob)  # May fail if no positive class
          auroc <- auc(roc_curve)
        }, error = function(e) {
          message("ROC calculation failed: ", e$message)
        })

        if (!is.null(roc_curve) & !is.null(auroc)) {
          # Get optimal threshold
          optimal_threshold <- coords(roc_curve, "best", ret = "threshold")

          # If 'coords' returns a single value (vector), no $threshold extraction needed
          threshold_value <- as.numeric(optimal_threshold[[1]])

          # Generate new predictions
          new_predictions <- ifelse(Decision_Tree.model.predict.prob > threshold_value, 1, 0)
          new_predictions <- factor(new_predictions, levels = levels(factor(test_df[[Dependency_gene]])))
          test_labels <- factor(test_df[[Dependency_gene]])

          # Compute confusion matrix using optimal threshold
          new_conf_matrix <- confusionMatrix(new_predictions, test_labels, positive = "1")

        } else {
          # Fallback to default confusion matrix if ROC fails
          new_conf_matrix <- Decision_Tree.model.predict.confusionMatrix
          threshold_value <- 0.5 # Default Threshold
          auroc <- -1 # as a place holder to indicate failure
        }

        # Calculate and rank feature importance
        feature_importance <- rank_feature_importance_caret(Decision_Tree.model)

        end_time <- Sys.time()

        time_taken <- end_time - start_time

        # Write final benchmark result
        final_benchmark_result <- rbind(final_benchmark_result,
                                        data.frame(Algorithm = "Decision Tree",
                                                   Hyperparameter = "cp",
                                                   Tuned_Value = Decision_Tree.model$bestTune$cp,
                                                   Optimal_Threshold = round(threshold_value,2),
                                                   Prediction_Accuracy = round(new_conf_matrix$overall["Accuracy"],2),
                                                   Prediction_Precision = round(new_conf_matrix$byClass["Precision"],2),
                                                   Prediction_Recall = round(new_conf_matrix$byClass["Recall"],2),
                                                   Prediction_F1 = round(new_conf_matrix$byClass["F1"],2),
                                                   Prediction_Kappa = round(new_conf_matrix$overall["Kappa"],2),
                                                   AccuracyPValue = round(new_conf_matrix$overall["AccuracyPValue"],2),
                                                   McnemarPValue = round(new_conf_matrix$overall["McnemarPValue"],2),
                                                   AUROC = round(auroc,2),
                                                   time_taken = round(as.numeric(time_taken, units = "secs"), 10),
                                                   feature_importance = feature_importance,
                                                   Validation_accuracy = round(Validation_accuracy,2)))

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
             gene_hits_percentage_cutoff_Upper = cutoff_Upper)

    #write.csv(final_benchmark_result,
    #          file = final_benchmark_result_write_out_filename,
    #          row.names = F)
    assign("final_benchmark_result", final_benchmark_result, envir = .GlobalEnv)  # Save in global env
  }
}
