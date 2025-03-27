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
model_benchmark <- function(Features,
                            Target,
                            Input_Data,
                            max_tuning_iteration = 100,
                            fold = 10, # k-fold cross validation
                            model = c("Random Forest", "Naïve Bayes", "Elastic Net", "SVM",
                                      "XGBoost", "AdaBoost", "Neural Network", "KNN", "Decision Tree"),
                            dependency_threshold,
                            gene_hits_percentage_cutoff = 0.2 ) {

  Features <- Features
  Dependency_gene <- Target
  threshold <- dependency_threshold
  cutoff <- gene_hits_percentage_cutoff
  # Setting Machine learning algorithm for benchmarking
  ML_model <- model

  # Setting Global training parameters
  ctrlspecs <- trainControl(method = "cv", number = fold, savePredictions = "all", allowParallel = TRUE) # fold is defined in the function

  # Setting iteration time
  max_tuning_iteration <- max_tuning_iteration # Defined in the function

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

  # Calculate the proportion of hits that are less than threshold
  if (mean(merge_data[[Dependency_gene]] < threshold, na.rm = TRUE) < cutoff) {
    print(paste0("gene hits percentage for ", Dependency_gene, "is ", mean(merge_data[[Dependency_gene]] < threshold, na.rm = TRUE),
                 " which is less than ", cutoff, ", thus skip model benchmarking"))
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
                                               max_tuning_iteration = NA,
                                               gene_hits_percentage_cutoff = NA))

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
        # Benchmarking Random Forest ---------------------------------------------------------------
        print("Benchmarking Random Forest Start")

        # Set up parallel backend
        cl <- makeCluster(num_cores)
        registerDoParallel(cl)
        # Initialize empty result storage
        RF_benchmark <- data.frame()
        # Run tuning in parallel
        RF_benchmark <- foreach(n = 1:max_tuning_iteration, .combine = rbind, .packages = c("randomForest")) %dopar% {

          print(paste0("Iteration ", n))
          oob.values <- vector(length=10)
          accuracy.values <- vector(length=10)
          precision.values <- vector(length=10)
          ntr.values <- vector(length=10)

          for (i in 1:10) {
            temp_oob <- vector(length=length(seq(500,1000, by = 250)))
            temp_acc <- vector(length=length(seq(500,1000, by = 250)))
            temp_prec <- vector(length=length(seq(500,1000, by = 250)))
            temp_ntr <- seq(500,1000, by = 250)
            count <- 1

            for (ntr in temp_ntr) {
              temp.model <- randomForest(as.formula(paste(Dependency_gene, "~ .")),
                                         data=train_df,
                                         mtry=i,
                                         ntree=ntr)

              # Extract confusion matrix
              conf_matrix <- temp.model$confusion
              if (nrow(conf_matrix) == 2) {
                TN <- conf_matrix[1, 1]
                FP <- conf_matrix[1, 2]
                FN <- conf_matrix[2, 1]
                TP <- conf_matrix[2, 2]

                accuracy <- (TP + TN) / sum(conf_matrix)
                precision <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)
              } else {
                accuracy <- NA
                precision <- NA
              }

              temp_oob[count] <- temp.model$err.rate[nrow(temp.model$err.rate), 1]
              temp_acc[count] <- accuracy
              temp_prec[count] <- precision
              count <- count + 1
            }

            best_idx <- which.min(temp_oob)
            oob.values[i] <- temp_oob[best_idx]
            accuracy.values[i] <- temp_acc[best_idx]
            precision.values[i] <- temp_prec[best_idx]
            ntr.values[i] <- temp_ntr[best_idx]
          }

          # Find optimal parameters
          best_mtry <- which.min(oob.values)
          training_error <- min(oob.values)
          training_accuracy <- accuracy.values[best_mtry]
          training_precision <- precision.values[best_mtry]
          best_ntr <- ntr.values[best_mtry]

          # Store results
          return(data.frame(iteration = n,
                            best_mtry = best_mtry,
                            best_ntr = best_ntr,
                            training_error = training_error,
                            training_accuracy = training_accuracy,
                            training_precision = training_precision))
        }

        # Stop cluster after execution
        stopCluster(cl)
        registerDoSEQ()  # Reset to sequential processing

        #write.csv(RF_benchmark, file = "RF_benchmark.csv", row.names = F)

        # Re-train the model using the best tuned hyper-parameters ---
        # Get benchmark summary
        RF_benchmark <- RF_benchmark %>%
          mutate(Hyperpar_comb = paste0(best_mtry, "-", best_ntr)) %>%
          group_by(Hyperpar_comb) %>%
          mutate(median_accuracy = median(training_accuracy))

        RF_benchmark_summary <- as.data.frame(table(RF_benchmark$Hyperpar_comb), stringsAsFactors = FALSE) %>%
          mutate(Var1 = as.character(Var1)) %>%  # Convert Var1 to integer for matching
          left_join(
            RF_benchmark %>% dplyr::select(Hyperpar_comb, median_accuracy) %>% unique(),
            by = c("Var1" = "Hyperpar_comb")
          )

        RF_best_tunned <- RF_benchmark_summary$Var1[which.max(RF_benchmark_summary$Freq)]
        RF_best_tunned_mtry <- as.numeric(str_split_i(RF_best_tunned, "-", 1))
        RF_best_tunned_ntree <- as.numeric(str_split_i(RF_best_tunned, "-", 2))

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
          optimal_threshold <- 0.5 # Default Threshold
          auroc <- -1 # as a place holder to indicate failure
        }

        # Print final result
        final_benchmark_result <- rbind(final_benchmark_result,
                                        data.frame(Algorithm = "Random Forest",
                                                   Hyperparameter = "mTry-nTree",
                                                   Tuned_Value = RF_best_tunned,
                                                   Optimal_Threshold = round(threshold_value,2),
                                                   Prediction_Accuracy = round(new_conf_matrix$overall["Accuracy"],2),
                                                   Prediction_Precision = round(new_conf_matrix$byClass["Precision"],2),
                                                   Prediction_Recall = round(new_conf_matrix$byClass["Recall"],2),
                                                   Prediction_F1 = round(new_conf_matrix$byClass["F1"],2),
                                                   Prediction_Kappa = round(new_conf_matrix$overall["Kappa"],2),
                                                   AccuracyPValue = round(new_conf_matrix$overall["AccuracyPValue"],2),
                                                   McnemarPValue = round(new_conf_matrix$overall["McnemarPValue"],2),
                                                   AUROC = round(auroc,2)))

        print("Benchmarking Random Forest END")

        # End of Benchmarking Random Forest ---


      } else if (MLmodel == "Naïve Bayes") {
        # Benchmarking Naïve Bayes ---------------------------------------------------------------
        print("Benchmarking Naïve Bayes Start")
        # Set up parallel backend
        cl <- makeCluster(num_cores)
        registerDoParallel(cl)
        # Initialize result storage
        NB_benchmark <- data.frame()
        # Define the tuning grid
        tune_grid <- expand.grid(
          usekernel = c(TRUE, FALSE),
          fL = seq(0, 2, by = 0.5),  # Laplace smoothing
          adjust = seq(0.5, 3, by = 0.5)  # Adjust parameter
        )

        # Run tuning in parallel
        NB_benchmark <- foreach(n = 1:max_tuning_iteration, .combine = rbind, .packages = c("caret", "dplyr", "e1071")) %dopar% {
          # Train Naïve Bayes model
          NB.model <- train(as.formula(paste(Dependency_gene, "~ .")),
                            data = train_df,
                            method = "nb",
                            trControl = ctrlspecs,
                            tuneGrid = tune_grid)


          # Extract best hyperparameters
          best_row_index <- as.numeric(rownames(NB.model$bestTune))
          result_of_bestTune <- NB.model$results[best_row_index, ] %>% mutate(iteration = n)

          return(result_of_bestTune)  # Each iteration returns its best result
        } # end of iteration

        # Stop cluster after execution
        stopCluster(cl)
        registerDoSEQ()  # Reset to sequential execution
        # write out the NB_benchmark result
        #write.csv(NB_benchmark, file = "NB_benchmark.csv", row.names = F)

        # Re-train the model using the best tuned hyper-parameters ---
        # Get benchmark summary
        NB_benchmark <- NB_benchmark %>%
          mutate(Hyperpara_comb = paste0(usekernel, "-", fL, "-",adjust)) %>%
          group_by(Hyperpara_comb) %>%
          mutate(median_accuracy = median(Accuracy))

        NB_benchmark_summary <- as.data.frame(table(NB_benchmark$Hyperpara_comb)) %>%
          left_join(.,
                    NB_benchmark %>% dplyr::select(Hyperpara_comb, median_accuracy) %>% unique(),
                    by = c("Var1" = "Hyperpara_comb"))

        NB_best_tunned <- NB_benchmark_summary$Var1[which.max(NB_benchmark_summary$Freq)]
        NB_best_tunned_usekernel <- as.logical(str_split_i(NB_best_tunned, "-", 1)) # usekernel is either TRUE or FALSE
        NB_best_tunned_fL <- as.numeric(str_split_i(NB_best_tunned, "-", 2))
        NB_best_tunned_adjust <- as.numeric(str_split_i(NB_best_tunned, "-", 3))
        NB_retrained <- data.frame()

        # retrain the NB model
        NB.model <- train(as.formula(paste(Dependency_gene, "~ .")),
                          data = train_df,
                          method = "nb",
                          trControl = ctrlspecs,
                          tuneGrid = expand.grid(
                            usekernel = NB_best_tunned_usekernel,
                            fL = NB_best_tunned_fL,
                            adjust = NB_best_tunned_adjust
                          ))

        NB.model.accuracy <- NB.model$results$Accuracy
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
          optimal_threshold <- 0.5 # Default Threshold
          auroc <- -1 # as a place holder to indicate failure
        }

        # Write final benchmark result
        final_benchmark_result <- rbind(final_benchmark_result,
                                        data.frame(Algorithm = "Naïve Bayes",
                                                   Hyperparameter = "usekernel-fL-adjust",
                                                   Tuned_Value = NB_best_tunned,
                                                   Optimal_Threshold = round(threshold_value,2),
                                                   Prediction_Accuracy = round(new_conf_matrix$overall["Accuracy"],2),
                                                   Prediction_Precision = round(new_conf_matrix$byClass["Precision"],2),
                                                   Prediction_Recall = round(new_conf_matrix$byClass["Recall"],2),
                                                   Prediction_F1 = round(new_conf_matrix$byClass["F1"],2),
                                                   Prediction_Kappa = round(new_conf_matrix$overall["Kappa"],2),
                                                   AccuracyPValue = round(new_conf_matrix$overall["AccuracyPValue"],2),
                                                   McnemarPValue = round(new_conf_matrix$overall["McnemarPValue"],2),
                                                   AUROC = round(auroc,2)))

        print("Benchmarking Naïve Bayes END")
        # End of Benchmarking Naïve Bayes ---


      } else if (MLmodel == "SVM") {
        # Benchmarking SVM ---------------------------------------------------------------
        print("Benchmarking SVM Start")
        cl <- makeCluster(num_cores)
        registerDoParallel(cl)
        kernel_list <- c("linear", "polynomial", "radial", "sigmoid")
        # Initialize an empty dataframe to store results
        SVM_benchmark <- data.frame()
        # Run tuning in parallel
        SVM_benchmark <- foreach(i = 1:max_tuning_iteration, .combine = rbind, .packages = c("e1071", "dplyr")) %dopar% {
          # Function to tune SVM for a given kernel
          tune_svm <- function(k) {
            if (k == "linear") {
              # Linear kernel only requires `cost`
              tune_out <- e1071::tune(e1071::svm,  # Ensure correct SVM reference
                                      as.formula(paste(Dependency_gene, "~ .")),
                                      data = train_df, kernel = k,
                                      ranges = list(cost = 10^seq(-3, 2, by = 1)),
                                      tunecontrol = tune.control(cross = 10))

              return(data.frame(iteration = i, kernel = k,
                                best_cost = tune_out$best.parameters$cost,
                                best_gamma = NA,  # No gamma for linear
                                best_error = tune_out$best.performance))
            } else {
              # Non-linear kernels require both `cost` and `gamma`
              tune_out <- e1071::tune(e1071::svm,  # Ensure correct SVM reference
                                      as.formula(paste(Dependency_gene, "~ .")),
                                      data = train_df, kernel = k,
                                      ranges = list(cost = 10^seq(-3, 2, by = 1),
                                                    gamma = 10^seq(-3, 2, by = 1)),
                                      tunecontrol = tune.control(cross = 10))

              return(data.frame(iteration = i, kernel = k,
                                best_cost = tune_out$best.parameters$cost,
                                best_gamma = tune_out$best.parameters$gamma,
                                best_error = tune_out$best.performance))
            }
          }

          # Run tuning for all kernels in parallel
          iteration_results <- do.call(rbind, lapply(kernel_list, tune_svm))

          # Select the best result (minimum error)
          best_result <- iteration_results[which.min(iteration_results$best_error), ]

          return(best_result)
        }

        # Stop cluster after execution
        stopCluster(cl)
        registerDoSEQ()  # Reset to sequential execution
        # write out the result
        #write.csv(SVM_benchmark, file = "SVM_benchmark.csv", row.names = F)

        # Re-train the model using the best tuned hyper-parameters ---
        # Get benchmark summary
        SVM_benchmark <- SVM_benchmark %>%
          mutate(Hyperparam = paste0(kernel,"-",best_cost, "-",best_gamma)) %>%
          group_by(Hyperparam) %>%
          mutate(mean_error_rate = median(best_error))

        SVM_benchmark_summary <- as.data.frame(table(SVM_benchmark$Hyperparam)) %>%
          left_join(.,
                    SVM_benchmark %>% dplyr::select(Hyperparam,mean_error_rate) %>% unique(),
                    by = c("Var1" = "Hyperparam"))


        SVM_best_tunned <- SVM_benchmark_summary$Var1[which.max(SVM_benchmark_summary$Freq)]
        SVM_best_tunned_kernel <- as.character(str_split_i(SVM_best_tunned, "-", 1))
        SVM_best_tunned_cost <- as.numeric(str_split_i(SVM_best_tunned, "-", 2))
        SVM_best_tunned_gamma <- as.numeric(str_split_i(SVM_best_tunned, "-", 3))
        SVM_retrained <- data.frame()


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
          optimal_threshold <- 0.5 # Default Threshold
          auroc <- -1 # as a place holder to indicate failure
        }

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
                                                   AUROC = round(auroc,2)))

        print("Benchmarking SVM END")
        # End of Benchmarking SVM ---


      } else if (MLmodel == "Elastic Net") {
        # Benchmarking Elastic Net ---------------------------------------------------------------
        print("Benchmarking Elastic Net Start")
        set.seed(101)
        alpha_vector <- seq(0, 1, length=10)
        lambda_vector <- 10^seq(5, -5, length=100)
        tune_grid <- expand.grid(alpha = alpha_vector, lambda = lambda_vector)
        cl <- makeCluster(num_cores)
        registerDoParallel(cl)

        # Initialize an empty dataframe to store results
        ECN_benchmark <- data.frame()

        # Run tuning in parallel
        ECN_benchmark <- foreach(n = 1:max_tuning_iteration, .combine = rbind, .packages = c("caret", "dplyr", "glmnet")) %dopar% {

          print(paste0("Iteration ", n))

          # Train Elastic Net model
          ECN.model <- train(as.formula(paste(Dependency_gene, "~ .")),
                             data = train_df,
                             preProcess = c("center", "scale"),
                             method = "glmnet",
                             tuneGrid = tune_grid,
                             trControl = ctrlspecs,  # 10-fold cross-validation
                             family = "binomial")

          # Extract best hyperparameters
          best_row_index <- as.numeric(rownames(ECN.model$bestTune))
          result_of_bestTune <- ECN.model$results[best_row_index, ] %>% mutate(iteration = n)

          return(result_of_bestTune)
        }

        # Stop cluster after execution
        stopCluster(cl)
        registerDoSEQ()  # Reset to sequential execution
        # Write out
        #write.csv(ECN_benchmark, file = "ECN_benchmark.csv", row.names = F)

        # Re-train the ECN model using the best tuned hyper-parameters ---
        # Get benchmark summary
        ECN_benchmark <- ECN_benchmark %>%
          mutate(Hyperpara_comb = paste0(alpha, "-", lambda)) %>%
          group_by(Hyperpara_comb) %>%
          mutate(median_accuracy = median(Accuracy))

        ECN_benchmark_summary <- as.data.frame(table(ECN_benchmark$Hyperpara_comb)) %>%
          left_join(.,
                    ECN_benchmark %>% dplyr::select(Hyperpara_comb, median_accuracy) %>% unique(),
                    by = c("Var1" = "Hyperpara_comb"))

        ECN_best_tunned <- ECN_benchmark_summary$Var1[which.max(ECN_benchmark_summary$Freq)]
        ECN_best_tunned_alpha <- as.numeric(str_split_i(ECN_best_tunned, "-", 1))
        ECN_best_tunned_lamda <- as.numeric(str_split_i(ECN_best_tunned, "-", 2))

        # Re-train the ECN model
        ECN.model <- train(as.formula(paste(Dependency_gene, "~ .")),
                           data = train_df,
                           preProcess=c("center", "scale"),
                           method = "glmnet",
                           tuneGrid=expand.grid(alpha = ECN_best_tunned_alpha, lambda=ECN_best_tunned_lamda),
                           trControl=ctrlspecs, # Train the model using the 10-fold validation
                           family="binomial")


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
          optimal_threshold <- 0.5 # Default Threshold
          auroc <- -1 # as a place holder to indicate failure
        }

        # Write final benchmark result
        final_benchmark_result <- rbind(final_benchmark_result,
                                        data.frame(Algorithm = "Elastic-Net",
                                                   Hyperparameter = "alpha-lamda",
                                                   Tuned_Value = unlist(ECN_best_tunned),
                                                   Optimal_Threshold = round(threshold_value,2),
                                                   Prediction_Accuracy = round(new_conf_matrix$overall["Accuracy"],2),
                                                   Prediction_Precision = round(new_conf_matrix$byClass["Precision"],2),
                                                   Prediction_Recall = round(new_conf_matrix$byClass["Recall"],2),
                                                   Prediction_F1 = round(new_conf_matrix$byClass["F1"],2),
                                                   Prediction_Kappa = round(new_conf_matrix$overall["Kappa"],2),
                                                   AccuracyPValue = round(new_conf_matrix$overall["AccuracyPValue"],2),
                                                   McnemarPValue = round(new_conf_matrix$overall["McnemarPValue"],2),
                                                   AUROC = round(auroc,2)))

        print("Benchmarking ECN END")
        # End of Benchmarking ECN ---


      } else if (MLmodel == "KNN") {
        # Benchmarking KNN ---------------------------------------------------------------
        print("Benchmarking KNN Start")
        KNN_benchmark <- data.frame()
        for (n in 1:max_tuning_iteration){
          print(paste0("Iteration ",n))
          metric <- "Accuracy"
          grid <- expand.grid(.k=seq(1,20,by=1))
          KNN.model <- train(as.formula(paste(Dependency_gene, "~ .")),
                             data= train_df,
                             method="knn",
                             metric=metric,
                             tuneGrid=grid,
                             trControl=ctrlspecs,
                             na.action = na.omit)
          row_index_of_bestTune <- rownames(KNN.model$bestTune)
          result_of_bestTune <- KNN.model$results[row_index_of_bestTune, ]
          result_of_bestTune <- result_of_bestTune %>% mutate(iteration = n)
          # Store result
          KNN_benchmark <- rbind(KNN_benchmark, result_of_bestTune)
        } # End of 1000 iteration
        #write.csv(KNN_benchmark, file = "KNN_benchmark.csv", row.names = F)

        # Re-train the KNN model using the best tuned hyper-parameters ---
        # Get benchmark summary
        KNN_benchmark <- KNN_benchmark %>%
          group_by(k) %>%
          mutate(median_accuracy = median(Accuracy))

        KNN_benchmark_summary <- as.data.frame(table(KNN_benchmark$k)) %>%
          mutate(Var1 = as.numeric(as.character(Var1))) %>%  # Convert Var1 to numeric
          left_join(.,
                    KNN_benchmark %>% dplyr::select(k, median_accuracy) %>% unique(),
                    by = c("Var1" = "k"))

        KNN_best_tunned <- KNN_benchmark_summary$Var1[which.max(KNN_benchmark_summary$Freq)]
        KNN_best_tunned_k <- as.numeric(KNN_best_tunned)

        # Re-train the KNN model
        KNN.model <- train(as.formula(paste(Dependency_gene, "~ .")),
                           data= train_df,
                           method="knn",
                           metric=metric,
                           tuneGrid=expand.grid(.k=KNN_best_tunned_k),
                           trControl=ctrlspecs,
                           na.action = na.omit)


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
          optimal_threshold <- 0.5 # Default Threshold
          auroc <- -1 # as a place holder to indicate failure
        }

        # Write final benchmark result
        final_benchmark_result <- rbind(final_benchmark_result,
                                        data.frame(Algorithm = "KNN",
                                                   Hyperparameter = "k",
                                                   Tuned_Value = KNN_best_tunned_k,
                                                   Optimal_Threshold = round(threshold_value,2),
                                                   Prediction_Accuracy = round(new_conf_matrix$overall["Accuracy"],2),
                                                   Prediction_Precision = round(new_conf_matrix$byClass["Precision"],2),
                                                   Prediction_Recall = round(new_conf_matrix$byClass["Recall"],2),
                                                   Prediction_F1 = round(new_conf_matrix$byClass["F1"],2),
                                                   Prediction_Kappa = round(new_conf_matrix$overall["Kappa"],2),
                                                   AccuracyPValue = round(new_conf_matrix$overall["AccuracyPValue"],2),
                                                   McnemarPValue = round(new_conf_matrix$overall["McnemarPValue"],2),
                                                   AUROC = round(auroc,2)))



        print("Benchmarking KNN END")
        # End of Benchmarking KNN ---


      } else if (MLmodel == "Neural Network") {
        # Benchmarking Neural Network ---------------------------------------------------------------
        print("Benchmarking Neural Network Start")
        # Set up parallel backend
        cl <- makeCluster(num_cores)
        registerDoParallel(cl)
        NeurNet_benchmark <- data.frame()
        grid_tune <- expand.grid(
          size = c(1:10),       # Number of hidden neurons
          decay = c(0.001, 0.01, 0.1)  # Regularization values
        )
        # Run tuning in parallel
        NeurNet_benchmark <- foreach(n = 1:max_tuning_iteration, .combine = rbind, .packages = c("caret", "dplyr", "nnet")) %dopar% {
          # Train Neural Network model
          NNET.model <- train(as.formula(paste(Dependency_gene, "~ .")),
                              data = train_df,
                              method = "nnet",
                              trControl = ctrlspecs,
                              tuneGrid = grid_tune,
                              linout = FALSE,  # Use linout = TRUE for regression
                              na.action = na.omit,
                              trace = FALSE)  # Suppress training output

          # Extract best hyperparameters
          best_row_index <- as.numeric(rownames(NNET.model$bestTune))
          result_of_bestTune <- NNET.model$results[best_row_index, ] %>% mutate(iteration = n)

          return(result_of_bestTune)
        }

        # Stop cluster after execution
        stopCluster(cl)
        registerDoSEQ()  # Reset to sequential execution
        # Write out
        #write.csv(NeurNet_benchmark, file = "NeurNet_benchmark.csv", row.names = F)

        # Re-train the NeurNet model using the best tuned hyper-parameters ---
        # Get benchmark summary
        NeurNet_benchmark <- NeurNet_benchmark %>%
          mutate(Hyperpara_comb = paste0(size, "-", decay)) %>%
          group_by(Hyperpara_comb) %>%
          mutate(median_accuracy = median(Accuracy))

        NeurNet_benchmark_summary <- as.data.frame(table(NeurNet_benchmark$Hyperpara_comb)) %>%
          mutate(Var1 = as.character(Var1)) %>%  # Convert Var1 to numeric
          left_join(.,
                    NeurNet_benchmark %>% dplyr::select(Hyperpara_comb, median_accuracy) %>% unique(),
                    by = c("Var1" = "Hyperpara_comb"))

        NeurNet_best_tunned <- NeurNet_benchmark_summary$Var1[which.max(NeurNet_benchmark_summary$Freq)]
        NeurNet_best_tunned_size <- as.numeric(str_split_i(NeurNet_best_tunned, "-", 1))
        NeurNet_best_tunned_decay <- as.numeric(str_split_i(NeurNet_best_tunned, "-", 2))


        # Re-train the NeurNet model
        NeurNet.model <- train(as.formula(paste(Dependency_gene, "~ .")),
                               data = train_df,
                               method = "nnet",
                               trControl = ctrlspecs,
                               tuneGrid = expand.grid(
                                 size = NeurNet_best_tunned_size,       # Number of hidden neurons
                                 decay = NeurNet_best_tunned_decay  # Regularization values
                               ),
                               linout = FALSE, # Use linout = TRUE for Regression
                               na.action = na.omit,
                               trace = FALSE) # Suppress output during training


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
          optimal_threshold <- 0.5 # Default Threshold
          auroc <- -1 # as a place holder to indicate failure
        }

        # Write final benchmark result
        final_benchmark_result <- rbind(final_benchmark_result,
                                        data.frame(Algorithm = "Neural Network",
                                                   Hyperparameter = "size-decay",
                                                   Tuned_Value = NeurNet_best_tunned,
                                                   Optimal_Threshold = round(threshold_value,2),
                                                   Prediction_Accuracy = round(new_conf_matrix$overall["Accuracy"],2),
                                                   Prediction_Precision = round(new_conf_matrix$byClass["Precision"],2),
                                                   Prediction_Recall = round(new_conf_matrix$byClass["Recall"],2),
                                                   Prediction_F1 = round(new_conf_matrix$byClass["F1"],2),
                                                   Prediction_Kappa = round(new_conf_matrix$overall["Kappa"],2),
                                                   AccuracyPValue = round(new_conf_matrix$overall["AccuracyPValue"],2),
                                                   McnemarPValue = round(new_conf_matrix$overall["McnemarPValue"],2),
                                                   AUROC = round(auroc,2)))

        print("Benchmarking Neural Network END")
        # End of Benchmarking Neural Network ---


      } else if (MLmodel == "AdaBoost") {
        # Benchmarking AdaBoost ---------------------------------------------------------------
        print("Benchmarking AdaBoost Start")
        AdaBoost_benchmark <- data.frame()
        grid_tune <- expand.grid(
          mfinal = seq(50, 150, by = 10),
          maxdepth = c(1,2,3,4),
          coeflearn = c("Breiman", "Freund", "Zhu")
        )
        # Set up parallel backend
        cl <- makeCluster(num_cores)
        registerDoParallel(cl)

        # Initialize storage for results
        AdaBoost_benchmark <- data.frame()

        # Run AdaBoost tuning in parallel
        AdaBoost_benchmark <- foreach(n = 1:max_tuning_iteration, .combine = rbind, .packages = c("caret", "dplyr")) %dopar% {

          print(paste0("Iteration ", n))

          # Train the Adaboost model
          AdaBoost.model <- train(as.formula(paste(Dependency_gene, "~ .")),
                                  data = train_df,
                                  method = "AdaBoost.M1",
                                  tuneGrid = grid_tune,
                                  trControl = ctrlspecs)

          row_index_of_bestTune <- rownames(AdaBoost.model$bestTune)
          result_of_bestTune <- AdaBoost.model$results[row_index_of_bestTune, ]
          result_of_bestTune <- result_of_bestTune %>% mutate(iteration = n)

          # Return result for foreach to combine
          result_of_bestTune
        }

        # Stop parallel cluster
        stopCluster(cl)
        registerDoSEQ()  # Reset to sequential processing
        # write out the AdaBoost_benchmark result
        #write.csv(AdaBoost_benchmark, file = "AdaBoost_benchmark.csv", row.names = F)

        # Re-train the AdaBoost model using the best tuned hyper-parameters ---
        # Get benchmark summary
        AdaBoost_benchmark <- AdaBoost_benchmark %>%
          mutate(Hyperpara_comb = paste0(coeflearn, "-", maxdepth, "-", mfinal)) %>%
          group_by(Hyperpara_comb) %>%
          mutate(median_accuracy = median(Accuracy))

        AdaBoost_benchmark_summary <- as.data.frame(table(AdaBoost_benchmark$Hyperpara_comb)) %>%
          mutate(Var1 = as.character(Var1)) %>%  # Convert Var1 to numeric
          left_join(.,
                    AdaBoost_benchmark %>% dplyr::select(Hyperpara_comb, median_accuracy) %>% unique(),
                    by = c("Var1" = "Hyperpara_comb"))

        AdaBoost_best_tunned <- AdaBoost_benchmark_summary$Var1[which.max(AdaBoost_benchmark_summary$Freq)]
        AdaBoost_best_tunned_coeflearn <- as.character(str_split_i(AdaBoost_best_tunned, "-", 1))
        AdaBoost_best_tunned_maxdepth <- as.numeric(str_split_i(AdaBoost_best_tunned, "-", 2))
        AdaBoost_best_tunned_mfinal <- as.numeric(str_split_i(AdaBoost_best_tunned, "-", 3))

        # Re-train the AdaBoost model
        AdaBoost.model <- train(as.formula(paste(Dependency_gene, "~ .")),
                                data = train_df,
                                method = "AdaBoost.M1",
                                tuneGrid = expand.grid(
                                  mfinal = AdaBoost_best_tunned_mfinal,
                                  maxdepth = AdaBoost_best_tunned_maxdepth,
                                  coeflearn = AdaBoost_best_tunned_coeflearn
                                ),
                                trControl = ctrlspecs)

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
          optimal_threshold <- 0.5 # Default Threshold
          auroc <- -1 # as a place holder to indicate failure
        }

        # Write final benchmark result
        final_benchmark_result <- rbind(final_benchmark_result,
                                        data.frame(Algorithm = "AdaBoost",
                                                   Hyperparameter = "coeflearn-maxdepth-mfinal",
                                                   Tuned_Value = AdaBoost_best_tunned,
                                                   Optimal_Threshold = round(threshold_value,2),
                                                   Prediction_Accuracy = round(new_conf_matrix$overall["Accuracy"],2),
                                                   Prediction_Precision = round(new_conf_matrix$byClass["Precision"],2),
                                                   Prediction_Recall = round(new_conf_matrix$byClass["Recall"],2),
                                                   Prediction_F1 = round(new_conf_matrix$byClass["F1"],2),
                                                   Prediction_Kappa = round(new_conf_matrix$overall["Kappa"],2),
                                                   AccuracyPValue = round(new_conf_matrix$overall["AccuracyPValue"],2),
                                                   McnemarPValue = round(new_conf_matrix$overall["McnemarPValue"],2),
                                                   AUROC = round(auroc,2)))

        print("Benchmarking AdaBoost END")
        # End of Benchmarking AdaBoost ---



      } else if (MLmodel == "XGBoost") {
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
        search_grid <- expand.grid(
          max_depth = c(2,4,6),
          eta = c(0.025,0.05,0.1,0.3), #Learning rate
          gamma = c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0), # pruning --> Should be tuned. i.e c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0)
          colsample_bytree = c(0.4, 0.6, 0.8, 1.0), # c(0.4, 0.6, 0.8, 1.0) subsample ratio of columns for tree
          min_child_weight = c(1,2,3), # c(1,2,3) # the larger, the more conservative the model
          #is; can be used as a stop
          subsample = c(0.5, 0.75, 1.0) # c(0.5, 0.75, 1.0) # used to prevent overfitting by sampling X% training
        )

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

          # Extract accuracy from XGBoost CV results
          if (!is.null(cv_results$evaluation_log)) {

            # Extract test error (misclassification rate)
            error_values <- cv_results$evaluation_log$test_error_mean

            if (!is.null(error_values) && any(!is.na(error_values) & !is.nan(error_values))) {

              # Convert error rate to accuracy
              accuracy_values <- 1 - error_values
              mean_accuracy <- max(accuracy_values, na.rm = TRUE)  # Higher accuracy is better

              if (!is.na(mean_accuracy) && mean_accuracy != -Inf && mean_accuracy > best_accuracy) {
                best_accuracy <- mean_accuracy
                best_params <- params
              }
            } else {
              warning("Warning: test_error_mean contains only NA or NULL values, skipping comparison.")
            }
          }

        }

        best_nrounds <- cv_results$best_iteration

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
          optimal_threshold <- 0.5 # Default Threshold
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
                                                   AUROC = round(auroc,2)))

        print("Benchmarking XGBoost END")
        # End of Benchmarking XGBoost ---


      } else if (MLmodel == "Decision Tree") {
        # Benchmarking Decision Tree ---------------------------------------------------------------
        print("Benchmarking Decision Tree Start")
        Decision_Tree_benchmark <- data.frame()
        grid_tune <- expand.grid(
          cp = seq(0.01, 0.1, 0.01)
        )
        for (n in 1:max_tuning_iteration){
          print(paste0("Iteration ",n))
          Decision_Tree.tuned <- train(
            as.formula(paste(Dependency_gene, "~ .")),
            data = train_df,
            method = "rpart",
            trControl = ctrlspecs,
            tuneGrid = grid_tune)

          row_index_of_bestTune <- rownames(Decision_Tree.tuned$bestTune)
          result_of_bestTune <- Decision_Tree.tuned$results[row_index_of_bestTune, ]
          result_of_bestTune <- result_of_bestTune %>% mutate(iteration = n)
          # Store result
          Decision_Tree_benchmark <- rbind(Decision_Tree_benchmark, result_of_bestTune)
        } # End of 1000 iteration
        # Write out
        #write.csv(Decision_Tree_benchmark, file = "Decision_Tree_benchmark.csv", row.names = F)


        # Re-train the model using the best tuned hyper-parameters ---
        # Get benchmark summary
        Decision_Tree_benchmark <- Decision_Tree_benchmark %>%
          group_by(cp) %>%
          mutate(median_accuracy = median(Accuracy))

        Decision_Tree_benchmark_summary <- as.data.frame(table(Decision_Tree_benchmark$cp)) %>%
          mutate(Var1 = as.numeric(as.character(Var1))) %>%
          left_join(.,
                    Decision_Tree_benchmark %>% dplyr::select(cp, median_accuracy) %>% unique(),
                    by = c("Var1" = "cp"))

        Decision_Tree_best_tunned_cp <- Decision_Tree_benchmark_summary$Var1[which.max(Decision_Tree_benchmark_summary$Freq)]


        # Re-train the model
        Decision_Tree.model <- train(
          as.formula(paste(Dependency_gene, "~ .")),
          data = train_df,
          method = "rpart",
          trControl = ctrlspecs,
          tuneGrid = expand.grid(cp = Decision_Tree_best_tunned_cp)
        )

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
          optimal_threshold <- 0.5 # Default Threshold
          auroc <- -1 # as a place holder to indicate failure
        }

        # Write final benchmark result
        final_benchmark_result <- rbind(final_benchmark_result,
                                        data.frame(Algorithm = "Decision Tree",
                                                   Hyperparameter = "cp",
                                                   Tuned_Value = Decision_Tree_best_tunned_cp,
                                                   Optimal_Threshold = round(threshold_value,2),
                                                   Prediction_Accuracy = round(new_conf_matrix$overall["Accuracy"],2),
                                                   Prediction_Precision = round(new_conf_matrix$byClass["Precision"],2),
                                                   Prediction_Recall = round(new_conf_matrix$byClass["Recall"],2),
                                                   Prediction_F1 = round(new_conf_matrix$byClass["F1"],2),
                                                   Prediction_Kappa = round(new_conf_matrix$overall["Kappa"],2),
                                                   AccuracyPValue = round(new_conf_matrix$overall["AccuracyPValue"],2),
                                                   McnemarPValue = round(new_conf_matrix$overall["McnemarPValue"],2),
                                                   AUROC = round(auroc,2)))

        print("Benchmarking Decision Tree END")
        # End of Benchmarking Decision Tree ---


      } else {
        print("Not a avaible model")
      }
    }

    print("Writing out")

    final_benchmark_result <- final_benchmark_result %>%
      mutate(max_tuning_iteration = max_tuning_iteration,
             gene_hits_percentage_cutoff = cutoff)

    #write.csv(final_benchmark_result,
    #          file = final_benchmark_result_write_out_filename,
    #          row.names = F)
    assign("final_benchmark_result", final_benchmark_result, envir = .GlobalEnv)  # Save in global env
  }
}
