\name{model_benchmark}
\alias{model_benchmark}
\title{For Machine Learning model benchmarking}
\usage{
model_benchmark(Features,
                Target,
                Input_Data,
                max_tuning_iteration = 100,
                fold = 10, # k-fold cross validation
                model = c("Random Forest", "Naïve Bayes", "Elastic Net", "SVM",
                          "XGBoost", "AdaBoost", "Neural Network", "KNN", "Decision Tree"),
                model_type = "Classification")  # or "Regression"
}
\description{
Benchmark 9 popular machine learning algorithms in cancer research, including "Random Forest", "Naïve Bayes", "Elastic Net", "SVM",
                  "XGBoost", "AdaBoost", "Neural Network", "KNN", "Decision Tree"
}
\arguments{
  \item{Features}{A list specifying the input features (independent variables) used for model training.}
  \item{Target}{The dependent variable (output) that the model aims to predict.}
  \item{Input_Data}{The complete dataset containing both features and the target variable.}
  \item{max_tuning_iteration}{Specifies the maximum number of tuning iterations for hyperparameter optimization.}
  \item{fold}{Number of k-folds used in cross-validation.}
  \item{model}{Vector value indicating which algorithm to be benchmarked.}
  \item{model_type}{Defines the type of predictive modeling task.}
}
\author{
Dingyin Sun
}
\examples{
model_benchmark(Features = "ITPRIP",
                Target = "SDHA",
                Input_Data = "input_data.csv",
                max_tuning_iteration = 100,
                fold = 10,
                model = c("Random Forest", "Naïve Bayes", "Elastic Net", "SVM",
                          "XGBoost", "AdaBoost", "Neural Network", "KNN", "Decision Tree"),
                model_type = "Classification")
}
