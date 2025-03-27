\name{model_benchmark}
\alias{model_benchmark}
\title{For Machine Learning model benchmarking}
\usage{
model_benchmark(Features,
                Target,
                Input_Data,
                dependency_threshold = -1.5,
                gene_hits_percentage_cutoff_Lower = 0.2,
                gene_hits_percentage_cutoff_Upper = 0.8,
                max_tuning_iteration = 50,
                fold = 10,
                model = c("Random Forest", "Naïve Bayes", "Elastic Net", "SVM",
                          "XGBoost", "AdaBoost", "Neural Network", "KNN", "Decision Tree"),
                XBoost_tuning_grid = "Simple",
                model_type = "Classification")  # or "Regression" (haven't implemented yet)
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
  \item{dependency_threshold}{Gene dependency cutoff. D300V: -1.5, WG: -0.5}
  \item{gene_hits_percentage_cutoff}{Cutoff for gene hits percentage. `gene_hits_percentage_cutoff_Lower = 0.2` means will exclude gene with hits percentage < 0.2. }
  \item{fold}{Number of k-folds used in cross-validation.}
  \item{model}{Vector value indicating which algorithm to be benchmarked.}
  \item{XBoost_tuning_grid}{Choose between "Simple" or "Fine".}
  \item{model_type}{Defines the type of predictive modeling task.}
}
\author{
Dingyin Sun
}
\examples{
model_benchmark(
      Features = "place_holder", # Just a placeholder, can be anything
      Target = "SDHA",
      Input_Data = "data.csv",
      dependency_threshold = -1.5,
      gene_hits_percentage_cutoff_Lower = 0.2,
      gene_hits_percentage_cutoff_Upper = 0.8,
      max_tuning_iteration = 50,
      fold = 10,
      XBoost_tuning_grid = "Simple",
      model = c("Random Forest", "Naïve Bayes", "Elastic Net", "SVM",
                "XGBoost", "AdaBoost", "Neural Network", "KNN", "Decision Tree")
    )
}
