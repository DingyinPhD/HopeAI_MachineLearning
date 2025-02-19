\name{model_benchmark}
\alias{model_benchmark}
\title{For Machine Learning model benchmarking}
\usage{
model_benchmark(Methylation_gene,
                Dependency_gene,
                Methylation_betascore_path,
                D300V_data_path,
                max_tuning_iteration = 100,
                fold = 10, # k-fold cross validation
                model = c("Random Forest", "Naïve Bayes", "Elastic Net", "SVM",
                          "XGBoost", "AdaBoost", "Neural Network", "KNN", "Decision Tree"))
}
\description{
Benchmark 9 popular machine learning algorithms in cancer research, including "Random Forest", "Naïve Bayes", "Elastic Net", "SVM",
                  "XGBoost", "AdaBoost", "Neural Network", "KNN", "Decision Tree"
}
\arguments{
  \item{Methylation_betascore_path}{File path to the beta score data for the specified \code{Methylation_gene}.}
  \item{D300V_data_path}{File path to the D300V dataset.}
  \item{max_tuning_iteration}{Number of iterations used for hyperparameter tuning in model training.}
  \item{fold}{Number of folds for cross-validation.}
  \item{model}{Vector value indicating which algorithm to be benchmarked.}
}
\author{
Dingyin Sun
}

\examples{
model_benchmark(Methylation_gene = "ITPRIP",
                Dependency_gene = "SDHA",
                Methylation_betascore_path = "/Users/dingyinsun/Desktop/Machine_Learning/Practice/ITPRIP_AllCpG_betascore.csv",
                D300V_data_path = "/Users/dingyinsun/Desktop/Dataset/ngpmp/NGPMP_CRISPR_data/zscoreMean_23batch_062024_transposed.csv",
                max_tuning_iteration = 100,
                fold = 10,
                model = c("Random Forest", "Naïve Bayes", "Elastic Net", "SVM",
                          "XGBoost", "AdaBoost", "Neural Network", "KNN", "Decision Tree"))
}
