**modelbenchmark_V2**: Randomly partition(80-20) at the start.

**modelbenchmark_V3**: With random 80-20 partition but repeat this multiple times until all data point had a chance to become testing datasets (20%).

**modelbenchmark_V4**: No random 80-20 partition at the start, all data becomes training datasets.

**modelbenchmark_V5**: Cross-validated SHAP, adding kernalshap() into each fold of cross-validation, currently only implemented in Elastic-Net

**modelbenchmark_V5**: This script implements strict nested cross-validation for both regression and classification tasks, ensuring an unbiased estimate of model performance by tuning hyperparameters only within the inner folds and evaluating them on untouched outer folds. It supports multiple models via caret, automatically saves per-fold metrics and tuned parameters, and integrates model-agnostic SHAP explanations (using kernelshap + shapviz) to provide both per-fold and aggregated feature importance summaries, along with visualization outputs for the final refit model trained on the full dataset.

Run SHAP value computation within each CV fold instead of on the entire dataset at once, to avoid data leakage.

Aggregate SHAP values across folds (e.g., averaging or ranking) to get a more robust estimate of feature importance.
