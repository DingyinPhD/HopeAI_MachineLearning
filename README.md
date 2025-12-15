### Model Benchmark Versions

<details>
<summary><strong>V2</strong> — Basic 80/20 split</summary>
Randomly partitions the dataset into 80% training and 20% testing.
</details>

<details>
<summary><strong>V3</strong> — Repeated 80/20 splits</summary>
Repeats random 80/20 partitions until all samples have served as test data (20%).
</details>

<details>
<summary><strong>V4</strong> — No test split</summary>
All data used as training data.
</details>

<details>
<summary><strong>V5</strong> — Cross-validated SHAP</summary>
Runs kernelshap() inside each CV fold and aggregates SHAP values; currently implemented for Elastic Net.
</details>

<details>
<summary><strong>V6</strong> — Strict Nested Cross-Validation + SHAP</summary>
Performs full nested CV for regression and classification, tunes hyperparameters only within inner folds, evaluates on untouched outer folds, saves metrics, tuned parameters, and aggregates SHAP explanations.
</details>

<details>
<summary><strong>V7</strong> — Extended Nested CV + Train-set SHAP</summary>
Same as V6, but also computes SHAP values on training sets in each fold. Based on methodology from “Explanations of ML Models in Repeated Nested Cross-Validation…”.
</details>
