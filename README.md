### Model Benchmarking Workflow Versions ###
#### Basic Splitting Strategies ####

V2: Single random 80/20 split

V3: Repeated 80/20 splits until full test-set coverage

V4: No test split; full dataset used for training

#### SHAP-Integrated Cross-Validation ####

V5: SHAP computed within each CV fold (Elastic Net implemented)

#### Nested Cross-Validation Pipelines ####

V6: Full nested CV (inner hyperparameter tuning, outer unbiased test evaluation); per-fold SHAP

V7: Same as V6 + SHAP on training sets in each fold; methodology aligned with Fig. 1 from “Explanations of ML Models in Repeated Nested Cross-Validation…”
