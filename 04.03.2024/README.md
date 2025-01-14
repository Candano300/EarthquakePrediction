## Summary Sheet
### Methodology
- **Data Split:** Training (60%), Validation (20%), Test (20%)
- **Balancing:** Applied **SMOTE** .
- **Model:** Random Forest with **GridSearchCV** for hyperparameter tuning.

###  Hyperparameters
- `n_estimators`: **200**
- `max_depth`: **30**
- `min_samples_split`: **2**
- `min_samples_leaf`: **1**
- `bootstrap`: **False**

### Results
- Evaluated with **Precision**, **Recall**, **F1-Score**, and **Confusion Matrix**.
