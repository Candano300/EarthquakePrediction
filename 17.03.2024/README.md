## Summary Sheet

### 1. Logistic Regression (Scikit-learn)
- **Hyperparameters:**
  - `multi_class='multinomial'`
  - `solver='lbfgs'`
  - `max_iter=5000`
  - `class_weight='balanced'`
- **Data Processing:**
  - `StandardScaler`
  -  **SMOTE**

### 2. Logistic Regression (PyTorch)
- A single linear layer is used.
- **Optimizer:** `Adam (lr=0.01)`
- **Loss Function:** `CrossEntropyLoss`

### 3. Single Layer Perceptron (SLP)
- Implemented with PyTorch.
- **Activation:** Linear (No hidden layer)
- **Optimizer:** `Adam (lr=0.01)`

### 4. Enhanced Multi-Layer Perceptron (MLP)
- **Architecture:**
  - Input Layer → Hidden Layer (128 units, **LeakyReLU**) → Dropout → Hidden Layer (64 units) → Output Layer
- **Regularization:** Dropout (`p=0.5`)
- **Optimizer:** `AdamW (lr=0.001)`
- **Learning Rate Scheduler:** Step decay (`gamma=0.5` every 50 epochs)

### Results
| **Model**                    | **Accuracy** |
|------------------------------|--------------|
| Logistic Regression (Sklearn) | ~33.75%      |
| Logistic Regression (PyTorch) | ~33.75%      |
| Single Layer Perceptron (SLP) | ~33.78%      |
| **Enhanced Multi-Layer Perceptron (MLP)** | **71.22%**  |



