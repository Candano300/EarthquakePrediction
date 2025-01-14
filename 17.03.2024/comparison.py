import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

data = pd.read_csv('/Users/Hazal/Desktop/MATH482/LAeq_fulltrain.csv' )

X = data.drop(columns='class')
y = data['class'] - 1  

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ================= Logistic Regression =================
log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=5000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_logreg = log_reg.predict(X_val_scaled)
print("Logistic Regression (Scikit-learn):")
print(classification_report(y_val, y_pred_logreg))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_logreg))
print("Accuracy:", accuracy_score(y_val, y_pred_logreg))

# ================= Logistic Regression (PyTorch) =================
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

input_dim = X_train_tensor.shape[1]
num_classes = len(y.unique())
model_logreg = LogisticRegressionModel(input_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_logreg.parameters(), lr=0.01)

# Training Logistic Regression (PyTorch)
for epoch in range(2000):
    model_logreg.train()
    optimizer.zero_grad()
    outputs = model_logreg(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

model_logreg.eval()
with torch.no_grad():
    val_outputs = model_logreg(X_val_tensor)
    _, val_preds = torch.max(val_outputs, 1)
    val_acc_logreg = (val_preds == y_val_tensor).sum().item() / y_val_tensor.size(0)
    print("Logistic Regression (PyTorch) Accuracy:", val_acc_logreg)

# ================= Single Layer Perceptron =================
class SLPModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SLPModel, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

model_slp = SLPModel(input_dim, num_classes)
optimizer_slp = optim.Adam(model_slp.parameters(), lr=0.01)

for epoch in range(500):
    model_slp.train()
    optimizer_slp.zero_grad()
    outputs_slp = model_slp(X_train_tensor)
    loss_slp = criterion(outputs_slp, y_train_tensor)
    loss_slp.backward()
    optimizer_slp.step()

model_slp.eval()
with torch.no_grad():
    val_outputs_slp = model_slp(X_val_tensor)
    _, val_preds_slp = torch.max(val_outputs_slp, 1)
    val_acc_slp = (val_preds_slp == y_val_tensor).sum().item() / y_val_tensor.size(0)
    print("Single Layer Perceptron (SLP) Accuracy:", val_acc_slp)

# ================= Multi Layer Perceptron (MLP) =================
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model_mlp = MLPModel(input_dim, hidden_dim=64, num_classes=num_classes)
optimizer_mlp = optim.Adam(model_mlp.parameters(), lr=0.01)

for epoch in range(500):
    model_mlp.train()
    optimizer_mlp.zero_grad()
    outputs_mlp = model_mlp(X_train_tensor)
    loss_mlp = criterion(outputs_mlp, y_train_tensor)
    loss_mlp.backward()
    optimizer_mlp.step()

model_mlp.eval()
with torch.no_grad():
    val_outputs_mlp = model_mlp(X_val_tensor)
    _, val_preds_mlp = torch.max(val_outputs_mlp, 1)
    val_acc_mlp = (val_preds_mlp == y_val_tensor).sum().item() / y_val_tensor.size(0)
    print("Multi Layer Perceptron (MLP) Accuracy:", val_acc_mlp)

# ================= Comparison Summary =================
print("\nModel Performance Summary:")
print("Logistic Regression (Scikit-learn):", accuracy_score(y_val, y_pred_logreg))
print("Logistic Regression (PyTorch):", val_acc_logreg)
print("Single Layer Perceptron (SLP):", val_acc_slp)
print("Multi Layer Perceptron (MLP):", val_acc_mlp)
