import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
data = pd.read_csv('/Users/Hazal/Desktop/MATH482/LAeq_fulltrain.csv'  )

X = data.drop(columns='class')
y = data['class']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)  # 0.25 x 0.8 = 0.2

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Random Forest model with hyperparameters matching the paper
param_grid = {
    'n_estimators': [200],  
    'max_depth': [30],     
    'min_samples_split': [2],  
    'min_samples_leaf': [1],  
    'bootstrap': [False]      
}
rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train_resampled, y_train_resampled)

best_rf = grid_search.best_estimator_

y_val_pred = best_rf.predict(X_val)
print("Validation Results:")
print(classification_report(y_val, y_val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
print("Accuracy:", accuracy_score(y_val, y_val_pred))

joblib.dump(best_rf, 'best_rf_model.pkl')

def predict_new(data_series):
    model = joblib.load('best_rf_model.pkl')
    prediction = model.predict([data_series])
    return prediction

