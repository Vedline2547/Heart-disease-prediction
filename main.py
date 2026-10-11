# Import necessary libraries
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv(r"C:\Users\HP\Desktop\ML\Heart-disease\heart.csv")
# Display first few rows
print("First five rows:")
print(df.head())
# Check missing values
print("\nMissing values:")
print(df.isnull().sum())
# Handle missing target values
df = df.dropna(subset=['target'])
# Separate features and target
X = df.drop('target', axis=1)
y = df['target']
# Split dataset before preprocessing to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
# Logistic Regression pipeline
log_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Random Forest pipeline
rf_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])
# Train Logistic Regression
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)
log_accuracy = accuracy_score(y_test, log_preds)
print(f"\nLogistic Regression Accuracy: {log_accuracy:.4f}")
# Train Random Forest
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
# Select the best model based on test accuracy
if rf_accuracy > log_accuracy:
    best_model = rf_model
    best_preds = rf_preds
    best_model_name = "Random Forest"
else:
    best_model = log_model
    best_preds = log_preds
    best_model_name = "Logistic Regression"

# Evaluate selected model
print(f"\nBest Model: {best_model_name}")
print(f"Accuracy Score: {accuracy_score(y_test, best_preds):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, best_preds, zero_division=0))
# Confusion matrix
cm = confusion_matrix(y_test, best_preds)
print("\nConfusion Matrix:")
print(cm)
# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm,annot=True,cmap='Blues',fmt='d')
plt.title(f'{best_model_name} - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Make predictions on new patient data
new_data = pd.DataFrame({
    'age': [45],
    'sex': [1],
    'cp': [2],
    'trestbps': [130],
    'chol': [230],
    'fbs': [0],
    'restecg': [1],
    'thalach': [150],
    'exang': [0],
    'oldpeak': [0.5],
    'slope': [2],
    'ca': [0],
    'thal': [2]
})

# Ensure new data uses the same feature order as training data
new_data = new_data[X.columns]

# Predict using the selected pipeline
prediction = best_model.predict(new_data)

print("\nPrediction for new patient:")
print(
    "Model predicts heart disease"
    if prediction[0] == 1
    else "Model predicts no heart disease"
)
