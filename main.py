# Customer Churn Prediction Project
# Author: PIYUSH KUMAR YADAV

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

print("Libraries Imported Successfully")

# Load Dataset
df = pd.read_csv("dataset.csv")
print("Dataset Loaded Successfully")

import os
print(os.listdir())
print("\nFirst 5 Rows")
print(df.head())

print("\nLast 5 Rows")
print(df.tail())

print("\nShape of Dataset")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

print("\nStatistical Summary (Including Categorical)")
print(df.describe(include="all"))

# check missing valu
print("\nMissing Values")
print(df.isnull().sum())
# check duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates()

# check data type and target variable
print(df.dtypes)
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True) * 100)

# visualize target variable
plt.figure(figsize=(6,4))
sns.countplot(x="Churn", data=df)
plt.title("Customer Churn Distribution")

import os
os.makedirs("images", exist_ok=True)
plt.savefig("images/graph_name.png")
plt.show()

# data cleaning
print("Missing Values:")
print(df.isnull().sum())
print("Duplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)
print(df.dtypes)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce") # convert total charge to numeric
print(df.isnull().sum()) # check missing value 

df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True) # fill missing value
print(df.isnull().sum())

# remove customer id
df.drop("customerID", axis=1, inplace=True)
print(df.head())

# -----EDA---

# churn distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Churn", data=df)
plt.title("Customer Churn Distribution")

import os
os.makedirs("images", exist_ok=True)
plt.savefig("images/graph_name.png")
plt.show()

# gender distribution
plt.figure(figsize=(6,4))
sns.countplot(x="gender", data=df)
plt.title("Gender Distribution")

import os
os.makedirs("images", exist_ok=True)
plt.savefig("images/graph_name.png")
plt.show()

# churn by gender
plt.figure(figsize=(6,4))
sns.countplot(x="gender", hue="Churn", data=df)
plt.title("Gender vs Churn")

import os
os.makedirs("images", exist_ok=True)
plt.savefig("images/graph_name.png")
plt.show()

# churn by contract type
plt.figure(figsize=(8,5))
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Contract Type vs Churn")
plt.xticks(rotation=20)

import os
os.makedirs("images", exist_ok=True)
plt.savefig("images/graph_name.png")
plt.show()

#monthly charge distribution
plt.figure(figsize=(8,5))
sns.histplot(df["MonthlyCharges"], bins=30, kde=True)
plt.title("Monthly Charges Distribution")

import os
os.makedirs("images", exist_ok=True)
plt.savefig("images/graph_name.png")
plt.show()

# tenor distribution
plt.figure(figsize=(8,5))
sns.histplot(df["tenure"], bins=30, kde=True)
plt.title("Customer Tenure Distribution")

import os
os.makedirs("images", exist_ok=True)
plt.savefig("images/graph_name.png")
plt.show()

#box plot
plt.figure(figsize=(8,5))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")

import os
os.makedirs("images", exist_ok=True)
plt.savefig("images/graph_name.png")
plt.show()

# data processing-------

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y[:10])

# one hot encoding
X = pd.get_dummies(X, drop_first=True)

import joblib
# Save feature names
feature_names = X.columns.tolist()

joblib.dump(feature_names, "model/feature_names.pkl")

print("Feature names saved successfully!")
print("Total NaN values in X:", X.isnull().sum().sum())

print("\nColumns containing NaN:")
print(X.columns[X.isnull().any()])

print("\nNaN count per column:")
print(X.isnull().sum()[X.isnull().sum() > 0])
print(X.head())

print("Shape after Encoding:", X.shape)

print("\nColumns with missing values:")
print(X.isnull().sum()[X.isnull().sum() > 0])

print("\nData types:")
print(X.dtypes)

#train test split
from sklearn.model_selection import train_test_split
# Remove every remaining missing value
X = X.fillna(0)

# Convert every column to float
X = X.astype(float)

print("NaN after fill:", X.isnull().sum().sum())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# feature scalling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# verify data
print(type(X_train))
print(type(X_test))

print(X_train.shape)
print(X_test.shape)

#logistic regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42)
import numpy as np

print("NaN in X_train:", np.isnan(X_train).sum())
print("NaN in X_test:", np.isnan(X_test).sum())

model.fit(X_train, y_train)
print("Model Trained Successfully!")

print("X_train Shape:", X_train.shape)
print("y_train Shape:", y_train.shape)

import numpy as np
print("NaN in X_train:", np.isnan(X_train).sum())
print("NaN in y_train:", np.isnan(y_train).sum())

print("Infinite values in X_train:", np.isinf(X_train).sum())
print("X_train dtype:", X_train.dtype)

# Prediction
y_pred = model.predict(X_test)
print("First 10 Predictions:")
print(y_pred[:10])

# evaluate
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(6,5))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No","Yes"],
            yticklabels=["No","Yes"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# classification report
print(classification_report(y_test, y_pred))

# roc auc score
from sklearn.metrics import roc_auc_score
prob = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, prob)
print("ROC-AUC Score:", auc)

from sklearn.metrics import roc_curve
fpr, tpr, threshold = roc_curve(y_test, prob)
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label="Logistic Regression")
plt.plot([0,1],[0,1],'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# save model 
import joblib
joblib.dump(model, "churn_model.pkl")
print("Model Saved Successfully!")

# save scaler
joblib.dump(scaler, "scaler.pkl")
print("Scaler Saved Successfully!")


# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
print("\nDecision Tree")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_dt))

print("\nClassification Report")
print(classification_report(y_test, y_pred_dt))

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n===== Random Forest =====")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report")
print(classification_report(y_test, y_pred_rf))

# compare model
models = ["Logistic Regression", "Decision Tree", "Random Forest"]
accuracies = [
    accuracy_score(y_test, y_pred),
    accuracy_score(y_test, y_pred_dt),
    accuracy_score(y_test, y_pred_rf)
]
comparison = pd.DataFrame({
    "Model": models,
    "Accuracy": accuracies
})
print(comparison)

# plot model comparision
plt.figure(figsize=(8,5))

plt.bar(comparison["Model"], comparison["Accuracy"])

plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()

# randim forest
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)
print(feature_importance.head(10))
# plot feature
plt.figure(figsize=(10,6))
sns.barplot(
    data=feature_importance.head(10),
    x="Importance",
    y="Feature"
)
plt.title("Top 10 Important Features")
plt.show()

import os
import joblib
os.makedirs("model", exist_ok=True)
joblib.dump(rf_model, "model/churn_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(feature_names, "model/feature_names.pkl")

print("Model Saved Successfully!")
