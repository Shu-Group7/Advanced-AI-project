import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_excel("student.xlsx")

# Preview
print(df.head())
print(df.info())
rows = df.shape[0]
cols = df.shape[1]

print("Total Records:", rows)
print("Total Columns:", cols)

df = df.drop(columns=["id", "Name"])

print(df.isnull().sum())

# Fill numeric with median
for col in df.select_dtypes(include=np.number):
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include='object'):
    df[col] = df[col].astype(str)   # ✅ convert ALL to string FIRST
    df[col] = df[col].fillna(df[col].mode()[0])

le = LabelEncoder()

for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])

X = df.drop("Depression", axis=1)
y = df["Depression"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sns.countplot(x='Depression', data=df)
plt.title("Depression Distribution")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

sns.boxplot(x="Depression", y="Academic Pressure", data=df)
plt.show()

sns.boxplot(x="Depression", y="Financial Stress", data=df)
plt.show()

sns.boxplot(x="Depression", y="Work/Study Hours", data=df)
plt.show()

#Logistic regression.
lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

#Random forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

#Model Evaluation

#1.Accuracy
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

#2.Classification report
print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_lr))

print("Random Forest Report:")
print(classification_report(y_test, y_pred_rf))

#3.confusion matrix

cm = confusion_matrix(y_test, y_pred_rf)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

new_data = pd.DataFrame({
    'Gender': [1],
    'Age': [21],
    'City': [3],
    'Working Professional or Student': [1],
    'Academic Pressure': [4],
    'CGPA': [7.5],
    'Study Satisfaction': [3],
    'Sleep Duration': [2],
    'Dietary Habits': [1],
    'Degree': [2],
    'Have you ever had suicidal thoughts ?': [0],
    'Work/Study Hours': [6],
    'Financial Stress': [3],
    'Family History of Mental Illness': [1]
})

new_data_scaled = scaler.transform(new_data)
prediction = rf.predict(new_data_scaled)

if prediction[0] == 1:
    print("Student is likely Depressed")
else:
    print("Student is Not Depressed")

from sklearn.model_selection import GridSearchCV
importances = rf.feature_importances_

import joblib

joblib.dump(rf, "model.pkl")
joblib.dump(scaler, "scaler.pkl")