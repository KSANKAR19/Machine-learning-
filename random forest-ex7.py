simple explanation 
  The program creates a dataset, converts categorical data into numbers, splits data into training and testing sets, 
  trains a Random Forest model using multiple decision trees, evaluates accuracy, predicts new data, and shows feature importance


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------------------
# Step 1: Create Dataset
# --------------------------------------------------
data = {
    'Study_Hours': ['High','High','Medium','Low','Low','Medium','High','Low','Medium','High'],
    'Attendance': ['Good','Average','Good','Poor','Average','Average','Good','Poor','Good','Average'],
    'Assignment': ['Yes','Yes','Yes','No','Yes','Yes','Yes','No','Yes','Yes'],
    'Result': ['Pass','Pass','Pass','Fail','Fail','Pass','Pass','Fail','Pass','Pass']
}

df = pd.DataFrame(data)

print("Original Dataset:\n", df)

# --------------------------------------------------
# Step 2: Label Encoding
# --------------------------------------------------
encoders = {}
for col in df.columns:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

print("\nEncoded Dataset:\n", df)

# --------------------------------------------------
# Step 3: Split Features & Target
# --------------------------------------------------
X = df[['Study_Hours', 'Attendance', 'Assignment']]
y = df['Result']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# --------------------------------------------------
# Step 4: Train Random Forest
# --------------------------------------------------
model = RandomForestClassifier(
    n_estimators=50,   # number of trees
    criterion='entropy',
    max_depth=3,
    random_state=0
)

model.fit(X_train, y_train)

print("\nRandom Forest Model Trained Successfully")

# --------------------------------------------------
# Step 5: Model Evaluation
# --------------------------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --------------------------------------------------
# Step 6: Predict New Student
# --------------------------------------------------
new_student = pd.DataFrame({
    'Study_Hours': encoders['Study_Hours'].transform(['Medium']),
    'Attendance': encoders['Attendance'].transform(['Good']),
    'Assignment': encoders['Assignment'].transform(['Yes'])
})

prediction = model.predict(new_student)
result = encoders['Result'].inverse_transform(prediction)

print("\nNew Student Prediction:")
print("Result =", result[0])

# --------------------------------------------------
# Step 7: Feature Importance
# --------------------------------------------------
importances = model.feature_importances_

plt.figure()
plt.bar(X.columns, importances)
plt.title("Feature Importance (Random Forest)")
plt.show()

-----------OUTPUT------------------
Original Dataset:
   Study_Hours Attendance Assignment Result
0        High       Good        Yes   Pass
1        High    Average        Yes   Pass
2      Medium       Good        Yes   Pass
3         Low       Poor         No   Fail
4         Low    Average        Yes   Fail
5      Medium    Average        Yes   Pass
6        High       Good        Yes   Pass
7         Low       Poor         No   Fail
8      Medium       Good        Yes   Pass
9        High    Average        Yes   Pass

Encoded Dataset:
    Study_Hours  Attendance  Assignment  Result
0            0           1           1       1
1            0           0           1       1
2            2           1           1       1
3            1           2           0       0
4            1           0           1       0
5            2           0           1       1
6            0           1           1       1
7            1           2           0       0
8            2           1           1       1
9            0           0           1       1

Random Forest Model Trained Successfully

Accuracy: 0.6666666666666666

Classification Report:
               precision    recall  f1-score   support

           0       0.00      0.00      0.00         1
           1       0.67      1.00      0.80         2

    accuracy                           0.67         3
   macro avg       0.33      0.50      0.40         3
weighted avg       0.44      0.67      0.53         3


New Student Prediction:
Result = Pass
C:\Users\DELL\anaconda3\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
C:\Users\DELL\anaconda3\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
C:\Users\DELL\anaconda3\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])



