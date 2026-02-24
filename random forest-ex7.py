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
