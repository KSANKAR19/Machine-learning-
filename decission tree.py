import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------
# Step 1: Create Small Dataset
# --------------------------------------------------
data = {
    'Weather': ['Sunny', 'Sunny', 'Rainy', 'Rainy'],
    'Humidity': ['High', 'Normal', 'High', 'Normal'],
    'Play': ['No', 'Yes', 'Yes', 'Yes']
}

df = pd.DataFrame(data)

# --------------------------------------------------
# Step 2: Label Encoding
# --------------------------------------------------
encoders = {}
for col in df.columns:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

print("Encoded Dataset:\n", df)

# --------------------------------------------------
# Step 3: Split Features & Target
# --------------------------------------------------
X = df[['Weather', 'Humidity']]
y = df['Play']

# --------------------------------------------------
# Step 4: Train Small Decision Tree
# --------------------------------------------------
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=1,   # 👈 Makes tree VERY SMALL
    random_state=0
)
model.fit(X, y)

print("\nModel Trained Successfully")

# --------------------------------------------------
# Step 5: Predict New Sample
# --------------------------------------------------
new_sample = pd.DataFrame({
    'Weather': encoders['Weather'].transform(['Sunny']),
    'Humidity': encoders['Humidity'].transform(['High'])
})

prediction = model.predict(new_sample)
result = encoders['Play'].inverse_transform(prediction)

print("\nPrediction for New Sample:")
print("Play =", result[0])

# --------------------------------------------------
# Step 6: Plot Small Tree
# --------------------------------------------------
plt.figure(figsize=(6, 4))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=encoders['Play'].classes_,
    filled=True,
    rounded=True,
    impurity=False
)
plt.title("Small Decision Tree (Entropy)")
plt.show()
