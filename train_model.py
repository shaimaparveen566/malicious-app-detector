import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# Load dataset
data = pd.read_csv("data/Android_Malware_Benign.csv")

# Drop missing values
data = data.dropna()

# Separate features and target
X = data.drop(columns=['Label'])
y = data['Label']

# Encode label
le = LabelEncoder()
y = le.fit_transform(y)

# -------------------------
# Get Top 10 Important Permissions
# -------------------------

temp_model = DecisionTreeClassifier(random_state=42)
temp_model.fit(X, y)

importances = temp_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})

top_10_features = feature_importance_df.sort_values(
    by='Importance', ascending=False
).head(10)

print("Top 10 Important Permissions:")
print(top_10_features)

# Keep only top 10 features
top_features = top_10_features['Feature'].values
X = X[top_features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train final Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model + feature list
joblib.dump(model, "model/decision_tree_model.pkl")
joblib.dump(top_features, "model/top_features.pkl")
joblib.dump(le, "model/label_encoder.pkl")

print("Model saved successfully!")
