# Retention Analysis for Telecom Customers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Step 2: Load dataset
df = pd.read_csv('telecom_customer_data.csv')

# Step 3: Preprocess data
df.dropna(inplace=True)
df = pd.get_dummies(df, drop_first=True)  # Encode categoricals

# Step 4: Split features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Optional: Feature importance
importances = model.feature_importances_
feat_names = X.columns
feat_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
feat_df.sort_values(by='Importance', ascending=False, inplace=True)

# Step 7: Visualize
plt.figure(figsize=(10,6))
sns.barplot(data=feat_df.head(10), x='Importance', y='Feature')
plt.title('Top 10 Important Features Driving Churn')
plt.tight_layout()
plt.show()
