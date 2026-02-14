#feature imp for regression usng decision tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load Auto MPG dataset
df = pd.read_csv(r"C:\Users\WELCOME\Downloads\auto-mpg.csv")   # keep file in same folder

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Convert horsepower to numeric
df['horsepower'] = pd.to_numeric(df['horsepower'])

# Drop missing values
df.dropna(inplace=True)

# Drop non-numeric column (car name) if present
if 'car name' in df.columns:
    df.drop(columns=['car name'], inplace=True)

# Define features and target
X = df.drop('mpg', axis=1)
y = df['mpg']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)

# Train model
model.fit(X_train, y_train)

# Get feature importances
importances = model.feature_importances_

# Print feature importance
print("Feature Importances:\n")
for feature, importance in zip(X.columns, importances):
    print(f"{feature}: {importance:.4f}")
