#feature imp for regression usng random forest

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv(r"C:\Users\WELCOME\Downloads\auto-mpg.csv")   # keep file in same folder

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Convert horsepower to numeric
df['horsepower'] = pd.to_numeric(df['horsepower'])

# Drop missing values
df.dropna(inplace=True)

# Drop non-numeric column
if 'car name' in df.columns:
    df.drop(columns=['car name'], inplace=True)

# Define features and target
X = df.drop('mpg', axis=1)
y = df['mpg']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Get feature importance
importances = model.feature_importances_

# Convert to percentage
importances_percent = importances * 100

print("Feature Importance (Random Forest):\n")
for feature, importance in zip(X.columns, importances_percent):
    print(f"{feature}: {importance:.2f}%")
