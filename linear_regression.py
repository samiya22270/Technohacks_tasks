# linear_regression.py

# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
# Replace 'data.csv' with your dataset file
df = pd.read_csv("data.csv")

# Display first 5 rows
print("First 5 Rows of Dataset:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing numeric values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Select features and target variable
# Example:
# Suppose the dataset has columns:
# Area, Bedrooms, Price
# We predict Price

X = df.iloc[:, :-1]   # Features
y = df.iloc[:, -1]    # Target variable

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

print("\nLinear Regression Model Trained Successfully!")
