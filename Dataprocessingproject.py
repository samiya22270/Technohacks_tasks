import pandas as pd

# Load dataset
# Replace 'data.csv' with your dataset file name

df = pd.read_csv('data.csv')

# Display first 5 rows
print("First 5 Rows of Dataset:")
print(df.head())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing numeric values with mean

df.fillna(df.mean(numeric_only=True), inplace=True)

# Remove duplicate rows

df.drop_duplicates(inplace=True)

# Normalize numeric columns
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

for column in numeric_columns:
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

# Save cleaned dataset

df.to_csv('cleaned_data.csv', index=False)

print("\nData preprocessing completed successfully!")
print("Cleaned dataset saved as cleaned_data.csv")
