import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import zscore

# Step 1: Load the Dataset
file_path = r"C:\Users\sraja\Downloads\archive\100_Sales.csv"
data = pd.read_csv(file_path)

# Step 2: Clean Data
# Drop empty columns like 'Unnamed: 9' and 'Unnamed: 10'
data = data.drop(columns=['Unnamed: 9', 'Unnamed: 10'], errors='ignore')

# Handle missing values in numeric columns
numeric_columns = data.select_dtypes(include=['number']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Remove duplicates
data = data.drop_duplicates()

# Detect and handle outliers using Z-scores (numeric columns only)
data = data[(np.abs(zscore(data[numeric_columns])) < 3).all(axis=1)]

# Step 3: Perform Statistical Analysis
print("Descriptive Statistics:")
print(data.describe())

# Step 4: Visualize Data
# 4.1 Histogram for Total Revenue
if 'Total_Revenue' in data.columns:
    plt.hist(data['Total_Revenue'], bins=30, edgecolor='k', color='skyblue')
    plt.title("Total Revenue Distribution")
    plt.xlabel("Total Revenue")
    plt.ylabel("Frequency")
    plt.show()

# 4.2 Boxplot for Total Profit
if 'Total_Profit' in data.columns:
    plt.boxplot(data['Total_Profit'], patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    plt.title("Total Profit Boxplot")
    plt.ylabel("Total Profit")
    plt.show()

# 4.3 Heatmap for Correlations
# Use only numeric columns for the correlation matrix
corr_matrix = data[numeric_columns].corr()

# Debug: Check the correlation matrix
print("Correlation Matrix:\n", corr_matrix)

# Plot the heatmap if the correlation matrix is valid
if not corr_matrix.empty:
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.show()
else:
    print("The correlation matrix is empty. Check your numeric columns or data preprocessing.")

# Step 5: Predictive Modeling
# Prepare features and target variable
if 'Total_Profit' in data.columns and 'Unit_Cost' in data.columns and 'Total_Revenue' in data.columns:
    X = data[['Total_Profit', 'Unit_Cost']]  # Features
    y = data['Total_Revenue']  # Target variable

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the Model
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"R² Score: {r2}")
    print(f"Mean Squared Error (MSE): {mse}")

    # Display the coefficients and intercept of the model
    print("Model Coefficients:", model.coef_)  # Corresponds to 'Total_Profit' and 'Unit_Cost'
    print("Model Intercept:", model.intercept_)
else:
    print("Columns 'Total_Profit', 'Unit_Cost', or 'Total_Revenue' are missing. Cannot train predictive model.")
