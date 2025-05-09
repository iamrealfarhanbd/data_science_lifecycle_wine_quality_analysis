import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# -------------------------
# 1. Data Acquisition
# -------------------------
# Load the Wine Quality (red wine) dataset from the UCI repository.
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

data = pd.read_csv(data_url, sep=';')
# Display the first few rows to confirm data loading
print("Dataset Head:\n", data.head())

print("\nDataset Shape:", data.shape)
print("\nDataset Info:")
print(data.info())
# -------------------------
# 2. Data Cleaning and Preprocessing
# -------------------------
# Check for missing values and duplicates
print("\nMissing Values in Each Column:\n", data.isnull().sum())
print("\nNumber of Duplicate Rows:", data.duplicated().sum())
# If duplicates exist, drop them (uncomment the line below if needed)
# data = data.drop_duplicates()
# -------------------------
# 3. Exploratory Data Analysis (EDA)
# -------------------------
# Plot a correlation heatmap to understand relationships between variables
plt.figure(figsize=(12, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm",
fmt=".2f")
plt.title("Correlation Matrix of Wine Quality Dataset")
plt.show()

# Boxplot for each feature to detect outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=data)
plt.xticks(rotation=45)
plt.title("Boxplot of Each Feature")
plt.show()
# Pairplot for visualizing pairwise relationships
sns.pairplot(data)
plt.suptitle("Pairplot for Wine Quality Data", y=1.02)
plt.show()

# -------------------------
# 4. Feature Engineering
# -------------------------
# As an example, create an interaction feature between 'fixed acidity' and 'pH'
data['acidity_ph_interaction'] = data['fixed acidity'] * data['pH']
# Optionally, you might perform additional feature scaling or transformation if needed
# -------------------------
# 5. Preparing Data for Modeling
# -------------------------
# Define features (X) and target variable (y)
X = data.drop('quality', axis=1)
y = data['quality']
# Split the dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.20, random_state=42)
print("\nTraining set size:", X_train.shape, "Test set size:",
X_test.shape)

# -------------------------
# 6. Model Development: Random Forest Regressor
# -------------------------
# Initialize the Random Forest regressor
rf = RandomForestRegressor(random_state=42)
# Fit the model on the training data
rf.fit(X_train, y_train)
# Predict on the test set
y_pred = rf.predict(X_test)
# Evaluate the model using RMSE and R2 Score
rmse = mean_squared_error(y_test, y_pred) # Changed to default squared=True
r2 = r2_score(y_test, y_pred)
print("\nInitial Model Performance:")
print("RMSE: {:.3f}".format(rmse))
print("R2 Score: {:.3f}".format(r2))

# -------------------------
# 7. Hyperparameter Tuning with GridSearchCV
# -------------------------
# Define a grid of hyperparameters for tuning
param_grid = {
'n_estimators': [100, 200, 300],
'max_depth': [None, 10, 20, 30],
'min_samples_split': [2, 5, 10]
}
# Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
cv=5, scoring='neg_mean_squared_error',
n_jobs=-1, verbose=1)
# Fit GridSearchCV on the training data
grid_search.fit(X_train, y_train)
# Retrieve the best model
best_rf = grid_search.best_estimator_
print("\nBest Parameters Found:", grid_search.best_params_)
# Predict using the best estimator
y_pred_best = best_rf.predict(X_test)
# Evaluate the tuned model
rmse_best = mean_squared_error(y_test, y_pred_best) # Changed to default squared=True
r2_best = r2_score(y_test, y_pred_best)
print("\nTuned Model Performance:")
print("Best RMSE: {:.3f}".format(rmse_best))
print("Best R2 Score: {:.3f}".format(r2_best))

# -------------------------
# 8. Visualizing the Results
# -------------------------
# Scatter plot: Actual vs Predicted Quality
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_best, alpha=0.6, color='blue',
edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
'r--', lw=2)
plt.xlabel('Actual Wine Quality')
plt.ylabel('Predicted Wine Quality')
plt.title('Actual vs. Predicted Wine Quality')
plt.show()

# -------------------------
# 9. Saving the Model for Future Use
# -------------------------
# Save the tuned model using joblib for later deployment or evaluation
model_filename = 'best_rf_model.pkl'
joblib.dump(best_rf, model_filename)
print("\nTuned model saved as:", model_filename)
