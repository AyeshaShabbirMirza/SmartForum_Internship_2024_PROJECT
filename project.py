# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Step 1: Load the dataset
cc_apps = pd.read_csv(r'C:\Users\Me\Downloads\cc_approvals.data', header=None)

# Step 2: Inspect data
print("First few rows of the dataset:\n", cc_apps.head())  # Display the first few rows
print("\nSummary statistics of the dataset:\n", cc_apps.describe())  # Print summary statistics
print("\nDataset information:\n", cc_apps.info())  # Print DataFrame information

# Step 3: Drop irrelevant features
cc_apps = cc_apps.drop([11, 13], axis=1)

# Step 4: Split into training and test sets
cc_apps_train, cc_apps_test = train_test_split(cc_apps, test_size=0.33, random_state=42)

# Replace '?'s with NaN in the train and test sets
cc_apps_train = cc_apps_train.replace('?', np.nan)
cc_apps_test = cc_apps_test.replace('?', np.nan)

# Step 5: Impute the missing values using SimpleImputer
# Numerical columns
num_cols = [2, 7, 10, 14]
num_imputer = SimpleImputer(strategy='mean')

cc_apps_train[num_cols] = num_imputer.fit_transform(cc_apps_train[num_cols])
cc_apps_test[num_cols] = num_imputer.transform(cc_apps_test[num_cols])

# Categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')
cat_cols = cc_apps_train.select_dtypes(include=['object']).columns

cc_apps_train[cat_cols] = cat_imputer.fit_transform(cc_apps_train[cat_cols])
cc_apps_test[cat_cols] = cat_imputer.transform(cc_apps_test[cat_cols])

# Verify no missing values remain
print("Total missing values in training set: ", cc_apps_train.isnull().sum().sum())
print("Total missing values in test set: ", cc_apps_test.isnull().sum().sum())

# Step 6: Convert categorical features to dummy variables
cc_apps_train = pd.get_dummies(cc_apps_train)
cc_apps_test = pd.get_dummies(cc_apps_test)

# Align test set with train set columns
cc_apps_test = cc_apps_test.reindex(columns=cc_apps_train.columns, fill_value=0)

# Step 7: Feature Scaling with MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Separate features and labels
X_train, y_train = cc_apps_train.iloc[:, :-1].values, cc_apps_train.iloc[:, -1].values
X_test, y_test = cc_apps_test.iloc[:, :-1].values, cc_apps_test.iloc[:, -1].values

# Rescale features
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 8: Train a Logistic Regression model
logreg = LogisticRegression()

# Fit the model
logreg.fit(X_train, y_train)

# Predict on the test set
y_pred = logreg.predict(X_test)

# Evaluate the model
print("Accuracy of logistic regression classifier: ", logreg.score(X_test, y_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 9: Hyperparameter Optimization using GridSearchCV
param_grid = {'tol': [0.01, 0.001, 0.0001], 'max_iter': [100, 150, 200]}
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV model
grid_model_result = grid_model.fit(X_train, y_train)

# Summarize results
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best Score: %f using %s" % (best_score, best_params))

# Extract the best model and evaluate it on the test set
best_model = grid_model_result.best_estimator_
print("Accuracy of the best logistic regression classifier: ", best_model.score(X_test, y_test))
