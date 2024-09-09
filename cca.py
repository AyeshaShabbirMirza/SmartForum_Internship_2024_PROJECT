# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import joblib  # For saving and loading the model

# Load the dataset
data = pd.read_csv(r'C:\Users\Me\Downloads\application_record.csv')

# Data Cleaning and Preprocessing
def preprocess_data(df):
    df = df.drop(columns=['ID'])
    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].fillna('Unknown')
    df['AGE_YEARS'] = (-df['DAYS_BIRTH']) // 365
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].apply(lambda x: np.nan if x > 0 else -x)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].fillna(df['DAYS_EMPLOYED'].median())
    income_criteria = df['AMT_INCOME_TOTAL'] > 150000
    age_criteria = (df['AGE_YEARS'] >= 21) & (df['AGE_YEARS'] <= 65)
    employment_criteria = (df['DAYS_EMPLOYED'] >= 365) & (df['DAYS_EMPLOYED'] <= 40 * 365)
    family_size_criteria = df['CNT_FAM_MEMBERS'] <= 4
    assets_criteria = (df['FLAG_OWN_CAR'] == 'Y') | (df['FLAG_OWN_REALTY'] == 'Y')
    df['APPROVED'] = (income_criteria & age_criteria & employment_criteria & family_size_criteria & assets_criteria).astype(int)
    return df

# Apply preprocessing
data = preprocess_data(data)

# Display the distribution of the newly created 'APPROVED' column
print("Distribution of 'APPROVED' Column:")
print(data['APPROVED'].value_counts(normalize=True))

# Categorical columns to encode
categorical_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 
                    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']

# Numerical columns to scale
numerical_cols = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS', 'AGE_YEARS']

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Split the data into training and testing sets
X = data.drop(columns=['APPROVED'])
y = data['APPROVED']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing and model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))])

# Define parameter grid for hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__max_depth': [3, 4, 5, 6],
    'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
    'classifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

# Hyperparameter optimization using RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, scoring='roc_auc',  # Reduced n_iter
                                   n_jobs=2, cv=5, verbose=3, random_state=42)  # Reduced n_jobs

# Fit the model
random_search.fit(X_train, y_train)

# Save the trained model
joblib.dump(random_search.best_estimator_, 'trained_model.pkl')
print("Model saved as 'trained_model.pkl'")

# Display the best parameters and the best score
print("Best Parameters: ", random_search.best_params_)
print("Best AUC Score: ", random_search.best_score_)

# Predict on the test set
y_pred = random_search.best_estimator_.predict(X_test)
y_proba = random_search.best_estimator_.predict_proba(X_test)[:, 1]

# Model evaluation metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("AUC-ROC Score: ", roc_auc_score(y_test, y_proba))

# Plotting AUC-ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label="AUC = {:.3f}".format(roc_auc_score(y_test, y_proba)))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()
