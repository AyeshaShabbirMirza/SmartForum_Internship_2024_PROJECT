import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('trained_model.pkl')

# Define preprocessing function
def preprocess_input(data):
    # Convert age to days and employment duration to days
    data['DAYS_BIRTH'] = -data['AGE_YEARS'] * 365
    data['DAYS_EMPLOYED'] = data['DAYS_EMPLOYED'].fillna(data['DAYS_EMPLOYED'].median())
    return data

# Function to provide disapproval reasons
def provide_disapproval_reasons(input_data):
    reasons = []
    
    if input_data['AMT_INCOME_TOTAL'][0] <= 150000:
        reasons.append("Income too low.")
    
    if not (21 <= input_data['AGE_YEARS'][0] <= 65):
        reasons.append("Age not within the acceptable range (21-65 years).")
    
    if not (365 <= input_data['DAYS_EMPLOYED'][0] <= 14600):
        reasons.append("Employment duration not within the acceptable range (1-40 years).")
    
    if input_data['CNT_FAM_MEMBERS'][0] > 4:
        reasons.append("Family size too large (more than 4 members).")
    
    if not (input_data['FLAG_OWN_CAR'][0] == 'Y' or input_data['FLAG_OWN_REALTY'][0] == 'Y'):
        reasons.append("No assets (car or real estate) owned.")
    
    return reasons

# Function to take user input and make predictions
def predict_credit_approval():
    # Collecting user input for credit card approval
    
    # Personal Information
    gender = input("Enter your gender (M for Male, F for Female): ")
    car_ownership = input("Do you own a car? (Y for Yes, N for No): ")
    realty_ownership = input("Do you own any real estate (like a house or land)? (Y for Yes, N for No): ")
    income_type = input("What type of income do you receive? (e.g., Salary from a job, Business income, etc.): ")
    education_type = input("What is your highest level of education? (e.g., Higher education, Secondary education, etc.): ")
    family_status = input("What is your current family status? (e.g., Married, Single, Divorced, etc.): ")
    housing_type = input("What is your housing situation? (e.g., Do you live in a house, an apartment, or do you rent?): ")
    occupation_type = input("What is your occupation? (e.g., Laborer, Teacher, Sales staff, etc.): ")

    # Collecting and converting numerical inputs
    age_years = int(input("Enter your age (in years): "))  # Your age in complete years
    years_employed = int(input("How many years have you been employed? "))  # Total years you have worked
    total_income = float(input("What is your total annual income? "))  # Your yearly income
    num_family_members = int(input("How many family members live with you? "))  # Total number of people in your household
    
    # Conditionally asking for the number of children based on marital status
    if family_status.lower() == 'married':
        num_children = int(input("How many children do you have? "))  # Number of children in your care
    else:
        num_children = 0  # Set to 0 if not married

    # Convert user input to DataFrame
    input_df = pd.DataFrame([{
        'CODE_GENDER': gender,
        'FLAG_OWN_CAR': car_ownership,
        'FLAG_OWN_REALTY': realty_ownership,
        'NAME_INCOME_TYPE': income_type,
        'NAME_EDUCATION_TYPE': education_type,
        'NAME_FAMILY_STATUS': family_status,
        'NAME_HOUSING_TYPE': housing_type,
        'OCCUPATION_TYPE': occupation_type,
        'CNT_CHILDREN': num_children,
        'AMT_INCOME_TOTAL': total_income,
        'DAYS_EMPLOYED': years_employed * 365,  # Convert years to days
        'CNT_FAM_MEMBERS': num_family_members,
        'AGE_YEARS': age_years
    }])
    
    # Preprocess input data
    preprocessed_input = preprocess_input(input_df)
    
    # Make predictions
    prediction = model.predict(preprocessed_input)
    probability = model.predict_proba(preprocessed_input)[:, 1]
    
    # Output results
    if prediction[0] == 1:
        print("Credit card approved.")
    else:
        print("Credit card disapproved.")
        reasons = provide_disapproval_reasons(preprocessed_input)
        print("Reasons for disapproval:")
        for reason in reasons:
            print(f"- {reason}")
    
    print(f"Probability of approval: {probability[0]:.4f}")

if __name__ == "__main__":
    predict_credit_approval()
