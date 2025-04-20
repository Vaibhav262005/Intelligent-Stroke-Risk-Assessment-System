import pickle
import pandas as pd
import numpy as np
import os

print("Creating feature_names.pkl file for the XGBoost model...")

# These are typical features after engineering based on the prediction code
def create_feature_names():
    # Basic features from the form input
    basic_features = [
        'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
        'Residence_type', 'avg_glucose_level', 'bmi'
    ]
    
    # Work type features
    work_type_features = [
        'work_type_Private', 'work_type_Self-employed', 'work_type_Govt_job',
        'work_type_children', 'work_type_Never_worked'
    ]
    
    # Smoking status features
    smoking_features = [
        'smoking_status_formerly smoked', 'smoking_status_never smoked',
        'smoking_status_smokes', 'smoking_status_Unknown'
    ]
    
    # Age related features
    age_features = [
        'age_squared', 'age_cubed', 'log_age', 'is_elderly', 'is_child', 'is_middle_aged'
    ]
    
    # BMI and glucose related features
    health_features = [
        'bmi_squared', 'log_bmi', 'high_bmi', 'glucose_squared', 'log_glucose',
        'high_glucose'
    ]
    
    # Interaction features
    interaction_features = [
        'bmi_age', 'glucose_age', 'hypertension_heart', 'hypertension_age',
        'heart_age', 'age_hypertension', 'age_heart', 'glucose_bmi', 'age_glucose',
        'smoking_heart', 'glucose_heart', 'bmi_hypertension', 'elderly_bmi',
        'elderly_heart', 'elderly_hypertension', 'elderly_glucose',
        'female_elderly', 'male_elderly', 'female_hypertension', 'male_hypertension'
    ]
    
    # Risk related features
    risk_features = [
        'risk_factor_count', 'high_risk', 'very_high_risk'
    ]
    
    # Categorical features for age, BMI, and glucose
    categorical_features = [
        'age_group_Child', 'age_group_Teen', 'age_group_Young Adult', 'age_group_Adult',
        'age_group_Senior', 'age_group_Elderly', 'age_group_Very Elderly',
        'bmi_category_Severely Underweight', 'bmi_category_Underweight', 
        'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese I',
        'bmi_category_Obese II', 'bmi_category_Obese III',
        'glucose_category_Hypoglycemic', 'glucose_category_Normal', 
        'glucose_category_Prediabetic', 'glucose_category_Mild Diabetic',
        'glucose_category_Diabetic', 'glucose_category_High Diabetic',
        'glucose_category_Very High'
    ]
    
    # Combine all features
    all_features = (basic_features + work_type_features + smoking_features +
                    age_features + health_features + interaction_features +
                    risk_features + categorical_features)
    
    return all_features

# Create the directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Generate and save the feature names
feature_names = create_feature_names()
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print(f"Saved {len(feature_names)} feature names to models/feature_names.pkl")
print("Feature names:", feature_names) 