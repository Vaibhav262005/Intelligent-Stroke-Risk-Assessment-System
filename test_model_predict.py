import pickle
import pandas as pd
import numpy as np
import os
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_model_predict')

def prepare_prediction_input(data):
    """
    Prepare input data for prediction in the same way as in routes.py
    """
    # Create DataFrame
    input_data = pd.DataFrame([data])
    
    # Apply mappings
    gender_mapping = {'Male': 1, 'Female': 0}
    married_mapping = {'Yes': 1, 'No': 0}
    residence_mapping = {'Urban': 1, 'Rural': 0}
    
    input_data['gender'] = input_data['gender'].map(gender_mapping)
    input_data['ever_married'] = input_data['ever_married'].map(married_mapping)
    input_data['Residence_type'] = input_data['Residence_type'].map(residence_mapping)
    
    # Generate engineered features
    # Categorize age
    def categorize_age(age):
        if age < 13:
            return 'Child'
        elif age < 19:
            return 'Teen'
        elif age < 36:
            return 'Young Adult'
        elif age < 51:
            return 'Adult'
        elif age < 66:
            return 'Senior'
        elif age < 81:
            return 'Elderly'
        else:
            return 'Very Elderly'
    
    # Categorize BMI
    def categorize_bmi(bmi):
        if bmi < 16.5:
            return 'Severely Underweight'
        elif bmi < 18.5:
            return 'Underweight'
        elif bmi < 25:
            return 'Normal'
        elif bmi < 30:
            return 'Overweight'
        elif bmi < 35:
            return 'Obese I'
        elif bmi < 40:
            return 'Obese II'
        else:
            return 'Obese III'
    
    # Categorize glucose
    def categorize_glucose(glucose):
        if glucose < 70:
            return 'Hypoglycemic'
        elif glucose < 100:
            return 'Normal'
        elif glucose < 125:
            return 'Prediabetic'
        elif glucose < 140:
            return 'Mild Diabetic'
        elif glucose < 180:
            return 'Diabetic'
        elif glucose < 250:
            return 'High Diabetic'
        else:
            return 'Very High'
    
    # Apply categorization
    input_data['age_group'] = input_data['age'].apply(categorize_age)
    input_data['bmi_category'] = input_data['bmi'].apply(categorize_bmi)
    input_data['glucose_category'] = input_data['avg_glucose_level'].apply(categorize_glucose)
    
    # Get dummies for categorical variables
    age_dummies = pd.get_dummies(input_data['age_group'], prefix='age_group')
    bmi_dummies = pd.get_dummies(input_data['bmi_category'], prefix='bmi_category')
    glucose_dummies = pd.get_dummies(input_data['glucose_category'], prefix='glucose_category')
    work_dummies = pd.get_dummies(input_data['work_type'], prefix='work_type')
    smoking_dummies = pd.get_dummies(input_data['smoking_status'], prefix='smoking_status')
    
    # Combine all features
    input_data = pd.concat([input_data, age_dummies, bmi_dummies, glucose_dummies, 
                           work_dummies, smoking_dummies], axis=1)
    
    # Create engineered features
    input_data['age_squared'] = input_data['age'] ** 2
    input_data['age_cubed'] = input_data['age'] ** 3
    input_data['log_age'] = np.log1p(input_data['age'])
    input_data['bmi_age'] = input_data['bmi'] * input_data['age']
    input_data['glucose_age'] = input_data['avg_glucose_level'] * input_data['age']
    input_data['is_elderly'] = (input_data['age'] >= 65).astype(int)
    input_data['is_child'] = (input_data['age'] <= 18).astype(int)
    input_data['is_middle_aged'] = ((input_data['age'] > 35) & (input_data['age'] < 55)).astype(int)
    input_data['high_glucose'] = (input_data['avg_glucose_level'] >= 126).astype(int)
    input_data['high_bmi'] = (input_data['bmi'] >= 30).astype(int)
    input_data['bmi_na'] = 0  # Since we're providing BMI
    input_data['bmi_squared'] = input_data['bmi'] ** 2
    input_data['log_bmi'] = np.log1p(input_data['bmi'])
    input_data['log_glucose'] = np.log1p(input_data['avg_glucose_level'])
    input_data['glucose_squared'] = input_data['avg_glucose_level'] ** 2
    
    # Create interaction features
    input_data['hypertension_heart'] = input_data['hypertension'] * input_data['heart_disease']
    input_data['hypertension_age'] = input_data['hypertension'] * input_data['age']
    input_data['heart_age'] = input_data['heart_disease'] * input_data['age']
    input_data['age_hypertension'] = input_data['age'] * input_data['hypertension']
    input_data['age_heart'] = input_data['age'] * input_data['heart_disease']
    input_data['glucose_bmi'] = input_data['avg_glucose_level'] * input_data['bmi']
    input_data['age_glucose'] = input_data['age'] * input_data['avg_glucose_level'] / 100
    smoking_map = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3}
    input_data['smoking_heart'] = input_data['smoking_status'].map(smoking_map) * input_data['heart_disease']
    input_data['glucose_heart'] = input_data['avg_glucose_level'] * input_data['heart_disease']
    input_data['bmi_hypertension'] = input_data['bmi'] * input_data['hypertension']
    input_data['elderly_bmi'] = input_data['is_elderly'] * input_data['bmi']
    input_data['elderly_heart'] = input_data['is_elderly'] * input_data['heart_disease']
    input_data['elderly_hypertension'] = input_data['is_elderly'] * input_data['hypertension']
    input_data['elderly_glucose'] = input_data['is_elderly'] * (input_data['avg_glucose_level'] > 140).astype(int)
    
    # Risk factor count
    input_data['risk_factor_count'] = (
        input_data['hypertension'] + 
        input_data['heart_disease'] + 
        (input_data['smoking_status'].map(smoking_map) > 0).astype(int) + 
        input_data['high_bmi'] + 
        input_data['high_glucose'] +
        input_data['is_elderly']
    )
    
    input_data['high_risk'] = (input_data['risk_factor_count'] >= 3).astype(int)
    input_data['very_high_risk'] = (input_data['risk_factor_count'] >= 4).astype(int)
    
    # Gender interactions
    input_data['female_elderly'] = ((input_data['gender'] == 0) & (input_data['age'] >= 65)).astype(int)
    input_data['male_elderly'] = ((input_data['gender'] == 1) & (input_data['age'] >= 65)).astype(int)
    input_data['female_hypertension'] = ((input_data['gender'] == 0) & (input_data['hypertension'] == 1)).astype(int)
    input_data['male_hypertension'] = ((input_data['gender'] == 1) & (input_data['hypertension'] == 1)).astype(int)
    
    # Drop original categorical columns
    input_data = input_data.drop(['age_group', 'bmi_category', 'glucose_category', 
                                 'work_type', 'smoking_status'], axis=1)
    
    return input_data

def main():
    try:
        # Sample input data
        sample_data = {
            'gender': 'Male',
            'age': 65,
            'hypertension': 1,
            'heart_disease': 0,
            'ever_married': 'Yes',
            'work_type': 'Private',
            'Residence_type': 'Urban',
            'avg_glucose_level': 140,
            'bmi': 28,
            'smoking_status': 'formerly smoked'
        }
        
        # Prepare input data
        input_data = prepare_prediction_input(sample_data)
        logger.info(f"Input data shape: {input_data.shape}")
        logger.info(f"Input data columns: {input_data.columns.tolist()}")
        
        # Load the XGBoost model
        model_path = os.path.join('models', 'xgboost_model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Make prediction
        logger.info("Making prediction directly with XGBoost model")
        prediction = model.predict_proba(input_data)
        stroke_probability = prediction[0][1] * 100
        
        # Determine risk category
        if stroke_probability < 5:
            risk_category = "Low Risk"
        elif stroke_probability < 15:
            risk_category = "Moderate Risk"
        else:
            risk_category = "High Risk"
        
        logger.info(f"Prediction result: {risk_category} ({stroke_probability:.2f}%)")
        
        print(f"\nStroke Risk Prediction: {risk_category} ({stroke_probability:.2f}%)")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 