import pickle
import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_model')

def load_model():
    try:
        # Try to load model from the saved files
        logger.info("Loading model.pkl")
        with open(os.path.join('models', 'model.pkl'), 'rb') as f:
            model = pickle.load(f)
            
        logger.info("Loading preprocessing.pkl")
        with open(os.path.join('models', 'preprocessing.pkl'), 'rb') as f:
            preprocessing = pickle.load(f)
            
        logger.info("Loading feature_names.pkl")
        with open(os.path.join('models', 'feature_names.pkl'), 'rb') as f:
            feature_names = pickle.load(f)
        
        logger.info(f"Feature names from pickle: {feature_names}")
        
        # Try to get feature names from the model
        if hasattr(model, 'feature_names_in_'):
            logger.info(f"Model's feature_names_in_: {model.feature_names_in_}")
        
        return model, preprocessing, feature_names
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None, None

def create_sample_input():
    # Create a sample input DataFrame similar to what would be submitted through the form
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
    return pd.DataFrame([sample_data])

def prepare_input_data(df, feature_names):
    # Apply mapping for categorical variables
    gender_mapping = {'Male': 1, 'Female': 0}
    married_mapping = {'Yes': 1, 'No': 0}
    residence_mapping = {'Urban': 1, 'Rural': 0}
    
    df['gender'] = df['gender'].map(gender_mapping)
    df['ever_married'] = df['ever_married'].map(married_mapping)
    df['Residence_type'] = df['Residence_type'].map(residence_mapping)
    
    # Define functions to categorize age, BMI and glucose
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
    
    # Create categorical features
    df['age_group'] = df['age'].apply(categorize_age)
    df['bmi_category'] = df['bmi'].apply(categorize_bmi)
    df['glucose_category'] = df['avg_glucose_level'].apply(categorize_glucose)
    
    # One-hot encode categorical features
    age_dummies = pd.get_dummies(df['age_group'], prefix='age_group')
    bmi_dummies = pd.get_dummies(df['bmi_category'], prefix='bmi_category')
    glucose_dummies = pd.get_dummies(df['glucose_category'], prefix='glucose_category')
    work_dummies = pd.get_dummies(df['work_type'], prefix='work_type')
    smoking_dummies = pd.get_dummies(df['smoking_status'], prefix='smoking_status')
    
    # Combine all features
    df = pd.concat([df, age_dummies, bmi_dummies, glucose_dummies, work_dummies, smoking_dummies], axis=1)
    
    # Add engineered features
    df['age_squared'] = df['age'] ** 2
    df['age_cubed'] = df['age'] ** 3
    df['log_age'] = np.log1p(df['age'])
    df['bmi_age'] = df['bmi'] * df['age']
    df['glucose_age'] = df['avg_glucose_level'] * df['age']
    df['is_elderly'] = (df['age'] >= 65).astype(int)
    df['is_child'] = (df['age'] <= 18).astype(int)
    df['is_middle_aged'] = ((df['age'] > 35) & (df['age'] < 55)).astype(int)
    df['high_glucose'] = (df['avg_glucose_level'] >= 126).astype(int)
    df['high_bmi'] = (df['bmi'] >= 30).astype(int)
    df['bmi_na'] = 0  # Since we're providing BMI
    df['bmi_squared'] = df['bmi'] ** 2
    df['log_bmi'] = np.log1p(df['bmi'])
    df['log_glucose'] = np.log1p(df['avg_glucose_level'])
    df['glucose_squared'] = df['avg_glucose_level'] ** 2
    
    # Create interaction features
    df['hypertension_heart'] = df['hypertension'] * df['heart_disease']
    df['hypertension_age'] = df['hypertension'] * df['age']
    df['heart_age'] = df['heart_disease'] * df['age']
    df['age_hypertension'] = df['age'] * df['hypertension']
    df['age_heart'] = df['age'] * df['heart_disease']
    df['glucose_bmi'] = df['avg_glucose_level'] * df['bmi']
    df['age_glucose'] = df['age'] * df['avg_glucose_level'] / 100
    smoking_map = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3}
    df['smoking_heart'] = df['smoking_status'].map(smoking_map) * df['heart_disease']
    df['glucose_heart'] = df['avg_glucose_level'] * df['heart_disease']
    df['bmi_hypertension'] = df['bmi'] * df['hypertension']
    df['elderly_bmi'] = df['is_elderly'] * df['bmi']
    df['elderly_heart'] = df['is_elderly'] * df['heart_disease']
    df['elderly_hypertension'] = df['is_elderly'] * df['hypertension']
    df['elderly_glucose'] = df['is_elderly'] * (df['avg_glucose_level'] > 140).astype(int)
    
    # Risk factor count
    df['risk_factor_count'] = (
        df['hypertension'] + 
        df['heart_disease'] + 
        (df['smoking_status'].map(smoking_map) > 0).astype(int) + 
        df['high_bmi'] + 
        df['high_glucose'] +
        df['is_elderly']
    )
    
    df['high_risk'] = (df['risk_factor_count'] >= 3).astype(int)
    df['very_high_risk'] = (df['risk_factor_count'] >= 4).astype(int)
    
    # Gender interactions
    df['female_elderly'] = ((df['gender'] == 0) & (df['age'] >= 65)).astype(int)
    df['male_elderly'] = ((df['gender'] == 1) & (df['age'] >= 65)).astype(int)
    df['female_hypertension'] = ((df['gender'] == 0) & (df['hypertension'] == 1)).astype(int)
    df['male_hypertension'] = ((df['gender'] == 1) & (df['hypertension'] == 1)).astype(int)
    
    # Handle missing dummy columns by adding them with 0s
    possible_age_groups = ['age_group_Child', 'age_group_Teen', 'age_group_Young Adult', 
                          'age_group_Adult', 'age_group_Senior', 'age_group_Elderly', 'age_group_Very Elderly']
    possible_bmi_categories = ['bmi_category_Severely Underweight', 'bmi_category_Underweight', 
                             'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese I', 
                             'bmi_category_Obese II', 'bmi_category_Obese III']
    possible_glucose_categories = ['glucose_category_Hypoglycemic', 'glucose_category_Normal', 
                                 'glucose_category_Prediabetic', 'glucose_category_Mild Diabetic', 
                                 'glucose_category_Diabetic', 'glucose_category_High Diabetic', 'glucose_category_Very High']
    possible_work_types = ['work_type_Private', 'work_type_Self-employed', 
                          'work_type_Govt_job', 'work_type_children', 'work_type_Never_worked']
    possible_smoking_statuses = ['smoking_status_formerly smoked', 'smoking_status_never smoked', 
                               'smoking_status_smokes', 'smoking_status_Unknown']
    
    # Add missing dummy columns with 0 values
    all_dummy_columns = possible_age_groups + possible_bmi_categories + possible_glucose_categories + possible_work_types + possible_smoking_statuses
    for col in all_dummy_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Drop original categorical columns
    df = df.drop(['age_group', 'bmi_category', 'glucose_category', 'work_type', 'smoking_status'], axis=1)
    
    # Ensure all required feature columns are present
    for feature in feature_names:
        if feature not in df.columns:
            logger.warning(f"Feature {feature} not in input data - adding with zeros")
            df[feature] = 0
    
    # Ensure columns are in the correct order
    df = df[feature_names]
    
    return df

def main():
    logger.info("Starting model test...")
    
    # Load the model
    model, preprocessing, feature_names = load_model()
    if model is None:
        logger.error("Failed to load model")
        return
    
    # Try loading the XGBoost model which might be more reliable
    try:
        logger.info("Loading xgboost_model.pkl")
        with open(os.path.join('models', 'xgboost_model.pkl'), 'rb') as f:
            xgb_model = pickle.load(f)
            
        # Print properties of the XGBoost model
        logger.info(f"XGBoost model type: {type(xgb_model)}")
        if hasattr(xgb_model, 'feature_names_in_'):
            logger.info(f"XGBoost feature_names_in_: {xgb_model.feature_names_in_}")
        if hasattr(xgb_model, 'feature_importances_'):
            # Get top 10 most important features
            feature_importances = pd.Series(xgb_model.feature_importances_, index=feature_names)
            top_features = feature_importances.nlargest(10)
            logger.info(f"Top 10 important features: {top_features}")
    except Exception as e:
        logger.error(f"Error loading XGBoost model: {str(e)}")

if __name__ == "__main__":
    main() 