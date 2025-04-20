import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, f1_score, recall_score
from sklearn.utils import resample
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import shap
import warnings
warnings.filterwarnings('ignore')

print("Starting enhanced stroke prediction model training with advanced techniques...")

# Load the dataset
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Load the hospital data
hospitals_df = pd.read_csv('HospitalsInIndia.csv')

# Clean the hospitals dataframe
hospitals_df = hospitals_df.iloc[:, 1:].copy()  # Remove first index column
hospitals_df = hospitals_df[hospitals_df['Hospital'].notna()]  # Remove any rows with NaN in Hospital name

# Create a mapping of state to list of hospitals
state_hospitals = {}
for state in hospitals_df['State'].unique():
    state_hospitals[state] = hospitals_df[hospitals_df['State'] == state][['Hospital', 'City', 'LocalAddress', 'Pincode']].values.tolist()

# Save the hospitals data for later use in the application
os.makedirs('models', exist_ok=True)
with open('models/state_hospitals.pkl', 'wb') as f:
    pickle.dump(state_hospitals, f)

print(f"Processed hospital data - {len(hospitals_df)} hospitals across {len(state_hospitals)} states")

# Data exploration and preprocessing
print("\nData exploration and preprocessing...")

# Basic information
print(f"Dataset shape: {df.shape}")
print(f"Column names: {df.columns.tolist()}")

# Drop the ID column as it's not relevant for prediction
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# Check missing values
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
print(f"\nMissing values percentage:")
for col, percent in zip(missing_percent.index, missing_percent.values):
    if percent > 0:
        print(f"{col}: {percent:.2f}%")

# Explore the target variable distribution
print(f"\nTarget variable (stroke) distribution:")
print(df['stroke'].value_counts(normalize=True) * 100)

# Analyze outliers and correlation with target
print("\nAnalyzing feature correlation with stroke...")
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    if col != 'stroke':
        print(f"{col}: {df.groupby('stroke')[col].mean()}")

# Handle missing values and encoding
print("\nHandling missing values...")

# Handle 'gender' column - map to numeric values
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0, 'Other': 2})

# Convert categorical variables to proper format
df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})
df['Residence_type'] = df['Residence_type'].map({'Urban': 1, 'Rural': 0})

# Create work type mapping
work_type_mapping = {
    'Private': 0, 
    'Self-employed': 1, 
    'Govt_job': 2, 
    'children': 3, 
    'Never_worked': 4
}
df['work_type'] = df['work_type'].map(work_type_mapping)

# Create smoking status mapping
smoking_mapping = {
    'never smoked': 0, 
    'formerly smoked': 1, 
    'smokes': 2, 
    'Unknown': 3
}
df['smoking_status'] = df['smoking_status'].map(smoking_mapping)

# ENHANCED FEATURE ENGINEERING
print("\nPerforming advanced feature engineering...")

# Age-related features
df['age_squared'] = df['age'] ** 2
df['age_cubed'] = df['age'] ** 3
df['log_age'] = np.log1p(df['age'])
df['age_group'] = pd.cut(
    df['age'], 
    bins=[0, 12, 18, 35, 50, 65, 80, 100],
    labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior', 'Elderly', 'Very Elderly']
)
df['is_elderly'] = (df['age'] >= 65).astype(int)
df['is_child'] = (df['age'] <= 18).astype(int)
df['is_middle_aged'] = ((df['age'] > 35) & (df['age'] < 55)).astype(int)

# BMI related features
df['bmi_na'] = df['bmi'].isna().astype(int)  # Missing BMI flag
df['bmi_squared'] = df['bmi'].fillna(df['bmi'].median()) ** 2
df['log_bmi'] = np.log1p(df['bmi'].fillna(df['bmi'].median()))

# Create categorical BMI with more detailed ranges
df['bmi_category'] = pd.cut(
    df['bmi'].fillna(df['bmi'].median()),
    bins=[0, 16.5, 18.5, 25, 30, 35, 40, 100],
    labels=['Severely Underweight', 'Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II', 'Obese III']
)

# Glucose features
df['log_glucose'] = np.log1p(df['avg_glucose_level'])
df['glucose_squared'] = df['avg_glucose_level'] ** 2

# Create glucose category with more detailed ranges
df['glucose_category'] = pd.cut(
    df['avg_glucose_level'],
    bins=[0, 70, 100, 125, 140, 180, 250, 500],
    labels=['Hypoglycemic', 'Normal', 'Prediabetic', 'Mild Diabetic', 'Diabetic', 'High Diabetic', 'Very High']
)

# More sophisticated interaction features
df['age_hypertension'] = df['age'] * df['hypertension']
df['age_heart'] = df['age'] * df['heart_disease']
df['glucose_bmi'] = df['avg_glucose_level'] * df['bmi'].fillna(df['bmi'].median())
df['hypertension_heart'] = df['hypertension'] * df['heart_disease']
df['age_glucose'] = df['age'] * df['avg_glucose_level'] / 100  # Scaled for better interpretation
df['smoking_heart'] = df['smoking_status'] * df['heart_disease']
df['glucose_heart'] = df['avg_glucose_level'] * df['heart_disease']
df['bmi_hypertension'] = df['bmi'].fillna(df['bmi'].median()) * df['hypertension']
df['elderly_bmi'] = df['is_elderly'] * df['bmi'].fillna(df['bmi'].median())
df['elderly_heart'] = df['is_elderly'] * df['heart_disease']
df['elderly_hypertension'] = df['is_elderly'] * df['hypertension']
df['elderly_glucose'] = df['is_elderly'] * (df['avg_glucose_level'] > 140).astype(int)

# Risk factor features
df['risk_factor_count'] = (
    df['hypertension'] + 
    df['heart_disease'] + 
    (df['smoking_status'] > 0).astype(int) + 
    (df['bmi'].fillna(df['bmi'].median()) > 30).astype(int) + 
    (df['avg_glucose_level'] > 140).astype(int) +
    (df['age'] > 65).astype(int)
)

df['high_risk'] = (df['risk_factor_count'] >= 3).astype(int)
df['very_high_risk'] = (df['risk_factor_count'] >= 4).astype(int)

# Gender interactions
df['female_elderly'] = ((df['gender'] == 0) & (df['age'] >= 65)).astype(int)
df['male_elderly'] = ((df['gender'] == 1) & (df['age'] >= 65)).astype(int)
df['female_hypertension'] = ((df['gender'] == 0) & (df['hypertension'] == 1)).astype(int)
df['male_hypertension'] = ((df['gender'] == 1) & (df['hypertension'] == 1)).astype(int)

# One-hot encode the categorical features we created
categorical_features = ['age_group', 'glucose_category', 'bmi_category']
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Identify features and target
X = df_encoded.drop(['stroke'], axis=1)  
y = df_encoded['stroke']

# Split data before applying any transformations (to avoid data leakage)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle missing values in BMI (using KNN imputation for better results)
bmi_imputer = KNNImputer(n_neighbors=5)
X_train['bmi'] = bmi_imputer.fit_transform(X_train[['bmi']])[:, 0]
X_test['bmi'] = bmi_imputer.transform(X_test[['bmi']])[:, 0]

# Using advanced resampling technique - SMOTETomek (combines over and under sampling)
print("\nHandling class imbalance with advanced techniques...")
smote_tomek = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)

print(f"Training set shape after resampling: {X_train_resampled.shape}")
print(f"Class distribution after resampling: {np.bincount(y_train_resampled)}")

# Apply advanced feature scaling - PowerTransformer can handle skewed features better
numerical_cols = X_train_resampled.select_dtypes(include=['float64', 'int64']).columns.tolist()
power_transformer = PowerTransformer(method='yeo-johnson')
X_train_resampled[numerical_cols] = power_transformer.fit_transform(X_train_resampled[numerical_cols])
X_test[numerical_cols] = power_transformer.transform(X_test[numerical_cols])

# Feature selection using RFECV (Recursive Feature Elimination with Cross-Validation)
print("\nPerforming recursive feature selection with enhanced cross-validation...")
selector = RFECV(
    estimator=RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
    step=1,
    cv=StratifiedKFold(10),  # Increased from 5 to 10 for better stability
    scoring='roc_auc',  # Use roc_auc for imbalanced data
    min_features_to_select=10,
    n_jobs=-1
)

X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
X_test_selected = selector.transform(X_test)

# Get selected feature names for later use
selected_features = X_train_resampled.columns[selector.support_].tolist()
print(f"Selected features ({len(selected_features)}): {selected_features}")

# Save the column names for prediction
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(selected_features, f)

# Save preprocessing elements for later use
preprocessing_elements = {
    'power_transformer': power_transformer,
    'numerical_cols': numerical_cols,
    'work_type_mapping': work_type_mapping,
    'smoking_mapping': smoking_mapping,
    'bmi_imputer': bmi_imputer,
    'feature_selector': selector
}

with open('models/preprocessing.pkl', 'wb') as f:
    pickle.dump(preprocessing_elements, f)

# Define expanded models to train including XGBoost, LightGBM and CatBoost
print("\nTraining multiple advanced models and ensembles...")
models = {
    'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
    'CatBoost': cb.CatBoostClassifier(random_state=42, verbose=0),
    'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'NeuralNetwork': MLPClassifier(random_state=42, max_iter=1000),
    'SVM': SVC(probability=True, random_state=42)
}

# Define expanded hyperparameters for grid search
param_grids = {
    'XGBoost': {
        'n_estimators': [200, 300],
        'learning_rate': [0.01, 0.03, 0.05],
        'max_depth': [4, 6, 8],
        'min_child_weight': [1, 3],
        'gamma': [0, 0.1],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'scale_pos_weight': [3, 5],  # Important for imbalanced classes
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5]
    },
    'LightGBM': {
        'n_estimators': [200, 300],
        'learning_rate': [0.01, 0.03, 0.05],
        'max_depth': [4, 6, 8],
        'num_leaves': [31, 50, 70],
        'feature_fraction': [0.8, 0.9],
        'bagging_fraction': [0.8, 0.9],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5],
        'class_weight': ['balanced']
    },
    'CatBoost': {
        'iterations': [200, 300],
        'learning_rate': [0.01, 0.05],
        'depth': [4, 6, 8],
        'l2_leaf_reg': [1, 3, 5],
        'border_count': [32, 128],
        'bagging_temperature': [0, 1],
        'class_weights': [[1, 5], [1, 10]]
    },
    'RandomForest': {
        'n_estimators': [200, 300],
        'max_depth': [None, 15, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample']
    },
    'GradientBoosting': {
        'n_estimators': [200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None]
    },
    'NeuralNetwork': {
        'hidden_layer_sizes': [(50,), (100,), (50, 30), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.001, 0.01],
        'early_stopping': [True]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'poly'],
        'class_weight': ['balanced']
    }
}

# Enhanced cross-validation strategy
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train, evaluate and compare models
best_models = {}
model_results = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    # Perform grid search
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grids[model_name],
        cv=cv_strategy,
        scoring='roc_auc',  # Better metric for imbalanced data
        n_jobs=-1,
        verbose=1
    )
    
    # Train on the resampled, selected feature dataset
    grid.fit(X_train_selected, y_train_resampled)
    
    # Save the best model
    best_models[model_name] = grid.best_estimator_
    
    # Predict on test set
    y_pred = best_models[model_name].predict(X_test_selected)
    y_pred_proba = best_models[model_name].predict_proba(X_test_selected)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Store results
    model_results[model_name] = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'f1_score': f1,
        'recall': recall,
        'best_params': grid.best_params_
    }
    
    print(f"{model_name} Best Parameters: {grid.best_params_}")
    print(f"{model_name} ROC AUC: {roc_auc:.4f}")
    print(f"{model_name} F1 Score: {f1:.4f}")
    print(f"{model_name} Recall: {recall:.4f}")
    
    # Save the individual model
    with open(f'models/{model_name.lower()}_model.pkl', 'wb') as f:
        pickle.dump(best_models[model_name], f)

# Create advanced ensemble models
print("\nCreating stacking and voting ensembles...")

# Create a voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', best_models['XGBoost']),
        ('lgbm', best_models['LightGBM']), 
        ('catboost', best_models['CatBoost']),
        ('rf', best_models['RandomForest'])
    ],
    voting='soft'  # Use probabilities for voting
)

voting_clf.fit(X_train_selected, y_train_resampled)
y_pred_voting = voting_clf.predict(X_test_selected)
y_pred_proba_voting = voting_clf.predict_proba(X_test_selected)[:, 1]

# Create a stacking classifier with logistic regression meta-learner
stacking_clf = StackingClassifier(
    estimators=[
        ('xgb', best_models['XGBoost']),
        ('lgbm', best_models['LightGBM']), 
        ('rf', best_models['RandomForest']),
        ('gb', best_models['GradientBoosting'])
    ],
    final_estimator=LogisticRegression(class_weight='balanced'),
    cv=5,
    stack_method='predict_proba'
)

stacking_clf.fit(X_train_selected, y_train_resampled)
y_pred_stacking = stacking_clf.predict(X_test_selected)
y_pred_proba_stacking = stacking_clf.predict_proba(X_test_selected)[:, 1]

# Evaluate ensemble models
voting_accuracy = accuracy_score(y_test, y_pred_voting)
voting_roc_auc = roc_auc_score(y_test, y_pred_proba_voting)
voting_f1 = f1_score(y_test, y_pred_voting)
voting_recall = recall_score(y_test, y_pred_voting)

stacking_accuracy = accuracy_score(y_test, y_pred_stacking)
stacking_roc_auc = roc_auc_score(y_test, y_pred_proba_stacking)
stacking_f1 = f1_score(y_test, y_pred_stacking)
stacking_recall = recall_score(y_test, y_pred_stacking)

print("\nVoting Ensemble Results:")
print(f"Accuracy: {voting_accuracy:.4f}")
print(f"ROC AUC: {voting_roc_auc:.4f}")
print(f"F1 Score: {voting_f1:.4f}")
print(f"Recall: {voting_recall:.4f}")

print("\nStacking Ensemble Results:")
print(f"Accuracy: {stacking_accuracy:.4f}")
print(f"ROC AUC: {stacking_roc_auc:.4f}")
print(f"F1 Score: {stacking_f1:.4f}")
print(f"Recall: {stacking_recall:.4f}")

# Save ensemble models
with open('models/voting_ensemble.pkl', 'wb') as f:
    pickle.dump(voting_clf, f)

with open('models/stacking_ensemble.pkl', 'wb') as f:
    pickle.dump(stacking_clf, f)

# Determine the best model based on ROC AUC
model_results['VotingEnsemble'] = {
    'accuracy': voting_accuracy,
    'roc_auc': voting_roc_auc,
    'f1_score': voting_f1,
    'recall': voting_recall
}

model_results['StackingEnsemble'] = {
    'accuracy': stacking_accuracy,
    'roc_auc': stacking_roc_auc,
    'f1_score': stacking_f1,
    'recall': stacking_recall
}

# Sort models by ROC AUC (best metric for imbalanced classification)
best_models_sorted = sorted(model_results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
best_model_name = best_models_sorted[0][0]

# Print final results and comparison
print("\nFinal Model Comparison (sorted by ROC AUC):")
for model_name, metrics in best_models_sorted:
    print(f"{model_name}: ROC AUC={metrics['roc_auc']:.4f}, F1={metrics['f1_score']:.4f}, Recall={metrics['recall']:.4f}")

print(f"\nBest model: {best_model_name} with ROC AUC of {best_models_sorted[0][1]['roc_auc']:.4f}")

# Prepare the best model for deployment with all necessary information
if best_model_name in ['VotingEnsemble', 'StackingEnsemble']:
    if best_model_name == 'VotingEnsemble':
        best_model = voting_clf
    else:
        best_model = stacking_clf
else:
    best_model = best_models[best_model_name]

# Add additional information to the model
best_model.feature_names = selected_features
best_model.preprocessing = preprocessing_elements
best_model.info = {
    'selected_features': selected_features,
    'metrics': best_models_sorted[0][1],
    'model_type': best_model_name,
    'model_version': '2.0'
}

# Save the best model as the primary model
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Generate SHAP values for the best model (if it's compatible)
print("\nGenerating SHAP explanations for model interpretability...")
try:
    if best_model_name in ['XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting']:
        # Sample a subset of training data for SHAP analysis
        X_sample = X_train_selected[:200]  # Use a small sample for efficiency
        
        # Create explainer
        if best_model_name == 'XGBoost':
            explainer = shap.TreeExplainer(best_model)
        elif best_model_name in ['RandomForest', 'GradientBoosting', 'LightGBM']:
            explainer = shap.TreeExplainer(best_model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Save SHAP values and explainer for later use
        with open('models/shap_explainer.pkl', 'wb') as f:
            pickle.dump({
                'explainer': explainer,
                'shap_values': shap_values,
                'sample_data': X_sample,
                'feature_names': selected_features
            }, f)
        
        print("SHAP explanations generated and saved.")
    else:
        print(f"SHAP explanation not generated for model type: {best_model_name}")

except Exception as e:
    print(f"Error generating SHAP explanations: {str(e)}")

print("\nModel training and evaluation completed successfully.")
print(f"Best model ({best_model_name}) saved as 'models/best_model.pkl'")
print("Additional models and preprocessing components saved in the 'models' directory.") 