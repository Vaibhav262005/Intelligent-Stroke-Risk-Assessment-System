from flask import Flask, render_template, request
from flask_login import LoginManager, current_user
from flask_migrate import Migrate
import joblib
import os
import numpy as np
import pickle
import sklearn
import logging

# Import db from models
from models import db, User

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy with the Flask app
db.init_app(app)

# Initialize Flask-Migrate
migrate = Migrate(app, db)

# Initialize Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def load_model():
    try:
        # Try to load the best performing model (ensemble or XGBoost)
        model_path = os.path.join('models', 'best_model.pkl')
        if not os.path.exists(model_path):
            model_path = os.path.join('models', 'model.pkl')
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        # Also load preprocessing elements and feature names
        preprocessing_path = os.path.join('models', 'preprocessing.pkl')
        with open(preprocessing_path, 'rb') as f:
            preprocessing = pickle.load(f)
            
        feature_names_path = os.path.join('models', 'feature_names.pkl')
        with open(feature_names_path, 'rb') as f:
            feature_names = pickle.load(f)
        
        # Handle preprocessing correctly - ensure it's a proper object with transform method
        # If preprocessing is a dict, we need to restructure it
        if isinstance(preprocessing, dict):
            class PreprocessingTransformer:
                def __init__(self, components):
                    self.components = components
                    
                def transform(self, X):
                    # Make a copy to avoid modifying original data
                    X_transformed = X.copy()
                    
                    # Apply transformations
                    if 'power_transformer' in self.components and 'numerical_cols' in self.components:
                        transformer = self.components['power_transformer']
                        cols = [c for c in self.components['numerical_cols'] if c in X_transformed.columns]
                        if cols:
                            X_transformed[cols] = transformer.transform(X_transformed[cols])
                    
                    # Handle feature selection if present
                    if 'feature_selector' in self.components and hasattr(self.components['feature_selector'], 'transform'):
                        # Only include features that are in the feature_selector's support
                        if hasattr(self.components['feature_selector'], 'support_'):
                            # Get the original features used during training
                            selected_indices = self.components['feature_selector'].support_
                            app.logger.info(f"Feature selector active: {sum(selected_indices)} features selected out of {len(selected_indices)}")
                            if hasattr(self.components['feature_selector'], 'estimator_') and hasattr(self.components['feature_selector'].estimator_, 'feature_names_in_'):
                                original_features = self.components['feature_selector'].estimator_.feature_names_in_
                                # Get the selected features
                                selected_features = original_features[selected_indices]
                                app.logger.info(f"Selected features: {selected_features[:10]}...")
                                # Ensure only these features are used
                                X_transformed = X_transformed[selected_features]
                    
                    return X_transformed
            
            # Create a transformer object from the dictionary
            preproc_transformer = PreprocessingTransformer(preprocessing)
            model.preprocessing = preproc_transformer
        else:
            model.preprocessing = preprocessing
            
        model.feature_names = feature_names
        
        # Try to load model info for reference
        try:
            with open(os.path.join('models', 'model_info.pkl'), 'rb') as f:
                model_info = pickle.load(f)
                model.info = model_info
                app.logger.info(f"Loaded model: {model_info.get('best_model_name', 'Unknown')} with ROC AUC: {model_info.get('roc_auc', 'Unknown')}")
        except Exception as e:
            app.logger.warning(f"Could not load model info: {str(e)}")
            
        app.logger.info("Model loaded successfully")
        return model
    except Exception as e:
        app.logger.error(f"Error loading model: {str(e)}")
        # Try to load fallback XGBoost model if main model fails
        try:
            app.logger.info("Attempting to load fallback XGBoost model")
            xgb_path = os.path.join('models', 'xgboost_model.pkl')
            with open(xgb_path, 'rb') as f:
                model = pickle.load(f)
                
            app.logger.info("XGBoost model loaded successfully")
            return model
        except Exception as e:
            app.logger.error(f"Error loading fallback model: {str(e)}")
            return None

# Register error handlers
from errors import init_app as init_error_handlers
init_error_handlers(app)

# Import routes after app initialization
from routes import *

if __name__ == "__main__":
    app.run(debug=True, port=7384)
