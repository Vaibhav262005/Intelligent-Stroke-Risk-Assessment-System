from flask import render_template, url_for, flash, redirect, request, send_file, jsonify
from flask_login import login_user, current_user, logout_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
import numpy as np
import pickle
import pandas as pd

from forms import (RegistrationForm, LoginForm, HealthDataForm, GlucoseReadingForm, 
                  ProfileEditForm, PasswordChangeForm, NotificationSettingsForm, PrivacySettingsForm)
from models import User, HealthData, GlucoseReading
from utils import (generate_pdf_report, read_glucose_sensor, calculate_bmi,
                  send_email_notification, format_glucose_notification, format_prediction_notification)
from errors import PDFGenerationError
from app import app, db, load_model

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', title='Home')

@app.route("/predict", methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            gender = request.form.get('gender')
            age = float(request.form.get('age'))
            hypertension = int(request.form.get('hypertension'))
            heart_disease = int(request.form.get('heart_disease'))
            ever_married = request.form.get('ever_married')
            work_type = request.form.get('work_type')
            residence_type = request.form.get('residence_type')
            avg_glucose_level = float(request.form.get('avg_glucose_level'))
            smoking_status = request.form.get('smoking_status')
            
            # Handle height and weight for BMI calculation
            height = float(request.form.get('height', 0))
            weight = float(request.form.get('weight', 0))
            
            # Calculate BMI if height and weight are provided
            if height > 0 and weight > 0:
                height_m = height / 100  # Convert cm to m
                bmi = weight / (height_m * height_m)
            else:
                bmi = float(request.form.get('bmi', 0))
            
            # Try direct prediction first using XGBoost model
            try:
                # Load the XGBoost model
                xgb_path = os.path.join('models', 'xgboost_model.pkl')
                with open(xgb_path, 'rb') as f:
                    xgb_model = pickle.load(f)
                
                # Load feature names to ensure we provide features in the right order
                with open(os.path.join('models', 'feature_names.pkl'), 'rb') as f:
                    feature_names = pickle.load(f)
                
                # Create a data dictionary with basic features
                input_data = {
                    'gender': gender,
                    'age': age,
                    'hypertension': hypertension,
                    'heart_disease': heart_disease,
                    'ever_married': ever_married,
                    'work_type': work_type,
                    'Residence_type': residence_type,
                    'avg_glucose_level': avg_glucose_level,
                    'bmi': bmi,
                    'smoking_status': smoking_status
                }
                
                # Apply categorical mappings
                df = pd.DataFrame([input_data])
                gender_mapping = {'Male': 1, 'Female': 0}
                married_mapping = {'Yes': 1, 'No': 0}
                residence_mapping = {'Urban': 1, 'Rural': 0}
                
                df['gender'] = df['gender'].map(gender_mapping)
                df['ever_married'] = df['ever_married'].map(married_mapping)
                df['Residence_type'] = df['Residence_type'].map(residence_mapping)
                
                # Create necessary engineered features
                app.logger.info(f"Creating engineered features for prediction")
                
                # This should match all the features used by the model
                feature_df = pd.DataFrame(index=df.index)
                for feature in feature_names:
                    if feature in df.columns:
                        feature_df[feature] = df[feature]
                    else:
                        # Set default values for missing features
                        feature_df[feature] = 0
                
                # Use the model to make a prediction
                app.logger.info(f"Making prediction with XGBoost model")
                predicted_proba = xgb_model.predict_proba(feature_df)
                stroke_probability = predicted_proba[0][1] * 100  # Convert to percentage
                
                # Determine stroke risk category
                if stroke_probability < 5:
                    risk_category = "Low Risk"
                elif stroke_probability < 15:
                    risk_category = "Moderate Risk"
                else:
                    risk_category = "High Risk"
                
                app.logger.info(f"XGBoost prediction successful: {risk_category} ({stroke_probability:.2f}%)")
                return render_template('index.html', 
                                     prediction_text=f"Stroke Risk: {risk_category} ({stroke_probability:.2f}%)",
                                     prediction_type="XGBoost Prediction",
                                     risk_percentage=stroke_probability)
                
            except Exception as e:
                app.logger.error(f"XGBoost prediction failed: {str(e)}")
                # Fall back to basic prediction using a simple risk score
                
                # Calculate a basic risk score (0-100) using known risk factors
                risk_score = 0
                
                # Age factor (0-30)
                if age < 40:
                    risk_score += 5
                elif age < 55:
                    risk_score += 10
                elif age < 65:
                    risk_score += 15
                elif age < 75:
                    risk_score += 20
                else:
                    risk_score += 30
                
                # Hypertension factor (0-15)
                risk_score += hypertension * 15
                
                # Heart disease factor (0-15)
                risk_score += heart_disease * 15
                
                # BMI factor (0-10)
                if bmi < 18.5:  # Underweight
                    risk_score += 2
                elif bmi < 25:  # Normal
                    risk_score += 0
                elif bmi < 30:  # Overweight
                    risk_score += 5
                else:  # Obese
                    risk_score += 10
                
                # Glucose factor (0-15)
                if avg_glucose_level < 100:  # Normal
                    risk_score += 0
                elif avg_glucose_level < 126:  # Prediabetes
                    risk_score += 7
                else:  # Diabetes
                    risk_score += 15
                
                # Smoking factor (0-10)
                if smoking_status == 'smokes':
                    risk_score += 10
                elif smoking_status == 'formerly smoked':
                    risk_score += 5
                
                # Married factor (0-2)
                if ever_married == 'Yes':
                    risk_score += 2
                
                # Gender factor (0-3)
                if gender == 'Male':
                    risk_score += 3
                
                # Scale to percentage (0-100)
                stroke_probability = risk_score
                
                # Determine stroke risk category
                if stroke_probability < 20:
                    risk_category = "Low Risk"
                elif stroke_probability < 40:
                    risk_category = "Moderate Risk"
                else:
                    risk_category = "High Risk"
                
                return render_template('index.html', 
                                     prediction_text=f"Stroke Risk: {risk_category} ({stroke_probability:.2f}%)",
                                     prediction_type="Basic Prediction",
                                     risk_percentage=stroke_probability)
        
        except Exception as e:
            app.logger.error(f"Prediction error: {str(e)}")
            return render_template('index.html', 
                                 prediction_text="An error occurred. Please check your inputs and try again.",
                                 prediction_type="Error",
                                 risk_percentage=0)
    
    return render_template('index.html', prediction_text="")

@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data)
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            flash('Email already exists. Please log in instead.', 'warning')
            return redirect(url_for('login'))
        else:
            user = User(username=form.username.data, 
                       email=form.email.data, 
                       password=hashed_password)
            
            try:
                db.session.add(user)
                db.session.commit()
                flash('Your account has been created! You can now log in.', 'success')
                return redirect(url_for('login'))
            except Exception as e:
                db.session.rollback()
                flash('An error occurred while creating your account. Please try again.', 'danger')
                app.logger.error(f"Registration error: {str(e)}")
    
    return render_template('register.html', title='Register', form=form)

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login unsuccessful. Please check email and password.', 'danger')
    
    return render_template('login.html', title='Login', form=form)

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route("/health-data", methods=['GET', 'POST'])
def health_data():
    if not current_user.is_authenticated:
        flash("Please log in to access this page.", "warning")
        return redirect(url_for('login'))
    
    form = HealthDataForm()
    
    # Pre-fill form with existing data if available
    health_profile = HealthData.query.filter_by(user_id=current_user.id).first()
    if health_profile and request.method == 'GET':
        form.age.data = health_profile.age
        form.gender.data = health_profile.gender
        form.height.data = health_profile.height
        form.weight.data = health_profile.weight
        form.hypertension.data = str(health_profile.hypertension)  # Convert to string for form
        form.heart_disease.data = str(health_profile.heart_disease)  # Convert to string for form
        form.ever_married.data = health_profile.ever_married
        form.work_type.data = health_profile.work_type
        form.residence_type.data = health_profile.residence_type
        form.avg_glucose_level.data = health_profile.avg_glucose_level
        form.smoking_status.data = health_profile.smoking_status
    
    if form.validate_on_submit():
        try:
            # Convert string to integer for these fields
            hypertension = int(form.hypertension.data) if form.hypertension.data in ['0', '1'] else 0
            heart_disease = int(form.heart_disease.data) if form.heart_disease.data in ['0', '1'] else 0
            
            if health_profile:
                # Update existing profile
                health_profile.age = form.age.data
                health_profile.gender = form.gender.data
                health_profile.height = form.height.data
                health_profile.weight = form.weight.data
                health_profile.hypertension = hypertension
                health_profile.heart_disease = heart_disease
                health_profile.ever_married = form.ever_married.data
                health_profile.work_type = form.work_type.data
                health_profile.residence_type = form.residence_type.data
                health_profile.avg_glucose_level = form.avg_glucose_level.data
                health_profile.smoking_status = form.smoking_status.data
                
                # Calculate BMI if height and weight provided
                if form.height.data and form.weight.data:
                    height_m = form.height.data / 100  # Convert cm to m
                    bmi = form.weight.data / (height_m * height_m)
                    health_profile.bmi = round(bmi, 2)
                    
                    bmi_category = ""
                    if bmi < 18.5:
                        bmi_category = "Underweight"
                    elif bmi < 25:
                        bmi_category = "Normal"
                    elif bmi < 30:
                        bmi_category = "Overweight"
                    else:
                        bmi_category = "Obese"
                    health_profile.bmi_category = bmi_category
            else:
                # Create new profile
                height_m = form.height.data / 100 if form.height.data else 0  # Convert cm to m
                bmi = form.weight.data / (height_m * height_m) if form.height.data and form.weight.data else 0
                
                bmi_category = ""
                if bmi < 18.5:
                    bmi_category = "Underweight"
                elif bmi < 25:
                    bmi_category = "Normal"
                elif bmi < 30:
                    bmi_category = "Overweight"
                else:
                    bmi_category = "Obese"
                
                health_profile = HealthData(
                    user_id=current_user.id,
                    age=form.age.data,
                    gender=form.gender.data,
                    height=form.height.data,
                    weight=form.weight.data,
                    hypertension=hypertension,
                    heart_disease=heart_disease,
                    ever_married=form.ever_married.data,
                    work_type=form.work_type.data,
                    residence_type=form.residence_type.data,
                    avg_glucose_level=form.avg_glucose_level.data,
                    smoking_status=form.smoking_status.data,
                    bmi=round(bmi, 2) if bmi else None,
                    bmi_category=bmi_category if bmi else None
                )
                db.session.add(health_profile)
            
            db.session.commit()
            flash("Your health data has been saved successfully!", "success")
            return redirect(url_for('dashboard'))
            
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error saving health data: {str(e)}")
            flash("An error occurred while saving your health data. Please try again.", "danger")
    
    return render_template('health_data.html', form=form, title="Health Data")

@app.route("/glucose-reading", methods=['GET', 'POST'])
@login_required
def glucose_reading():
    form = GlucoseReadingForm()
    if form.validate_on_submit():
        reading = GlucoseReading(
            user_id=current_user.id,
            glucose_level=form.glucose_level.data,
            reading_type=form.reading_type.data,
            notes=form.notes.data,
            timestamp=datetime.utcnow()
        )
        
        try:
            db.session.add(reading)
            db.session.commit()
            flash('Your glucose reading has been recorded!', 'success')
            
            # Handle email notification if requested
            if form.send_email_notification.data and current_user.email_notifications:
                notification_data = format_glucose_notification(current_user, reading)
                
                # Configure email settings for testing
                os.environ['SMTP_SERVER'] = 'smtp.gmail.com'
                os.environ['SMTP_PORT'] = '587'
                os.environ['SENDER_EMAIL'] = 'your-test-email@gmail.com'  # Update with a test email
                os.environ['SENDER_PASSWORD'] = 'your-app-password'  # Update with an app password
                
                success = send_email_notification(
                    current_user.email,
                    notification_data['subject'],
                    notification_data['email_body']
                )
                if success:
                    flash('Email notification sent successfully.', notification_data['alert_level'])
                else:
                    flash('Failed to send email notification. Please check your email settings.', 'danger')
            
            return redirect(url_for('dashboard'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred while saving your glucose reading. Please try again.', 'danger')
            app.logger.error(f"Glucose reading save error: {str(e)}")
    
    return render_template('glucose_reading.html', title='Glucose Reading', form=form)

@app.route("/dashboard")
@login_required
def dashboard():
    health_data = HealthData.query.filter_by(user_id=current_user.id).first()
    glucose_readings = GlucoseReading.query.filter_by(user_id=current_user.id).order_by(GlucoseReading.timestamp.desc()).limit(10)
    return render_template('dashboard.html', 
                         title='Dashboard',
                         health_data=health_data,
                         glucose_readings=glucose_readings)

@app.route("/generate-report")
@login_required
def generate_report():
    try:
        health_data = HealthData.query.filter_by(user_id=current_user.id).first()
        if not health_data:
            flash('Please complete your health profile first.', 'warning')
            return redirect(url_for('health_data'))
        
        glucose_readings = GlucoseReading.query.filter_by(user_id=current_user.id).order_by(GlucoseReading.timestamp.desc()).limit(30)
        
        pdf_path = generate_pdf_report(current_user, health_data, list(glucose_readings))
        
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
    
    except PDFGenerationError as e:
        flash('Error generating PDF report. Please try again later.', 'danger')
        app.logger.error(f"PDF generation error: {str(e)}")
        return redirect(url_for('dashboard'))
    except Exception as e:
        flash('An unexpected error occurred. Please try again later.', 'danger')
        app.logger.error(f"Unexpected error in generate_report: {str(e)}")
        return redirect(url_for('dashboard'))

@app.route("/profile")
@login_required
def profile():
    return render_template('profile.html', title='Profile')

@app.route("/edit-profile", methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = ProfileEditForm()
    
    # Pre-fill form with existing data
    if request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
        form.phone_number.data = current_user.phone_number
        form.bio.data = current_user.bio
    
    if form.validate_on_submit():
        # Check if username already exists
        if form.username.data != current_user.username:
            user = User.query.filter_by(username=form.username.data).first()
            if user:
                flash('Username already exists. Please choose a different one.', 'danger')
                return render_template('edit_profile.html', title='Edit Profile', form=form)
        
        # Check if email already exists
        if form.email.data != current_user.email:
            user = User.query.filter_by(email=form.email.data).first()
            if user:
                flash('Email already exists. Please choose a different one.', 'danger')
                return render_template('edit_profile.html', title='Edit Profile', form=form)
        
        # Update user profile
        current_user.username = form.username.data
        current_user.email = form.email.data
        current_user.phone_number = form.phone_number.data
        current_user.bio = form.bio.data
        
        try:
            db.session.commit()
            flash('Your profile has been updated!', 'success')
            return redirect(url_for('profile'))
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error updating profile: {str(e)}")
            flash('An error occurred while updating your profile. Please try again.', 'danger')
    
    return render_template('edit_profile.html', title='Edit Profile', form=form)

@app.route("/change-password", methods=['GET', 'POST'])
@login_required
def change_password():
    form = PasswordChangeForm()
    
    if form.validate_on_submit():
        # Verify current password
        if not check_password_hash(current_user.password, form.current_password.data):
            flash('Current password is incorrect.', 'danger')
            return render_template('change_password.html', title='Change Password', form=form)
        
        # Hash the new password
        hashed_password = generate_password_hash(form.new_password.data)
        current_user.password = hashed_password
        
        try:
            db.session.commit()
            flash('Your password has been updated!', 'success')
            return redirect(url_for('profile'))
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error changing password: {str(e)}")
            flash('An error occurred while changing your password. Please try again.', 'danger')
    
    return render_template('change_password.html', title='Change Password', form=form)

@app.route("/notification-settings", methods=['GET', 'POST'])
@login_required
def notification_settings():
    form = NotificationSettingsForm()
    
    # Pre-fill form with existing data
    if request.method == 'GET':
        form.email_notifications.data = current_user.email_notifications
        form.sms_notifications.data = current_user.sms_notifications if hasattr(current_user, 'sms_notifications') else False
        form.health_reminders.data = current_user.health_reminders
        form.glucose_reminders.data = current_user.glucose_reminders
        form.medication_reminders.data = current_user.medication_reminders
        form.app_updates.data = current_user.app_updates
    
    if form.validate_on_submit():
        # Update notification settings
        current_user.email_notifications = form.email_notifications.data
        if hasattr(current_user, 'sms_notifications'):
            current_user.sms_notifications = form.sms_notifications.data
        current_user.health_reminders = form.health_reminders.data
        current_user.glucose_reminders = form.glucose_reminders.data
        current_user.medication_reminders = form.medication_reminders.data
        current_user.app_updates = form.app_updates.data
        
        try:
            db.session.commit()
            flash('Your notification settings have been updated!', 'success')
            return redirect(url_for('profile'))
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error updating notification settings: {str(e)}")
            flash('An error occurred while updating your notification settings. Please try again.', 'danger')
    
    return render_template('notification_settings.html', title='Notification Settings', form=form)

@app.route("/privacy-settings", methods=['GET', 'POST'])
@login_required
def privacy_settings():
    form = PrivacySettingsForm()
    
    # Pre-fill form with existing data
    if request.method == 'GET':
        form.share_health_data.data = current_user.share_health_data
        form.public_profile.data = current_user.public_profile
        form.data_retention.data = current_user.data_retention
    
    if form.validate_on_submit():
        # Update privacy settings
        current_user.share_health_data = form.share_health_data.data
        current_user.public_profile = form.public_profile.data
        current_user.data_retention = form.data_retention.data
        
        try:
            db.session.commit()
            flash('Your privacy settings have been updated!', 'success')
            return redirect(url_for('profile'))
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error updating privacy settings: {str(e)}")
            flash('An error occurred while updating your privacy settings. Please try again.', 'danger')
    
    return render_template('privacy_settings.html', title='Privacy Settings', form=form)

@app.route("/hospitals", methods=['GET', 'POST'])
@login_required
def hospitals():
    try:
        # Load hospital data from pickle file
        with open('models/state_hospitals.pkl', 'rb') as f:
            state_hospitals = pickle.load(f)
        
        # Get all available states
        states = sorted(list(state_hospitals.keys()))
        
        selected_state = None
        hospitals_list = []
        
        if request.method == 'POST':
            selected_state = request.form.get('state')
            if selected_state and selected_state in state_hospitals:
                hospitals_list = state_hospitals[selected_state]
        
        return render_template('hospitals.html', 
                              title='Find Hospitals',
                              states=states,
                              selected_state=selected_state,
                              hospitals=hospitals_list)
    except Exception as e:
        flash(f'Error loading hospital data: {str(e)}', 'danger')
        app.logger.error(f"Hospital data error: {str(e)}")
        return render_template('hospitals.html', 
                              title='Find Hospitals',
                              states=[],
                              selected_state=None,
                              hospitals=[])

@app.route("/public-hospitals", methods=['GET', 'POST'])
def public_hospitals():
    try:
        # Load hospital data from pickle file
        with open('models/state_hospitals.pkl', 'rb') as f:
            state_hospitals = pickle.load(f)
        
        # Get all available states
        states = sorted(list(state_hospitals.keys()))
        
        selected_state = None
        hospitals_list = []
        
        if request.method == 'POST':
            selected_state = request.form.get('state')
            if selected_state and selected_state in state_hospitals:
                hospitals_list = state_hospitals[selected_state]
        
        return render_template('hospitals.html', 
                              title='Find Hospitals',
                              states=states,
                              selected_state=selected_state,
                              hospitals=hospitals_list)
    except Exception as e:
        flash(f'Error loading hospital data: {str(e)}', 'danger')
        app.logger.error(f"Hospital data error: {str(e)}")
        return render_template('hospitals.html', 
                              title='Find Hospitals',
                              states=[],
                              selected_state=None,
                              hospitals=[])

@app.route("/all-hospitals")
def all_hospitals():
    try:
        # Load hospital data from pickle file
        with open('models/state_hospitals.pkl', 'rb') as f:
            state_hospitals = pickle.load(f)
        
        # Prepare all hospitals as a list
        all_hospitals_list = []
        for state, hospitals in state_hospitals.items():
            for hospital in hospitals:
                all_hospitals_list.append({
                    'name': hospital[0],
                    'city': hospital[1],
                    'address': hospital[2],
                    'pincode': hospital[3],
                    'state': state
                })
        
        return render_template('all_hospitals.html', 
                              title='All Hospitals',
                              hospitals=all_hospitals_list)
    except Exception as e:
        flash(f'Error loading hospital data: {str(e)}', 'danger')
        app.logger.error(f"Hospital data error: {str(e)}")
        return render_template('all_hospitals.html', 
                              title='All Hospitals',
                              hospitals=[])

@app.route("/bmi-calculator", methods=['GET', 'POST'])
def bmi_calculator():
    bmi = None
    category = None
    health_risk = None
    
    if request.method == 'POST':
        try:
            weight = float(request.form.get('weight', 0))
            height = float(request.form.get('height', 0))
            
            if weight > 0 and height > 0:
                # Calculate BMI (weight in kg, height in cm)
                height_m = height / 100
                bmi = weight / (height_m * height_m)
                bmi = round(bmi, 1)
                
                # Determine BMI category
                if bmi < 18.5:
                    category = "Underweight"
                    health_risk = "Increased risk of nutritional deficiency and osteoporosis"
                elif 18.5 <= bmi < 25:
                    category = "Normal weight"
                    health_risk = "Lowest risk of health problems"
                elif 25 <= bmi < 30:
                    category = "Overweight"
                    health_risk = "Increased risk of heart disease, high blood pressure, and diabetes"
                elif 30 <= bmi < 35:
                    category = "Obese (Class I)"
                    health_risk = "High risk of heart disease, high blood pressure, and diabetes"
                elif 35 <= bmi < 40:
                    category = "Obese (Class II)"
                    health_risk = "Very high risk of health problems"
                else:
                    category = "Extremely Obese (Class III)"
                    health_risk = "Extremely high risk of serious health problems"
        except Exception as e:
            flash("Please enter valid height and weight values", "danger")
            app.logger.error(f"BMI calculation error: {str(e)}")
    
    return render_template('bmi_calculator.html', title='BMI Calculator', bmi=bmi, category=category, health_risk=health_risk)

@app.route("/glucose-monitor", methods=['GET', 'POST'])
@login_required
def glucose_monitor():
    # Get the latest readings
    glucose_readings = GlucoseReading.query.filter_by(user_id=current_user.id).order_by(GlucoseReading.timestamp.desc()).limit(20)
    
    # Initialize sensor status and reading
    sensor_status = None
    sensor_reading = None
    
    if request.method == 'POST':
        if 'manual_reading' in request.form:
            try:
                # Process manual glucose reading
                glucose_level = float(request.form.get('glucose_level', 0))
                reading_type = request.form.get('reading_type', 'manual')
                notes = request.form.get('notes', '')
                
                if glucose_level > 0:
                    reading = GlucoseReading(
                        user_id=current_user.id,
                        glucose_level=glucose_level,
                        reading_type=reading_type,
                        notes=notes,
                        timestamp=datetime.utcnow()
                    )
                    
                    db.session.add(reading)
                    db.session.commit()
                    flash('Glucose reading recorded successfully!', 'success')
                    return redirect(url_for('glucose_monitor'))
            except Exception as e:
                flash(f'Error recording glucose reading: {str(e)}', 'danger')
                app.logger.error(f"Manual glucose reading error: {str(e)}")
        
        elif 'use_sensor' in request.form:
            # Try to read from a glucose sensor if connected
            sensor_data = read_glucose_sensor()
            
            if sensor_data['status'] == 'success':
                try:
                    reading = GlucoseReading(
                        user_id=current_user.id,
                        glucose_level=sensor_data['glucose_level'],
                        reading_type='sensor',
                        notes='Automatic reading from sensor',
                        timestamp=datetime.utcnow()
                    )
                    
                    db.session.add(reading)
                    db.session.commit()
                    flash('Sensor reading recorded successfully!', 'success')
                    return redirect(url_for('glucose_monitor'))
                except Exception as e:
                    flash(f'Error recording sensor reading: {str(e)}', 'danger')
                    app.logger.error(f"Sensor glucose reading error: {str(e)}")
            else:
                sensor_status = 'error'
                flash(f'Sensor error: {sensor_data["message"]}', 'warning')
    
    # Calculate stats for the readings
    stats = {
        'avg': 0,
        'min': 0,
        'max': 0,
        'count': 0,
        'high_count': 0,
        'low_count': 0,
        'normal_count': 0
    }
    
    readings_list = list(glucose_readings)
    if readings_list:
        glucose_values = [r.glucose_level for r in readings_list]
        stats['avg'] = round(sum(glucose_values) / len(glucose_values), 1)
        stats['min'] = min(glucose_values)
        stats['max'] = max(glucose_values)
        stats['count'] = len(glucose_values)
        stats['high_count'] = sum(1 for v in glucose_values if v > 180)
        stats['low_count'] = sum(1 for v in glucose_values if v < 70)
        stats['normal_count'] = stats['count'] - stats['high_count'] - stats['low_count']
    
    return render_template('glucose_monitor.html', 
                          title='Glucose Monitor',
                          readings=readings_list,
                          stats=stats,
                          sensor_status=sensor_status)

@app.route("/delete-glucose-reading/<int:reading_id>", methods=['POST'])
@login_required
def delete_glucose_reading(reading_id):
    try:
        # Find the glucose reading by ID and check if it belongs to the current user
        reading = GlucoseReading.query.filter_by(id=reading_id, user_id=current_user.id).first_or_404()
        
        # Delete the reading
        db.session.delete(reading)
        db.session.commit()
        
        flash('Glucose reading deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error deleting glucose reading: {str(e)}")
        flash('An error occurred while deleting the glucose reading.', 'danger')
    
    return redirect(url_for('dashboard'))

@app.route("/delete-all-glucose-readings", methods=['POST'])
@login_required
def delete_all_glucose_readings():
    try:
        # Delete all glucose readings for the current user
        readings = GlucoseReading.query.filter_by(user_id=current_user.id).all()
        
        if not readings:
            flash('No glucose readings found to delete.', 'info')
            return redirect(url_for('dashboard'))
        
        # Count readings for feedback message
        count = len(readings)
        
        # Delete each reading
        for reading in readings:
            db.session.delete(reading)
        
        # Commit the transaction
        db.session.commit()
        
        flash(f'Successfully deleted {count} glucose readings.', 'success')
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error deleting all glucose readings: {str(e)}")
        flash('An error occurred while deleting glucose readings.', 'danger')
    
    return redirect(url_for('dashboard'))

@app.route("/send-prediction-notification", methods=['POST'])
@login_required
def send_prediction_notification():
    prediction_text = request.form.get('prediction_text', '')
    risk_percentage = float(request.form.get('risk_percentage', 0))
    send_email = 'send_email_notification' in request.form
    send_sms = 'send_sms_notification' in request.form
    
    if not prediction_text or risk_percentage <= 0:
        flash('No prediction data available for notification.', 'warning')
        return redirect(url_for('predict'))
    
    # Format notification messages
    notification_data = format_prediction_notification(current_user, prediction_text, risk_percentage)
    
    # Send email notification if requested and enabled in settings
    if send_email and current_user.email_notifications:
        # Configure email settings for testing
        os.environ['SMTP_SERVER'] = 'smtp.gmail.com'
        os.environ['SMTP_PORT'] = '587'
        os.environ['SENDER_EMAIL'] = 'your-test-email@gmail.com'  # Update with a test email
        os.environ['SENDER_PASSWORD'] = 'your-app-password'  # Update with an app password
        
        success = send_email_notification(
            current_user.email,
            notification_data['subject'],
            notification_data['email_body']
        )
        if success:
            flash('Email notification sent successfully.', notification_data['alert_level'])
        else:
            flash('Failed to send email notification. Please check your email settings.', 'danger')
    
    # Send SMS notification if user has enabled SMS notifications and has a phone number
    has_sms_notifications = hasattr(current_user, 'sms_notifications') and current_user.sms_notifications
    if send_sms and has_sms_notifications and current_user.phone_number:
        from utils import send_sms
        
        success, details = send_sms(
            current_user.phone_number,
            notification_data['sms_body']
        )
        
        if success:
            flash('SMS notification sent successfully.', notification_data['alert_level'])
        else:
            error_msg = details.get('error', 'Unknown error')
            app.logger.error(f"SMS notification error: {error_msg}")
            flash(f'Failed to send SMS notification: {error_msg}', 'danger')
    
    return redirect(url_for('predict'))