from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

# Define db but don't initialize it here
db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    phone_number = db.Column(db.String(20), nullable=True)
    bio = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Notification settings
    email_notifications = db.Column(db.Boolean, default=True)
    sms_notifications = db.Column(db.Boolean, default=True)
    health_reminders = db.Column(db.Boolean, default=True)
    glucose_reminders = db.Column(db.Boolean, default=True)
    medication_reminders = db.Column(db.Boolean, default=True)
    app_updates = db.Column(db.Boolean, default=True)
    
    # Privacy settings
    share_health_data = db.Column(db.Boolean, default=False)
    public_profile = db.Column(db.Boolean, default=False)
    data_retention = db.Column(db.String(10), default='forever')
    
    # Relationships
    health_data = db.relationship('HealthData', backref='user', lazy=True)
    glucose_readings = db.relationship('GlucoseReading', backref='user', lazy=True)

class HealthData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    height = db.Column(db.Float, nullable=False)
    weight = db.Column(db.Float, nullable=False)
    hypertension = db.Column(db.Boolean, nullable=False)
    heart_disease = db.Column(db.Boolean, nullable=False)
    ever_married = db.Column(db.String(3), nullable=False)
    work_type = db.Column(db.String(20), nullable=False)
    residence_type = db.Column(db.String(10), nullable=False)
    avg_glucose_level = db.Column(db.Float, nullable=False)
    smoking_status = db.Column(db.String(20), nullable=False)
    bmi = db.Column(db.Float, nullable=True)
    bmi_category = db.Column(db.String(20), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class HealthRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    weight = db.Column(db.Float)
    height = db.Column(db.Float)
    bmi = db.Column(db.Float)
    bmi_category = db.Column(db.String(20))
    stroke_risk = db.Column(db.String(20))
    doctor_recommendation = db.Column(db.String(100))
    diet_recommendation = db.Column(db.Text)
    health_tips = db.Column(db.Text)

class GlucoseReading(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    glucose_level = db.Column(db.Float, nullable=False)
    reading_type = db.Column(db.String(20))  # e.g., 'fasting', 'post-meal'
    notes = db.Column(db.Text)

class HealthTip(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(50), nullable=False)  # exercise, diet, sleep, stress
    age_group = db.Column(db.String(20))  # young, adult, senior
    risk_level = db.Column(db.String(20))  # low, moderate, high
    tip_content = db.Column(db.Text, nullable=False)

class DietRecommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    risk_level = db.Column(db.String(20), nullable=False)
    diet_plan = db.Column(db.Text, nullable=False)
    foods_to_eat = db.Column(db.Text)
    foods_to_avoid = db.Column(db.Text)
    meal_schedule = db.Column(db.Text) 