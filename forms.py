from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, FloatField, IntegerField, SelectField, TextAreaField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Email, Length, EqualTo, NumberRange, Optional, Regexp

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')

class HealthDataForm(FlaskForm):
    weight = FloatField('Weight (kg)', validators=[DataRequired(), NumberRange(min=20, max=300)])
    height = FloatField('Height (cm)', validators=[DataRequired(), NumberRange(min=100, max=250)])
    age = IntegerField('Age', validators=[DataRequired(), NumberRange(min=1, max=120)])
    gender = SelectField('Gender', choices=[('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other')], validators=[DataRequired()])
    hypertension = SelectField('Hypertension', choices=[('0', 'No'), ('1', 'Yes')], validators=[DataRequired()])
    heart_disease = SelectField('Heart Disease', choices=[('0', 'No'), ('1', 'Yes')], validators=[DataRequired()])
    ever_married = SelectField('Ever Married', choices=[('No', 'No'), ('Yes', 'Yes')], validators=[DataRequired()])
    work_type = SelectField('Work Type', 
                          choices=[('Private', 'Private'), 
                                 ('Self-employed', 'Self-employed'),
                                 ('Govt_job', 'Government Job'),
                                 ('children', 'Children'),
                                 ('Never_worked', 'Never Worked')],
                          validators=[DataRequired()])
    residence_type = SelectField('Residence Type', 
                               choices=[('Urban', 'Urban'), ('Rural', 'Rural')],
                               validators=[DataRequired()])
    avg_glucose_level = FloatField('Average Glucose Level (mg/dL)', validators=[DataRequired(), NumberRange(min=50, max=400)])
    smoking_status = SelectField('Smoking Status',
                               choices=[('never smoked', 'Never Smoked'),
                                      ('formerly smoked', 'Formerly Smoked'),
                                      ('smokes', 'Currently Smoking'),
                                      ('Unknown', 'Unknown')],
                               validators=[DataRequired()])
    submit = SubmitField('Save Health Data')

class GlucoseReadingForm(FlaskForm):
    glucose_level = FloatField('Glucose Level', validators=[DataRequired(), NumberRange(min=50, max=400)])
    reading_type = SelectField('Reading Type', 
                             choices=[('fasting', 'Fasting'),
                                    ('post_meal', 'Post Meal'),
                                    ('random', 'Random')],
                             validators=[DataRequired()])
    notes = TextAreaField('Notes')
    
    # Notification options
    send_email_notification = BooleanField('Send Email Notification')
    
    submit = SubmitField('Save Reading')

class ProfileEditForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    phone_number = StringField('Phone Number', validators=[Optional(), Regexp(r'^\+?[0-9]{10,15}$', message="Invalid phone number format. Use country code and number (e.g., +1234567890).")])
    bio = TextAreaField('Bio', validators=[Length(max=200)])
    submit = SubmitField('Update Profile')

class PasswordChangeForm(FlaskForm):
    current_password = PasswordField('Current Password', validators=[DataRequired()])
    new_password = PasswordField('New Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm New Password', validators=[DataRequired(), EqualTo('new_password')])
    submit = SubmitField('Change Password')

class NotificationSettingsForm(FlaskForm):
    email_notifications = BooleanField('Email Notifications')
    sms_notifications = BooleanField('SMS Notifications')
    health_reminders = BooleanField('Health Check Reminders')
    glucose_reminders = BooleanField('Glucose Reading Reminders')
    medication_reminders = BooleanField('Medication Reminders')
    app_updates = BooleanField('App Updates')
    submit = SubmitField('Save Notification Settings')

class PrivacySettingsForm(FlaskForm):
    share_health_data = BooleanField('Share Health Data for Research (Anonymized)')
    public_profile = BooleanField('Public Profile')
    data_retention = SelectField('Data Retention Period', 
                              choices=[
                                  ('1', '1 Year'),
                                  ('3', '3 Years'),
                                  ('5', '5 Years'),
                                  ('forever', 'Forever')
                              ])
    submit = SubmitField('Save Privacy Settings')

class PredictionNotificationForm(FlaskForm):
    send_email_notification = BooleanField('Send Email Notification')
    send_sms_notification = BooleanField('Send SMS Notification')
    submit = SubmitField('Submit') 