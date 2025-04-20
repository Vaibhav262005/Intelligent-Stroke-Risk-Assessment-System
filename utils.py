import math
# Make serial optional
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

from datetime import datetime
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from errors import PDFGenerationError
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Add Twilio imports and setup
try:
    from twilio.rest import Client
    from twilio.base.exceptions import TwilioRestException
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    logging.warning("Twilio package not installed. SMS functionality will be disabled.")

# Load Twilio credentials from environment variables
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID', '')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN', '')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER', '')

def calculate_bmi(weight, height):
    """Calculate BMI and return category
    weight in kg, height in cm"""
    height_m = height / 100
    bmi = weight / (height_m * height_m)
    
    if bmi < 18.5:
        category = "Underweight"
    elif 18.5 <= bmi < 25:
        category = "Normal"
    elif 25 <= bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    
    return round(bmi, 2), category

def get_doctor_recommendation(risk_level, has_heart_disease, has_hypertension):
    """Get doctor recommendations based on risk factors"""
    recommendations = []
    
    if risk_level == "High":
        recommendations.append("Neurologist - Immediate consultation recommended")
        if has_heart_disease:
            recommendations.append("Cardiologist - Regular monitoring required")
    elif risk_level == "Moderate":
        recommendations.append("General Physician - Regular check-ups recommended")
        if has_hypertension:
            recommendations.append("Cardiologist - Periodic monitoring advised")
    else:
        recommendations.append("General Physician - Annual check-up recommended")
    
    return recommendations

def get_diet_recommendation(risk_level, bmi_category):
    """Get diet recommendations based on risk level and BMI"""
    diet_plan = {
        "High": {
            "description": "Low-sodium, low-fat diet with emphasis on heart-healthy foods",
            "foods_to_eat": [
                "Leafy greens", "Whole grains", "Lean proteins",
                "Fish rich in omega-3", "Berries", "Nuts"
            ],
            "foods_to_avoid": [
                "Salt and high-sodium foods",
                "Saturated fats",
                "Processed foods",
                "Red meat",
                "Added sugars"
            ]
        },
        "Moderate": {
            "description": "Heart-healthy diet with reduced sugar intake",
            "foods_to_eat": [
                "Fruits", "Vegetables", "Whole grains",
                "Lean proteins", "Low-fat dairy"
            ],
            "foods_to_avoid": [
                "Excess sugar",
                "Processed foods",
                "High-fat dairy"
            ]
        },
        "Low": {
            "description": "Balanced diet with variety of nutrients",
            "foods_to_eat": [
                "Variety of fruits and vegetables",
                "Whole grains",
                "Lean proteins",
                "Healthy fats"
            ],
            "foods_to_avoid": [
                "Excess processed foods",
                "Sugary drinks"
            ]
        }
    }
    
    # Adjust based on BMI
    if bmi_category == "Overweight" or bmi_category == "Obese":
        diet_plan[risk_level]["description"] += " with calorie restriction"
    
    return diet_plan[risk_level]

def get_health_tips(age, bmi_category, risk_level):
    """Get personalized health tips based on user factors"""
    tips = {
        "exercise": [],
        "sleep": [],
        "stress": []
    }
    
    # Exercise tips
    if age < 40:
        tips["exercise"] = [
            "Aim for 150 minutes of moderate aerobic activity per week",
            "Include strength training 2-3 times per week",
            "Try high-intensity interval training (HIIT)"
        ]
    else:
        tips["exercise"] = [
            "Start with low-impact exercises like walking or swimming",
            "Gradually increase activity level",
            "Include balance exercises"
        ]
    
    # Sleep tips
    tips["sleep"] = [
        "Aim for 7-9 hours of sleep per night",
        "Maintain a consistent sleep schedule",
        "Create a relaxing bedtime routine"
    ]
    
    # Stress management
    tips["stress"] = [
        "Practice daily meditation or deep breathing",
        "Take regular breaks during work",
        "Engage in relaxing activities"
    ]
    
    return tips

def read_glucose_sensor():
    """Read glucose data from serial port (Arduino/Raspberry Pi)"""
    if not SERIAL_AVAILABLE:
        return {
            'status': 'error',
            'message': 'Serial module not available'
        }
    
    try:
        # Try to find an available serial port
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            return {
                'status': 'error',
                'message': 'No serial ports available'
            }
            
        # Use the first available port
        port = ports[0].device
        
        # Configure the serial port
        ser = serial.Serial(
            port=port,  # Use detected port
            baudrate=9600,
            timeout=1
        )
        
        # Read data
        data = ser.readline().decode('utf-8').strip()
        ser.close()
        
        # Check if data is valid
        if not data:
            return {
                'status': 'error',
                'message': 'No data received from sensor'
            }
            
        # Parse the data (adjust based on your sensor's output format)
        glucose_level = float(data)
        timestamp = datetime.now()
        
        return {
            'glucose_level': glucose_level,
            'timestamp': timestamp,
            'status': 'success'
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

def generate_pdf_report(user, health_data, glucose_readings):
    """Generate a PDF report for the user with their health data and glucose readings"""
    try:
        # Create reports directory if it doesn't exist
        os.makedirs('reports', exist_ok=True)
        
        # Generate a unique filename
        filename = f"reports/health_report_{user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Create the PDF document
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # Add title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        elements.append(Paragraph(f"Health Report for {user.username}", title_style))
        elements.append(Spacer(1, 12))
        
        # Add user info
        elements.append(Paragraph("User Information", styles['Heading2']))
        user_data = [
            ["Username:", user.username],
            ["Email:", user.email],
            ["Report Date:", datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]
        user_table = Table(user_data, colWidths=[100, 300])
        user_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(user_table)
        elements.append(Spacer(1, 20))
        
        # Add health data
        if health_data:
            elements.append(Paragraph("Health Profile", styles['Heading2']))
            health_data_list = [
                ["Age:", f"{health_data.age} years"],
                ["Gender:", health_data.gender],
                ["Height:", f"{health_data.height} cm"],
                ["Weight:", f"{health_data.weight} kg"],
                ["Hypertension:", "Yes" if health_data.hypertension else "No"],
                ["Heart Disease:", "Yes" if health_data.heart_disease else "No"],
                ["Work Type:", health_data.work_type],
                ["Residence Type:", health_data.residence_type],
                ["Average Glucose Level:", f"{health_data.avg_glucose_level} mg/dL"],
                ["Smoking Status:", health_data.smoking_status]
            ]
            
            # Calculate BMI
            bmi, bmi_category = calculate_bmi(health_data.weight, health_data.height)
            health_data_list.append(["BMI:", f"{bmi} ({bmi_category})"])
            
            health_table = Table(health_data_list, colWidths=[150, 250])
            health_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
            ]))
            elements.append(health_table)
            elements.append(Spacer(1, 20))
            
            # Add health recommendations
            elements.append(Paragraph("Health Recommendations", styles['Heading2']))
            
            # Get doctor recommendations
            risk_level = "High" if health_data.hypertension or health_data.heart_disease else "Moderate"
            doctor_recs = get_doctor_recommendation(risk_level, health_data.heart_disease, health_data.hypertension)
            
            for rec in doctor_recs:
                elements.append(Paragraph(f"• {rec}", styles['Normal']))
            
            elements.append(Spacer(1, 12))
            
            # Get diet recommendations
            diet_recs = get_diet_recommendation(risk_level, bmi_category)
            elements.append(Paragraph("Diet Recommendations:", styles['Heading3']))
            elements.append(Paragraph(f"• {diet_recs['description']}", styles['Normal']))
            
            elements.append(Paragraph("Foods to Eat:", styles['Heading4']))
            for food in diet_recs['foods_to_eat']:
                elements.append(Paragraph(f"• {food}", styles['Normal']))
                
            elements.append(Paragraph("Foods to Avoid:", styles['Heading4']))
            for food in diet_recs['foods_to_avoid']:
                elements.append(Paragraph(f"• {food}", styles['Normal']))
            
            elements.append(Spacer(1, 12))
            
            # Get health tips
            health_tips = get_health_tips(health_data.age, bmi_category, risk_level)
            elements.append(Paragraph("Health Tips:", styles['Heading3']))
            
            elements.append(Paragraph("Exercise:", styles['Heading4']))
            for tip in health_tips['exercise']:
                elements.append(Paragraph(f"• {tip}", styles['Normal']))
                
            elements.append(Paragraph("Sleep:", styles['Heading4']))
            for tip in health_tips['sleep']:
                elements.append(Paragraph(f"• {tip}", styles['Normal']))
                
            elements.append(Paragraph("Stress Management:", styles['Heading4']))
            for tip in health_tips['stress']:
                elements.append(Paragraph(f"• {tip}", styles['Normal']))
            
            elements.append(Spacer(1, 20))
        
        # Add glucose readings
        if glucose_readings:
            elements.append(Paragraph("Recent Glucose Readings", styles['Heading2']))
            
            # Create table data
            table_data = [["Date", "Level (mg/dL)", "Type", "Notes"]]
            for reading in glucose_readings:
                table_data.append([
                    reading.timestamp.strftime('%Y-%m-%d %H:%M'),
                    f"{reading.glucose_level}",
                    reading.reading_type,
                    reading.notes or "-"
                ])
            
            # Create table
            table = Table(table_data, colWidths=[100, 80, 80, 200])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            elements.append(table)
        
        # Build the PDF
        doc.build(elements)
        
        return filename
    
    except Exception as e:
        raise PDFGenerationError(f"Error generating PDF report: {str(e)}")

def send_email_notification(recipient_email, subject, message):
    """
    Send an email notification using SMTP.
    
    Args:
        recipient_email (str): The recipient's email address
        subject (str): The subject of the email
        message (str): The body of the email (HTML)
    
    Returns:
        bool: True if email was sent successfully, False otherwise
    """
    try:
        # Get email configuration from environment variables
        smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.environ.get('SMTP_PORT', 587))
        sender_email = os.environ.get('SENDER_EMAIL', 'your-app-email@example.com')
        sender_password = os.environ.get('SENDER_PASSWORD', 'your-app-password')
        
        # Log email configuration for debugging
        logger.info(f"Email Configuration - Server: {smtp_server}, Port: {smtp_port}, Sender: {sender_email}")
        
        # Create a multipart message
        msg = MIMEMultipart()
        msg['From'] = f"Brain Stroke Prediction App <{sender_email}>"
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # Add body to email
        msg.attach(MIMEText(message, 'html'))
        
        # Connect to SMTP server
        logger.info(f"Connecting to SMTP server: {smtp_server}:{smtp_port}")
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Start TLS encryption
        
        # Login and send
        logger.info(f"Attempting to login with email: {sender_email}")
        server.login(sender_email, sender_password)
        
        logger.info(f"Sending email to: {recipient_email}")
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Email notification sent successfully to {recipient_email}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email notification: {str(e)}")
        return False

def format_glucose_notification(user, reading):
    """
    Format a notification message for a glucose reading.
    
    Args:
        user (User): The user object
        reading (GlucoseReading): The glucose reading object
    
    Returns:
        dict: A dictionary containing the email subject, email body, and alert level
    """
    reading_type = reading.reading_type.capitalize()
    glucose_level = reading.glucose_level
    
    # Determine if the reading is high, low, or normal
    if glucose_level > 180:
        status = "HIGH"
        alert_level = "warning"
    elif glucose_level < 70:
        status = "LOW"
        alert_level = "danger"
    else:
        status = "NORMAL"
        alert_level = "success"
    
    # Format email subject
    subject = f"Glucose Alert: {status} Reading - {glucose_level} mg/dL"
    
    # Format email body (HTML)
    email_body = f"""
    <html>
    <body>
        <h2>Glucose Reading Alert</h2>
        <p>Hello {user.username},</p>
        <p>A new glucose reading has been recorded:</p>
        <ul>
            <li><strong>Reading Type:</strong> {reading_type}</li>
            <li><strong>Glucose Level:</strong> <span style="color: {'red' if status == 'HIGH' or status == 'LOW' else 'green'}">{glucose_level} mg/dL ({status})</span></li>
            <li><strong>Date/Time:</strong> {reading.timestamp.strftime('%Y-%m-%d %H:%M')}</li>
        </ul>
        <p>
            {'<strong style="color: red">Your glucose level is high. Consider checking with your healthcare provider.</strong>' if status == 'HIGH' else ''}
            {'<strong style="color: red">Your glucose level is low. Consider having a snack or glucose tablet.</strong>' if status == 'LOW' else ''}
        </p>
        <p>Stay healthy!</p>
        <p>- Brain Stroke Prediction App</p>
    </body>
    </html>
    """
    
    return {
        'subject': subject,
        'email_body': email_body,
        'alert_level': alert_level
    }

def format_prediction_notification(user, prediction_text, risk_percentage):
    """
    Format a notification message for a stroke risk prediction.
    
    Args:
        user (User): The user object
        prediction_text (str): The prediction result text
        risk_percentage (float): The risk percentage
    
    Returns:
        dict: A dictionary containing the email subject, email body, SMS body, and alert level
    """
    # Determine risk level
    if risk_percentage > 15:
        risk_level = "HIGH"
        alert_level = "danger"
    elif risk_percentage > 5:
        risk_level = "MODERATE"
        alert_level = "warning"
    else:
        risk_level = "LOW"
        alert_level = "success"
    
    # Format email subject
    subject = f"Stroke Risk Alert: {risk_level} Risk - {risk_percentage:.2f}%"
    
    # Format email body (HTML)
    email_body = f"""
    <html>
    <body>
        <h2>Stroke Risk Prediction Alert</h2>
        <p>Hello {user.username},</p>
        <p>A new stroke risk prediction has been calculated:</p>
        <p><strong>Result:</strong> <span style="color: {'red' if risk_level == 'HIGH' else 'orange' if risk_level == 'MODERATE' else 'green'}">{prediction_text}</span></p>
        <p>
            {'<strong style="color: red">Your stroke risk is high. Please consult with your healthcare provider as soon as possible.</strong>' if risk_level == 'HIGH' else ''}
            {'<strong style="color: orange">Your stroke risk is moderate. Consider discussing with your healthcare provider at your next visit.</strong>' if risk_level == 'MODERATE' else ''}
        </p>
        <p>Stay healthy!</p>
        <p>- Brain Stroke Prediction App</p>
    </body>
    </html>
    """
    
    # Format SMS body (plain text)
    sms_body = f"Brain Stroke Alert: {risk_level} RISK ({risk_percentage:.2f}%). "
    
    if risk_level == "HIGH":
        sms_body += "Please consult your doctor ASAP."
    elif risk_level == "MODERATE":
        sms_body += "Consider discussing with your doctor soon."
    else:
        sms_body += "Keep up the good health habits."
    
    return {
        'subject': subject,
        'email_body': email_body,
        'sms_body': sms_body,
        'alert_level': alert_level
    }

def send_sms(to_number, message):
    """
    Send an SMS message using Twilio.
    
    Args:
        to_number (str): The recipient's phone number with country code (e.g., +1234567890)
        message (str): The message to send
        
    Returns:
        bool: True if the message was sent successfully, False otherwise
        dict: Contains error details if any occurred
    """
    if not TWILIO_AVAILABLE:
        return False, {"error": "Twilio package not installed"}
    
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
        return False, {"error": "Twilio credentials not configured"}
    
    if not to_number:
        return False, {"error": "Recipient phone number not provided"}
    
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=to_number
        )
        return True, {"message_sid": message.sid}
    except TwilioRestException as e:
        logging.error(f"Twilio error: {str(e)}")
        return False, {"error": str(e)}
    except Exception as e:
        logging.error(f"Error sending SMS: {str(e)}")
        return False, {"error": "Unknown error occurred while sending SMS"} 