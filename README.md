# Brain Stroke Risk Prediction System

A comprehensive web application for monitoring health metrics and predicting stroke risk using machine learning. This application helps users track their health data, glucose readings, and provides risk assessment based on various health parameters.

## Features

- **User Authentication**
  - Secure registration and login system
  - Profile management
  - Password protection

- **Health Data Management**
  - Record and update personal health metrics
  - Track vital signs and health conditions
  - Monitor glucose readings over time

- **Risk Assessment**
  - Machine learning-based stroke risk prediction
  - Real-time health status monitoring
  - Comprehensive health reports

- **Notification System**
  - Email notifications for health alerts
  - SMS notifications for critical updates
  - Customizable notification preferences

- **Data Visualization**
  - Interactive dashboard
  - Health metrics tracking
  - Glucose reading history

## Technology Stack

- **Backend**
  - Python 3.8+
  - Flask (Web Framework)
  - SQLAlchemy (ORM)
  - Flask-Login (Authentication)
  - Flask-WTF (Forms)
  - ReportLab (PDF Generation)
  - Twilio (SMS Notifications)

- **Frontend**
  - HTML5
  - CSS3
  - Bootstrap 5
  - Bootstrap Icons

- **Database**
  - SQLite (Development)
  - PostgreSQL (Production)

- **Machine Learning**
  - Scikit-learn
  - Pandas
  - NumPy

## Prerequisites

Before running the application, ensure you have the following installed:

1. Python 3.8 or higher
2. pip (Python package manager)
3. virtualenv (recommended)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/brain-stroke-prediction.git
   cd brain-stroke-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   # On Windows
   set FLASK_APP=app.py
   set FLASK_ENV=development
   set SECRET_KEY=your-secret-key
   
   # For SMS notifications (Twilio)
   set TWILIO_ACCOUNT_SID=your-twilio-account-sid
   set TWILIO_AUTH_TOKEN=your-twilio-auth-token
   set TWILIO_PHONE_NUMBER=your-twilio-phone-number

   # On macOS/Linux
   export FLASK_APP=app.py
   export FLASK_ENV=development
   export SECRET_KEY=your-secret-key
   
   # For SMS notifications (Twilio)
   export TWILIO_ACCOUNT_SID=your-twilio-account-sid
   export TWILIO_AUTH_TOKEN=your-twilio-auth-token
   export TWILIO_PHONE_NUMBER=your-twilio-phone-number
   ```

5. Initialize the database:
   ```bash
   flask db init
   flask db migrate
   flask db upgrade
   ```

## Running the Application

1. Start the Flask development server:
   ```bash
   flask run
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Project Structure

```
brain-stroke-prediction/
├── app/
│   ├── __init__.py
│   ├── models.py
│   ├── forms.py
│   ├── routes.py
│   └── utils.py
├── migrations/
├── static/
│   ├── css/
│   ├── js/
│   └── img/
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── health_data.html
│   ├── glucose_reading.html
│   └── profile.html
├── tests/
├── .env
├── .gitignore
├── app.py
├── config.py
├── requirements.txt
└── README.md
```

## Usage Guide

1. **Registration and Login**
   - Create a new account using the registration form
   - Log in with your credentials
   - Access your profile to manage account settings

2. **Health Data Entry**
   - Navigate to the Health Data page
   - Enter your personal health metrics
   - Update information as needed

3. **Glucose Reading Management**
   - Record new glucose readings
   - View reading history
   - Track trends over time

4. **Dashboard and Reports**
   - View your health profile
   - Monitor recent glucose readings
   - Generate comprehensive health reports

5. **Notification Management**
   - Configure email notifications
   - Set up SMS alerts (requires phone number)
   - Customize notification preferences

## SMS Notification Setup

To enable SMS notifications:

1. Create a Twilio account at [https://www.twilio.com](https://www.twilio.com)
2. Obtain your Account SID and Auth Token from the Twilio Dashboard
3. Purchase a Twilio phone number
4. Set the required environment variables (see Installation section)
5. Add your phone number in your profile settings
6. Enable SMS notifications in your notification settings

**Note**: The application will gracefully handle cases where Twilio isn't configured or when users haven't set a phone number.

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository or contact the development team.

## Acknowledgments

- Bootstrap for the frontend framework
- Flask team for the web framework
- Twilio for SMS notification services
- All contributors and maintainers

## Security Considerations

- Keep your SECRET_KEY secure and never commit it to version control
- Use environment variables for sensitive information
- Regularly update dependencies to patch security vulnerabilities
- Implement proper input validation and sanitization
- Use HTTPS in production

## Deployment

For production deployment:

1. Set up a production-grade web server (e.g., Gunicorn)
2. Configure a reverse proxy (e.g., Nginx)
3. Set up SSL/TLS certificates
4. Use a production-grade database
5. Configure proper logging and monitoring
6. Set up backup procedures

## Maintenance

Regular maintenance tasks:

1. Update dependencies
2. Monitor error logs
3. Backup database
4. Check system health
5. Review security measures
6. Update documentation 