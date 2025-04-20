from app import app, db
from models import User, HealthData, HealthRecord, GlucoseReading, HealthTip, DietRecommendation

# Create an application context
with app.app_context():
    # Create all tables
    db.create_all()
    print("Database tables created successfully!")
    
    # You can add sample data here if needed
    # For example:
    # if User.query.count() == 0:
    #    admin = User(username="admin", email="admin@example.com", password="hashed_password")
    #    db.session.add(admin)
    #    db.session.commit() 