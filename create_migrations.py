from app import app, db, migrate
from flask_migrate import init, migrate as migrate_command, upgrade
import os

# Create an application context
with app.app_context():
    # Check if migrations directory already exists
    if not os.path.exists('migrations'):
        # Initialize migrations
        init(directory='migrations')
        print("Migrations directory created!")
    
    # Create a migration
    migrate_command(directory='migrations', message='Initial migration')
    print("Migration created!")
    
    # Apply the migration
    upgrade(directory='migrations')
    print("Database migrated successfully!") 