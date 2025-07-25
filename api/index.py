from app import app

# This file is needed for Vercel's serverless functions deployment
# It imports the Flask app from app.py

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000) 