#!/usr/bin/env python3
"""
Deployment script for MicrobeLLM web interface on Render
Downloads database if needed and starts the web server
"""

import os
import sys
import subprocess
from pathlib import Path

def download_database():
    """Download the database file if it doesn't exist"""
    db_path = Path("microbellm.db")
    
    # Check if database already exists
    if db_path.exists():
        print(f"Database already exists: {db_path}")
        return
    
    # Get database URL from environment variable or use default
    db_url = os.getenv('MICROBELLM_DB_URL', 'https://f000.backblazeb2.com/file/llm-bioeval/microbellm.db')
    
    print(f"Downloading database from {db_url}...")
    
    try:
        # Use wget or curl to download the database
        result = subprocess.run(
            ["wget", "-O", "microbellm.db", db_url],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            # Try curl if wget fails
            print("wget failed, trying curl...")
            result = subprocess.run(
                ["curl", "-L", "-o", "microbellm.db", db_url],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"Error downloading database: {result.stderr}")
                sys.exit(1)
        
        print("Database downloaded successfully")
        
    except Exception as e:
        print(f"Error downloading database: {e}")
        sys.exit(1)

def main():
    """Main entry point for the web application"""
    
    # Download database if needed
    download_database()
    
    # Check if we can import required modules
    try:
        import flask
        import flask_socketio
        import pandas
        import numpy
    except ImportError as e:
        print(f"Error: Missing required module: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        sys.exit(1)
    
    # Import and run the web app
    try:
        from microbellm.web_app import app
    except ImportError as e:
        print(f"Error importing web app: {e}")
        sys.exit(1)
    
    # Get port from environment variable (Render sets this)
    port = int(os.getenv('PORT', 5000))
    host = '0.0.0.0'  # Need to bind to all interfaces for Render
    
    # Check for API key
    if not os.getenv('OPENROUTER_API_KEY'):
        print("=" * 50)
        print("WARNING: OPENROUTER_API_KEY not set!")
        print("=" * 50)
        print("Prediction features will not be available.")
        print("=" * 50)
        print()
    
    print(f"Starting MicrobeLLM Web Interface on {host}:{port}...")
    
    # Run with gunicorn if available (production), otherwise use Flask dev server
    try:
        import gunicorn
        # Gunicorn is available, use it for production
        cmd = [
            "gunicorn",
            "--bind", f"{host}:{port}",
            "--worker-class", "eventlet",
            "--workers", "1",
            "--timeout", "120",
            "--log-level", "info",
            "microbellm.web_app:app"
        ]
        print(f"Starting with gunicorn: {' '.join(cmd)}")
        subprocess.run(cmd)
    except ImportError:
        # Fall back to Flask development server
        print("WARNING: Running with Flask development server (not for production)")
        app.run(host=host, port=port, debug=False)

if __name__ == '__main__':
    main()