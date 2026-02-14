"""
Simple launcher for the Housing Price Prediction Web App
Run this file to start the application
"""

import sys
import os

def main():
    print("="*60)
    print("Housing Price Prediction Web Application")
    print("="*60)

    # Check Flask
    try:
        import flask
        print(f"✓ Flask {flask.__version__} found")
    except ImportError:
        print("✗ Flask not found. Installing dependencies...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "../requierements.txt"])
        print("✓ Dependencies installed")

    # Check model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'xgboost.joblib')
    if os.path.exists(model_path):
        print(f"✓ Model found: {model_path}")
    else:
        print("✗ Model not found. Training model...")
        import train_model
        if not train_model.train_model():
            print("✗ Failed to train model")
            sys.exit(1)

    # Start app
    print("\n" + "="*60)
    print("Starting web server on http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")

    # Import and run
    from app import app
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")

