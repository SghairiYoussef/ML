"""
Quick Script to Train XGBoost Advanced Ensemble Model
Run this script to train the missing advanced model
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    required = ['pandas', 'numpy', 'xgboost', 'sklearn', 'joblib']
    missing = []

    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    return missing

def install_dependencies(packages):
    """Install missing packages"""
    print(f"\n📦 Installing missing packages: {', '.join(packages)}")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)
        return True
    except Exception as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_training_data():
    """Check if training data exists"""
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(project_dir, 'data', 'train', 'trains.csv')

    if not os.path.exists(train_path):
        print(f"❌ Training data not found at: {train_path}")
        print(f"   Please ensure training data exists before running this script.")
        return False

    print(f"✅ Training data found: {train_path}")
    return True

def train_model():
    """Train the advanced model"""
    try:
        from train_advanced_model import train_advanced_model
        print("\n🚀 Starting model training...")
        print("="*70)
        success = train_advanced_model()
        return success
    except Exception as e:
        print(f"\n❌ Training failed with error:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def update_metrics():
    """Prompt user to update metrics file"""
    print("\n" + "="*70)
    print("📝 IMPORTANT: Update Model Metrics")
    print("="*70)
    print("\nPlease add the model metrics to: model/models_metrics.json")
    print("\nAdd this entry (use actual values from training output):")
    print("""
  "xgboost_advanced": {
    "r2": "0.72",
    "mae": "165000",
    "mape": "68.5",
    "rmse": "320000"
  }
""")
    print("\nThen restart your web application.")

def main():
    print("="*70)
    print("XGBoost Advanced Ensemble Model - Quick Training")
    print("="*70)

    # Step 1: Check dependencies
    print("\n🔍 Checking dependencies...")
    missing = check_dependencies()

    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        response = input("\nInstall missing packages now? (y/n): ").strip().lower()

        if response == 'y':
            if not install_dependencies(missing):
                print("\n❌ Failed to install dependencies.")
                print("   Please install manually: pip install -r requierements.txt")
                return False
        else:
            print("\n❌ Cannot proceed without dependencies.")
            print("   Install manually: pip install -r requierements.txt")
            return False
    else:
        print("✅ All dependencies installed")

    # Step 2: Check training data
    print("\n🔍 Checking training data...")
    if not check_training_data():
        return False

    # Step 3: Train model
    print("\n" + "="*70)
    print("Starting Training Process")
    print("="*70)
    print("\n⏱️  This may take 5-10 minutes depending on your hardware...")
    print("💡 You'll see progress updates as training proceeds.\n")

    success = train_model()

    if success:
        print("\n" + "="*70)
        print("✅ SUCCESS! Model Training Completed")
        print("="*70)
        update_metrics()
        print("\n✨ Your advanced model is ready to use!")
        return True
    else:
        print("\n" + "="*70)
        print("❌ Training Failed")
        print("="*70)
        print("\nPlease check the error messages above.")
        print("Common issues:")
        print("  - Missing or corrupt training data")
        print("  - Insufficient memory")
        print("  - Data quality problems")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

