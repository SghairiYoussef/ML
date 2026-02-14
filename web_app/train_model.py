"""
Improved training script with better feature engineering and model optimization
This should be run before starting the web application
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_advanced_features(df):
    """Create additional engineered features"""
    df = df.copy()

    # Price per square meter (only for training, not for prediction input)
    # df['area_per_room'] = df['area_m2'] / (df['rooms'] + 1)  # Avoid division by zero

    # Total amenities score
    amenity_cols = ['garden', 'terrace', 'garage', 'elevator', 'swimming_pool',
                    'doorman', 'cellar', 'air_conditioning', 'heating', 'security',
                    'double_glazing', 'reinforced_door', 'equipped_kitchen']
    df['total_amenities'] = df[amenity_cols].sum(axis=1)

    # Luxury features (pool, doorman, etc.)
    df['has_luxury_features'] = ((df['swimming_pool'] == 1) |
                                  (df['doorman'] == 1) |
                                  (df['total_amenities'] >= 7)).astype(int)

    # Area categories
    df['area_category'] = pd.cut(df['area_m2'],
                                   bins=[0, 50, 100, 150, 200, 300, 1000000],
                                   labels=[0, 1, 2, 3, 4, 5])
    df['area_category'] = df['area_category'].astype(float)

    # Room categories
    df['room_category'] = pd.cut(df['rooms'],
                                   bins=[-1, 1, 2, 3, 4, 100],
                                   labels=[0, 1, 2, 3, 4])
    df['room_category'] = df['room_category'].astype(float)

    # Interaction features
    df['area_rooms_interaction'] = df['area_m2'] * df['rooms']

    return df

def remove_outliers(df, y, column='price'):
    """Remove extreme outliers using IQR method"""
    Q1 = y.quantile(0.05)  # 5th percentile (less aggressive)
    Q3 = y.quantile(0.95)  # 95th percentile (less aggressive)
    IQR = Q3 - Q1

    lower_bound = Q1 - 2.0 * IQR  # More lenient bounds
    upper_bound = Q3 + 2.0 * IQR

    mask = (y >= lower_bound) & (y <= upper_bound)
    print(f"   Removing {(~mask).sum()} outliers (kept {mask.sum()} rows)")

    return df[mask], y[mask]

def train_model():
    """Train and save an improved model for the web application"""

    print("="*60)
    print("TRAINING IMPROVED MODEL FOR WEB APPLICATION")
    print("="*60)

    # Paths
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_data_path = os.path.join(project_dir, 'data', 'train', 'trains.csv')
    model_dir = os.path.join(project_dir, 'model')
    model_path = os.path.join(model_dir, 'xgboost.joblib')
    scaler_path = os.path.join(model_dir, 'scaler.joblib')

    # Check if training data exists
    if not os.path.exists(train_data_path):
        print(f"\n❌ Training data not found at: {train_data_path}")
        print("\nPlease run the preprocessing pipeline first to generate training data:")
        print("  1. Go to the cleaning directory")
        print("  2. Run: python pipeline.py")
        return False

    # Load training data
    print(f"\n📂 Loading training data from: {train_data_path}")
    df_train = pd.read_csv(train_data_path)
    print(f"   Loaded {len(df_train)} rows, {len(df_train.columns)} columns")

    # Check if price column exists
    if 'price' not in df_train.columns:
        print("\n❌ Error: 'price' column not found in training data")
        return False

    # Display price statistics
    print(f"\n📊 Price Statistics:")
    print(f"   Min: {df_train['price'].min():,.0f}")
    print(f"   Max: {df_train['price'].max():,.0f}")
    print(f"   Mean: {df_train['price'].mean():,.0f}")
    print(f"   Median: {df_train['price'].median():,.0f}")

    # Prepare features and target
    print("\n🔧 Preparing features and target...")
    y = df_train['price']

    # Drop columns not used for training
    columns_to_drop = ['price', 'title', 'location']
    X = df_train.drop(columns=[col for col in columns_to_drop if col in df_train.columns])

    # Handle categorical variables BEFORE feature engineering
    if 'property_type' in X.columns:
        print("   Encoding property_type...")
        X = pd.get_dummies(X, columns=['property_type'], prefix='property_type')

    # Ensure all columns are numeric
    print("   Converting to numeric...")
    X = X.apply(pd.to_numeric, errors='coerce')

    # Fill NaN values
    print("   Filling missing values...")
    X = X.fillna(X.median())

    # Create advanced features
    print("   Creating advanced features...")
    X = create_advanced_features(X)

    # Remove outliers
    print("   Removing outliers...")
    X, y = remove_outliers(X, y)

    print(f"   Final feature shape: {X.shape}")
    print(f"   Number of features: {len(X.columns)}")

    # Split data for validation
    print("\n✂️ Splitting data for validation...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Validation set: {len(X_val)} samples")

    # Scale features (important for better predictions)
    print("\n📏 Scaling features...")
    scaler = StandardScaler()

    # Fit on training data only
    numeric_cols = ['area_m2', 'rooms', 'total_amenities', 'area_rooms_interaction']
    numeric_cols = [col for col in numeric_cols if col in X_train.columns]

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()

    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])

    # Train XGBoost model with optimized parameters
    print("\n🚀 Training XGBoost model with optimized parameters...")
    import xgboost as xgb

    # Improved hyperparameters - reduced complexity to prevent overfitting
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 5,                     # Shallower trees to reduce overfitting
        'learning_rate': 0.05,              # Lower learning rate
        'n_estimators': 300,                # Fewer trees
        'subsample': 0.7,                   # More aggressive subsampling
        'colsample_bytree': 0.7,            # More aggressive feature sampling
        'min_child_weight': 5,              # Higher value to prevent overfitting
        'gamma': 0.2,                       # Higher minimum loss reduction
        'reg_alpha': 0.5,                   # Stronger L1 regularization
        'reg_lambda': 2.0,                  # Stronger L2 regularization
        'random_state': 42,
        'n_jobs': -1,                       # Use all cores
        'verbosity': 0
    }

    model = xgb.XGBRegressor(**params)

    # Train with early stopping (updated for XGBoost 3.x)
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False
    )

    # Get best iteration if available
    if hasattr(model, 'best_iteration'):
        print(f"   Best iteration: {model.best_iteration}")
    else:
        print(f"   Training completed with {params['n_estimators']} estimators")

    # Evaluate model on training data
    print("\n📊 Evaluating model on TRAINING data...")
    y_train_pred = model.predict(X_train_scaled)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    # Calculate MAPE (Mean Absolute Percentage Error)
    train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100

    print(f"   - RMSE: {train_rmse:,.2f}")
    print(f"   - MAE: {train_mae:,.2f}")
    print(f"   - MAPE: {train_mape:.2f}%")
    print(f"   - R²: {train_r2:.4f}")

    # Evaluate model on validation data
    print("\n📊 Evaluating model on VALIDATION data...")
    y_val_pred = model.predict(X_val_scaled)

    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    val_mape = np.mean(np.abs((y_val - y_val_pred) / y_val)) * 100

    print(f"   - RMSE: {val_rmse:,.2f}")
    print(f"   - MAE: {val_mae:,.2f}")
    print(f"   - MAPE: {val_mape:.2f}%")
    print(f"   - R²: {val_r2:.4f}")

    # Check for overfitting
    if train_r2 - val_r2 > 0.1:
        print("\n⚠️ Warning: Model may be overfitting (large gap between train and val R²)")
    else:
        print("\n✅ Model shows good generalization!")

    # Feature importance
    print("\n🎯 Top 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X_train_scaled.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")

    # Save model
    print(f"\n💾 Saving model to: {model_path}")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, model_path)

    # Save scaler
    print(f"💾 Saving scaler to: {scaler_path}")
    joblib.dump(scaler, scaler_path)

    # Save scaling column names
    scaling_cols_path = os.path.join(model_dir, 'scaling_columns.txt')
    with open(scaling_cols_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(numeric_cols))
    print(f"💾 Saved scaling columns to: {scaling_cols_path}")

    # Verify save
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / 1024  # KB
        print(f"\n   ✅ Model saved successfully!")
        print(f"   - Size: {file_size:.2f} KB")
        print(f"   - Location: {model_path}")
    else:
        print(f"   ❌ Error: Failed to save model")
        return False

    # Save feature names for reference
    feature_names_path = os.path.join(model_dir, 'feature_names.txt')
    with open(feature_names_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(X.columns.tolist()))
    print(f"   ✅ Feature names saved to: {feature_names_path}")

    # Save feature importance
    importance_path = os.path.join(model_dir, 'feature_importance.csv')
    feature_importance.to_csv(importance_path, index=False)
    print(f"   ✅ Feature importance saved to: {importance_path}")

    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\n✨ Model Performance Summary:")
    print(f"   Training R²: {train_r2:.4f} | Validation R²: {val_r2:.4f}")
    print(f"   Training MAE: {train_mae:,.0f} | Validation MAE: {val_mae:,.0f}")
    print(f"   Training MAPE: {train_mape:.2f}% | Validation MAPE: {val_mape:.2f}%")
    print("\nYou can now start the web application:")
    print("  cd web_app")
    print("  python app.py")
    print("\n")

    return True


if __name__ == "__main__":
    success = train_model()
    if not success:
        sys.exit(1)

