"""
Flask Web Application for Housing Price Prediction
"""

from flask import Flask, render_template, request, jsonify, send_file, session
import pandas as pd
import numpy as np
import os
import sys
from werkzeug.utils import secure_filename
import joblib
import traceback
from datetime import datetime
import json

# Add parent directory to path to import pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cleaning.pipeline import (
    drop_unnecessary_columns, remove_missing_price, update_features_from_title,
    remove_residence_rows, fill_property_type_from_title, remove_missing_property_type,
    remove_all_missing_key_features, impute_rooms_and_area, final_imputation
)
from inferance.inferance import load_model, predict

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['MODEL_PATH'] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'xgboost_advanced.joblib')

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(os.path.dirname(app.config['MODEL_PATH']), exist_ok=True)

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_location_features(df):
    """Extract city, district, and region from location (same as training)"""
    df = df.copy()

    # Split location into parts
    df['location_str'] = df['location'].fillna('Unknown, Unknown')
    location_parts = df['location_str'].str.split(',', expand=True)

    df['city'] = location_parts[0].str.strip() if location_parts.shape[1] > 0 else 'Unknown'
    df['district'] = location_parts[1].str.strip() if location_parts.shape[1] > 1 else 'Unknown'

    # Define premium locations
    premium_cities = [
        'Gammarth', 'La Marsa', 'Carthage', 'Sidi Bou Said',
        'Les Jardins de Carthage', 'Ain Zaghouan Nord', 'La Soukra'
    ]

    mid_cities = [
        'Ariana', 'Ennasr', 'Menzah', 'Manar', 'Lac', 'Berges du Lac',
        'Sousse', 'Hammamet', 'Monastir', 'Sfax'
    ]

    df['location_tier'] = 2  # Standard
    df.loc[df['city'].isin(premium_cities), 'location_tier'] = 3  # Premium
    df.loc[df['city'].isin(mid_cities), 'location_tier'] = 2.5  # Mid-tier

    # Coastal indicator
    coastal_keywords = ['Hammamet', 'La Marsa', 'Gammarth', 'Carthage', 'Sousse',
                        'Monastir', 'Bizerte', 'Klibia', 'Mahdia']
    df['is_coastal'] = df['city'].apply(
        lambda x: 1 if any(kw in str(x) for kw in coastal_keywords) else 0
    )

    # Tourist area indicator
    tourist_areas = ['Hammamet', 'Sousse', 'Monastir', 'Djerba', 'Mahdia']
    df['is_tourist_area'] = df['city'].apply(
        lambda x: 1 if any(kw in str(x) for kw in tourist_areas) else 0
    )

    return df


def encode_location_frequency(df, column, frequencies):
    """Encode location by frequency (use pre-calculated frequencies from training)"""
    df[f'{column}_frequency'] = df[column].map(frequencies).fillna(0)
    df[f'{column}_is_common'] = (df[f'{column}_frequency'] >= 10).astype(int)
    return df


def create_advanced_features(df):
    """Create additional engineered features (same as training)"""
    df = df.copy()

    # Total amenities score
    amenity_cols = ['garden', 'terrace', 'garage', 'elevator', 'swimming_pool',
                    'doorman', 'cellar', 'air_conditioning', 'heating', 'security',
                    'double_glazing', 'reinforced_door', 'equipped_kitchen']
    df['total_amenities'] = df[amenity_cols].sum(axis=1)

    # Premium amenities (more valuable)
    premium_amenities = ['swimming_pool', 'doorman', 'elevator', 'air_conditioning']
    df['premium_amenities'] = df[premium_amenities].sum(axis=1)

    # Basic amenities
    basic_amenities = ['heating', 'equipped_kitchen', 'garage']
    df['basic_amenities'] = df[basic_amenities].sum(axis=1)

    # Security features
    security_features = ['security', 'doorman', 'reinforced_door', 'double_glazing']
    df['security_score'] = df[security_features].sum(axis=1)

    # Luxury features (pool, doorman, etc.)
    df['has_luxury_features'] = ((df['swimming_pool'] == 1) |
                                  (df['doorman'] == 1) |
                                  (df['total_amenities'] >= 7)).astype(int)

    # Area categories
    df['area_category'] = pd.cut(df['area_m2'],
                                   bins=[0, 50, 100, 150, 200, 300, 500, 10000],
                                   labels=[0, 1, 2, 3, 4, 5, 6])
    df['area_category'] = df['area_category'].astype(float).fillna(0)

    # Room categories
    df['room_category'] = pd.cut(df['rooms'],
                                   bins=[-1, 1, 2, 3, 4, 6, 100],
                                   labels=[0, 1, 2, 3, 4, 5])
    df['room_category'] = df['room_category'].astype(float).fillna(0)

    # Interaction features
    df['area_rooms_interaction'] = df['area_m2'] * df['rooms']
    df['area_per_room'] = df['area_m2'] / (df['rooms'] + 1)  # Avoid division by zero

    # Quality indicators
    df['amenity_density'] = df['total_amenities'] / (df['area_m2'] / 100 + 1)
    df['space_quality'] = df['area_per_room'] * (1 + df['total_amenities'] / 10)

    # Price indicators (interaction with location)
    df['location_quality_score'] = df['location_tier'] * (1 + df['total_amenities'] / 10)
    df['coastal_premium_score'] = df['is_coastal'] * df['has_luxury_features'] * df['area_category']

    return df


def preprocess_for_prediction(df):
    """
    Apply preprocessing pipeline for prediction
    Removes target column if present and applies all transformations
    """
    try:
        # Store original data for display
        original_df = df.copy()

        # Remove price column if present (for display purposes)
        has_price = 'price' in df.columns
        if has_price:
            df = df.drop(columns=['price'])

        # 1. Drop unnecessary columns (except location - we need it first!)
        columns_to_drop_early = ['scraped_at', 'url', 'bathrooms']
        df = df.drop(columns=[col for col in columns_to_drop_early if col in df.columns])

        # 2. Extract features from title
        df = update_features_from_title(df)

        # 3. Remove residence rows without info
        df = remove_residence_rows(df)

        # 4. Fill property type from title
        df = fill_property_type_from_title(df)

        # 5. Remove rows without property type
        initial_len = len(df)
        df = remove_missing_property_type(df)
        removed_rows = initial_len - len(df)

        # 6. Remove rows with all key features missing
        df = remove_all_missing_key_features(df)

        # 7. Impute rooms and area
        df = impute_rooms_and_area(df)

        # 8. Final imputation
        df = final_imputation(df)

        # 9. Ensure all feature columns exist and fill missing ones
        feature_columns = [
            'area_m2', 'rooms', 'swimming_pool', 'garden', 'terrace',
            'garage', 'elevator', 'air_conditioning', 'heating',
            'equipped_kitchen', 'security', 'doorman', 'cellar',
            'double_glazing', 'reinforced_door'
        ]

        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        # 10. EXTRACT LOCATION FEATURES (before dropping location!)
        if 'location' in df.columns:
            df = extract_location_features(df)

            # Load city/district frequencies from training (for encoding)
            model_dir = os.path.dirname(app.config['MODEL_PATH'])
            # Use default frequencies if files don't exist
            city_freq = {'Unknown': 0}
            district_freq = {'Unknown': 0}

            df = encode_location_frequency(df, 'city', city_freq)
            df = encode_location_frequency(df, 'district', district_freq)

            # Drop original location columns
            df = df.drop(columns=['location', 'location_str', 'city', 'district'], errors='ignore')

        # 11. Encode property_type using one-hot encoding
        if 'property_type' in df.columns:
            df = pd.get_dummies(df, columns=['property_type'], prefix='property_type')

        # 12. Drop non-numeric columns (title)
        df = df.drop(columns=['title'], errors='ignore')

        # 13. Ensure all values are numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        # 14. Fill any remaining NaN values with median
        df = df.fillna(df.median()).fillna(0)

        # 15. Create advanced features (SAME AS TRAINING)
        df = create_advanced_features(df)

        return df, original_df, removed_rows, None

    except Exception as e:
        return None, None, 0, str(e)


@app.route('/')
def index():
    """Render home page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and initial validation"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only CSV files are allowed'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # Load and validate CSV
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            os.remove(filepath)
            return jsonify({'error': f'Invalid CSV file: {str(e)}'}), 400

        if len(df) == 0:
            os.remove(filepath)
            return jsonify({'error': 'CSV file is empty'}), 400

        # Store filename in session
        session['uploaded_file'] = unique_filename

        # Get preview data
        preview_data = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'preview': df.head(10).to_dict('records'),
            'filename': filename
        }

        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'data': preview_data
        })

    except Exception as e:
        return jsonify({'error': f'Upload error: {str(e)}'}), 500


@app.route('/preprocess', methods=['POST'])
def preprocess():
    """Preprocess the uploaded data"""
    try:
        if 'uploaded_file' not in session:
            return jsonify({'error': 'No file uploaded'}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['uploaded_file'])

        if not os.path.exists(filepath):
            return jsonify({'error': 'Uploaded file not found'}), 404

        # Load data
        df = pd.read_csv(filepath)

        # Preprocess
        processed_df, original_df, removed_rows, error = preprocess_for_prediction(df)

        if error:
            return jsonify({'error': f'Preprocessing error: {error}'}), 500

        # Save processed data
        processed_filename = f"processed_{session['uploaded_file']}"
        processed_filepath = os.path.join(app.config['RESULTS_FOLDER'], processed_filename)
        processed_df.to_csv(processed_filepath, index=False)
        session['processed_file'] = processed_filename

        return jsonify({
            'success': True,
            'message': 'Data preprocessed successfully',
            'data': {
                'original_rows': len(original_df),
                'processed_rows': len(processed_df),
                'removed_rows': removed_rows,
                'features': processed_df.columns.tolist()
            }
        })

    except Exception as e:
        return jsonify({'error': f'Preprocessing error: {str(e)}\n{traceback.format_exc()}'}), 500


@app.route('/predict', methods=['POST'])
def make_prediction():
    """Make predictions on preprocessed data"""
    try:
        if 'processed_file' not in session:
            return jsonify({'error': 'No preprocessed data available'}), 400

        processed_filepath = os.path.join(app.config['RESULTS_FOLDER'], session['processed_file'])

        if not os.path.exists(processed_filepath):
            return jsonify({'error': 'Preprocessed file not found'}), 404

        # Check if model exists
        if not os.path.exists(app.config['MODEL_PATH']):
            return jsonify({
                'error': 'Model not found. Please train a model first.',
                'suggestion': 'Run the training script to create a model file at: ' + app.config['MODEL_PATH']
            }), 404

        # Load processed data
        df_processed = pd.read_csv(processed_filepath)

        # Load model
        model = joblib.load(app.config['MODEL_PATH'])

        # Check if it's an ensemble model
        model_dir = os.path.dirname(app.config['MODEL_PATH'])
        model_type_path = os.path.join(model_dir, 'model_type.txt')
        is_ensemble = False
        if os.path.exists(model_type_path):
            with open(model_type_path, 'r') as f:
                model_type = f.read().strip()
                is_ensemble = (model_type == 'ensemble')

        # Load scaler (try advanced first, then fallback)
        scaler_path = os.path.join(model_dir, 'scaler_advanced.joblib')
        scaling_cols_path = os.path.join(model_dir, 'scaling_columns_advanced.txt')

        if not os.path.exists(scaler_path):
            scaler_path = os.path.join(model_dir, 'scaler.joblib')
            scaling_cols_path = os.path.join(model_dir, 'scaling_columns.txt')

        if os.path.exists(scaler_path) and os.path.exists(scaling_cols_path):
            scaler = joblib.load(scaler_path)
            with open(scaling_cols_path, 'r', encoding='utf-8') as f:
                scaling_cols = [line.strip() for line in f.readlines()]

            # Apply scaling to the same columns as training
            df_scaled = df_processed.copy()
            existing_cols = [col for col in scaling_cols if col in df_scaled.columns]
            if existing_cols:
                df_scaled[existing_cols] = scaler.transform(df_scaled[existing_cols])
            df_processed = df_scaled

        # Load training feature names to align features (try advanced first)
        feature_names_path = os.path.join(model_dir, 'feature_names_advanced.txt')
        if not os.path.exists(feature_names_path):
            feature_names_path = os.path.join(model_dir, 'feature_names.txt')

        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r', encoding='utf-8') as f:
                training_features = [line.strip() for line in f.readlines()]

            # Add missing features with zeros
            for col in training_features:
                if col not in df_processed.columns:
                    df_processed[col] = 0

            # Reorder columns to match training
            df_processed = df_processed[training_features]

        # Make predictions based on model type
        if is_ensemble and isinstance(model, dict):
            # Ensemble prediction
            model_xgb = model['xgb']
            model_rf = model['rf']
            weights = model['weights']

            pred_xgb = model_xgb.predict(df_processed)
            pred_rf = model_rf.predict(df_processed)
            predictions = weights[0] * pred_xgb + weights[1] * pred_rf
        else:
            # Single model prediction
            predictions = model.predict(df_processed)

        # Load original data for display
        original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['uploaded_file'])
        df_original = pd.read_csv(original_filepath)

        # Create results dataframe
        df_results = df_original.copy()

        # Handle removed rows during preprocessing
        if len(predictions) < len(df_original):
            # Mark removed rows
            df_results['predicted_price'] = np.nan
            df_results.loc[:len(predictions)-1, 'predicted_price'] = predictions
        else:
            df_results['predicted_price'] = predictions

        # Save results
        results_filename = f"results_{session['uploaded_file']}"
        results_filepath = os.path.join(app.config['RESULTS_FOLDER'], results_filename)
        df_results.to_csv(results_filepath, index=False)
        session['results_file'] = results_filename

        # Calculate statistics
        valid_predictions = predictions[~np.isnan(predictions)]
        stats = {
            'count': len(valid_predictions),
            'mean': float(np.mean(valid_predictions)),
            'median': float(np.median(valid_predictions)),
            'min': float(np.min(valid_predictions)),
            'max': float(np.max(valid_predictions)),
            'std': float(np.std(valid_predictions))
        }

        # Prepare results for display
        results_data = df_results.head(100).to_dict('records')

        return jsonify({
            'success': True,
            'message': 'Predictions completed successfully',
            'data': {
                'predictions': results_data,
                'statistics': stats,
                'total_rows': len(df_results),
                'successful_predictions': len(valid_predictions)
            }
        })

    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}\n{traceback.format_exc()}'}), 500


@app.route('/download')
def download_results():
    """Download results as CSV"""
    try:
        if 'results_file' not in session:
            return jsonify({'error': 'No results available'}), 400

        results_filepath = os.path.join(app.config['RESULTS_FOLDER'], session['results_file'])

        if not os.path.exists(results_filepath):
            return jsonify({'error': 'Results file not found'}), 404

        return send_file(
            results_filepath,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )

    except Exception as e:
        return jsonify({'error': f'Download error: {str(e)}'}), 500


@app.route('/model-info')
def model_info():
    """Get model information"""
    try:
        model_exists = os.path.exists(app.config['MODEL_PATH'])

        info = {
            'model_exists': model_exists,
            'model_path': app.config['MODEL_PATH']
        }

        if model_exists:
            model = load_model(app.config['MODEL_PATH'])
            info['model_type'] = type(model).__name__

            if hasattr(model, 'n_features_in_'):
                info['n_features'] = model.n_features_in_

        return jsonify(info)

    except Exception as e:
        return jsonify({'error': f'Error getting model info: {str(e)}'}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413


if __name__ == '__main__':
    print("="*60)
    print("Starting Housing Price Prediction Web Application")
    print("="*60)
    print(f"\nModel path: {app.config['MODEL_PATH']}")
    print(f"Model exists: {os.path.exists(app.config['MODEL_PATH'])}")
    print("\nStarting server on http://127.0.0.1:5000")
    print("="*60)

    app.run(debug=True, host='0.0.0.0', port=5000)

