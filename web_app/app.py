"""
Flask Web Application for Housing Price Prediction
Supports: Random Forest, KNN (k=1/5/10/100),
          XGBoost (Bayesian / Random / Grid Search)
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cleaning.pipeline import (
    drop_unnecessary_columns, remove_missing_price, update_features_from_title,
    remove_residence_rows, fill_property_type_from_title, remove_missing_property_type,
    remove_all_missing_key_features, impute_rooms_and_area, final_imputation
)
from inferance.inferance import load_model, predict

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER']  = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR   = os.path.join(PROJECT_DIR, 'model')

MODEL_REGISTRY = {
    'random_forest':    {'file': 'random_forest.joblib',     'label': 'Random Forest'},
    'knn_k1':           {'file': 'knn_k1.joblib',            'label': 'KNN (k=1)'},
    'knn_k5':           {'file': 'knn_k5.joblib',            'label': 'KNN (k=5)'},
    'knn_k10':          {'file': 'knn_k10.joblib',           'label': 'KNN (k=10)'},
    'knn_k100':         {'file': 'knn_k100.joblib',          'label': 'KNN (k=100)'},
    'xgboost_bayesian': {'file': 'xgboost_bayesian.joblib',  'label': 'XGBoost (Bayesian Opt.)'},
    'xgboost_random':   {'file': 'xgboost_random.joblib',    'label': 'XGBoost (Random Search)'},
    'xgboost_grid':     {'file': 'xgboost_grid.joblib',      'label': 'XGBoost (Grid Search)'},
    'xgboost_advanced': {'file': 'xgboost_advanced.joblib',  'label': 'XGBoost (Advanced Ensemble)'},
}

DEFAULT_MODEL = 'xgboost_advanced'

os.makedirs(app.config['UPLOAD_FOLDER'],  exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_location_features(df):
    df = df.copy()
    df['location_str'] = df['location'].fillna('Unknown, Unknown')
    location_parts = df['location_str'].str.split(',', expand=True)
    df['city']     = location_parts[0].str.strip() if location_parts.shape[1] > 0 else 'Unknown'
    df['district'] = location_parts[1].str.strip() if location_parts.shape[1] > 1 else 'Unknown'
    premium_cities = ['Gammarth','La Marsa','Carthage','Sidi Bou Said',
                      'Les Jardins de Carthage','Ain Zaghouan Nord','La Soukra']
    mid_cities = ['Ariana','Ennasr','Menzah','Manar','Lac','Berges du Lac',
                  'Sousse','Hammamet','Monastir','Sfax']
    df['location_tier'] = 2
    df.loc[df['city'].isin(premium_cities), 'location_tier'] = 3
    df.loc[df['city'].isin(mid_cities),     'location_tier'] = 2.5
    coastal_kw = ['Hammamet','La Marsa','Gammarth','Carthage','Sousse',
                  'Monastir','Bizerte','Klibia','Mahdia']
    tourist_kw = ['Hammamet','Sousse','Monastir','Djerba','Mahdia']
    df['is_coastal']      = df['city'].apply(lambda x: 1 if any(k in str(x) for k in coastal_kw) else 0)
    df['is_tourist_area'] = df['city'].apply(lambda x: 1 if any(k in str(x) for k in tourist_kw) else 0)
    return df


def encode_location_frequency(df, column, frequencies):
    df[f'{column}_frequency'] = df[column].map(frequencies).fillna(0)
    df[f'{column}_is_common'] = (df[f'{column}_frequency'] >= 10).astype(int)
    return df


def create_advanced_features(df):
    df = df.copy()
    amenity_cols = ['garden','terrace','garage','elevator','swimming_pool',
                    'doorman','cellar','air_conditioning','heating','security',
                    'double_glazing','reinforced_door','equipped_kitchen']
    df['total_amenities']    = df[amenity_cols].sum(axis=1)
    df['premium_amenities']  = df[['swimming_pool','doorman','elevator','air_conditioning']].sum(axis=1)
    df['basic_amenities']    = df[['heating','equipped_kitchen','garage']].sum(axis=1)
    df['security_score']     = df[['security','doorman','reinforced_door','double_glazing']].sum(axis=1)
    df['has_luxury_features'] = (
        (df['swimming_pool'] == 1) | (df['doorman'] == 1) | (df['total_amenities'] >= 7)
    ).astype(int)
    df['area_category'] = pd.cut(df['area_m2'],
                                  bins=[0,50,100,150,200,300,500,10000],
                                  labels=[0,1,2,3,4,5,6]).astype(float).fillna(0)
    df['room_category']  = pd.cut(df['rooms'],
                                   bins=[-1,1,2,3,4,6,100],
                                   labels=[0,1,2,3,4,5]).astype(float).fillna(0)
    df['area_rooms_interaction'] = df['area_m2'] * df['rooms']
    df['area_per_room']          = df['area_m2'] / (df['rooms'] + 1)
    df['amenity_density']        = df['total_amenities'] / (df['area_m2'] / 100 + 1)
    df['space_quality']          = df['area_per_room'] * (1 + df['total_amenities'] / 10)
    df['location_quality_score'] = df['location_tier'] * (1 + df['total_amenities'] / 10)
    df['coastal_premium_score']  = df['is_coastal'] * df['has_luxury_features'] * df['area_category']
    return df


def preprocess_for_prediction(df):
    try:
        original_df = df.copy()
        if 'price' in df.columns:
            df = df.drop(columns=['price'])
        for c in ['scraped_at','url','bathrooms']:
            if c in df.columns:
                df = df.drop(columns=[c])
        df = update_features_from_title(df)
        df = remove_residence_rows(df)
        df = fill_property_type_from_title(df)
        initial_len = len(df)
        df = remove_missing_property_type(df)
        removed_rows = initial_len - len(df)
        df = remove_all_missing_key_features(df)
        df = impute_rooms_and_area(df)
        df = final_imputation(df)
        for col in ['area_m2','rooms','swimming_pool','garden','terrace','garage',
                    'elevator','air_conditioning','heating','equipped_kitchen',
                    'security','doorman','cellar','double_glazing','reinforced_door']:
            if col not in df.columns:
                df[col] = 0
        if 'location' in df.columns:
            df = extract_location_features(df)
            df = encode_location_frequency(df, 'city',     {'Unknown': 0})
            df = encode_location_frequency(df, 'district', {'Unknown': 0})
            df = df.drop(columns=['location','location_str','city','district'], errors='ignore')
        if 'property_type' in df.columns:
            df = pd.get_dummies(df, columns=['property_type'], prefix='property_type')
        df = df.drop(columns=['title'], errors='ignore')
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.fillna(df.median()).fillna(0)
        df = create_advanced_features(df)
        return df, original_df, removed_rows, None
    except Exception as e:
        return None, None, 0, str(e)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/models', methods=['GET'])
def get_models():
    available = []
    for key, info in MODEL_REGISTRY.items():
        path = os.path.join(MODEL_DIR, info['file'])
        available.append({'key': key, 'label': info['label'], 'trained': os.path.exists(path)})
    return jsonify({'models': available})


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only CSV files are allowed'}), 400
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        try:
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding='latin-1')
        except Exception as e:
            os.remove(filepath)
            return jsonify({'error': f'Invalid CSV file: {str(e)}'}), 400
        if len(df) == 0:
            os.remove(filepath)
            return jsonify({'error': 'CSV file is empty'}), 400
        session['uploaded_file'] = unique_filename
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'data': {
                'rows': len(df), 'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'preview': df.head(10).to_dict('records'),
                'filename': filename
            }
        })
    except Exception as e:
        return jsonify({'error': f'Upload error: {str(e)}'}), 500


@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        if 'uploaded_file' not in session:
            return jsonify({'error': 'No file uploaded'}), 400
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['uploaded_file'])
        if not os.path.exists(filepath):
            return jsonify({'error': 'Uploaded file not found'}), 404
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding='latin-1')
        processed_df, original_df, removed_rows, error = preprocess_for_prediction(df)
        if error:
            return jsonify({'error': f'Preprocessing error: {error}'}), 500
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
    try:
        if 'processed_file' not in session:
            return jsonify({'error': 'No preprocessed data available'}), 400
        processed_filepath = os.path.join(app.config['RESULTS_FOLDER'], session['processed_file'])
        if not os.path.exists(processed_filepath):
            return jsonify({'error': 'Preprocessed file not found'}), 404

        body = request.get_json(silent=True) or {}
        model_key = body.get('model', DEFAULT_MODEL)
        if model_key not in MODEL_REGISTRY:
            return jsonify({'error': f'Unknown model key: {model_key}'}), 400

        model_info = MODEL_REGISTRY[model_key]
        model_path = os.path.join(MODEL_DIR, model_info['file'])
        if not os.path.exists(model_path):
            return jsonify({
                'error': f'Model "{model_info["label"]}" has not been trained yet.',
                'suggestion': 'Run train_all_models.py to train all models.'
            }), 404

        try:
            df_processed = pd.read_csv(processed_filepath, encoding='utf-8')
        except UnicodeDecodeError:
            df_processed = pd.read_csv(processed_filepath, encoding='latin-1')
        model = joblib.load(model_path)

        scaler_path       = os.path.join(MODEL_DIR, 'scaler_advanced.joblib')
        scaling_cols_path = os.path.join(MODEL_DIR, 'scaling_columns_advanced.txt')
        if not os.path.exists(scaler_path):
            scaler_path       = os.path.join(MODEL_DIR, 'scaler.joblib')
            scaling_cols_path = os.path.join(MODEL_DIR, 'scaling_columns.txt')

        if os.path.exists(scaler_path) and os.path.exists(scaling_cols_path):
            scaler = joblib.load(scaler_path)
            try:
                with open(scaling_cols_path, 'r', encoding='utf-8') as f:
                    scaling_cols = [l.strip() for l in f.readlines()]
            except UnicodeDecodeError:
                with open(scaling_cols_path, 'r', encoding='latin-1') as f:
                    scaling_cols = [l.strip() for l in f.readlines()]
            df_scaled = df_processed.copy()
            existing_cols = [c for c in scaling_cols if c in df_scaled.columns]
            if existing_cols:
                df_scaled[existing_cols] = scaler.transform(df_scaled[existing_cols])
            df_processed = df_scaled

        feature_names_path = os.path.join(MODEL_DIR, 'feature_names_advanced.txt')
        if not os.path.exists(feature_names_path):
            feature_names_path = os.path.join(MODEL_DIR, 'feature_names.txt')
        if os.path.exists(feature_names_path):
            try:
                with open(feature_names_path, 'r', encoding='utf-8') as f:
                    training_features = [l.strip() for l in f.readlines()]
            except UnicodeDecodeError:
                with open(feature_names_path, 'r', encoding='latin-1') as f:
                    training_features = [l.strip() for l in f.readlines()]
            for col in training_features:
                if col not in df_processed.columns:
                    df_processed[col] = 0
            df_processed = df_processed[training_features]

        if isinstance(model, dict) and 'xgb' in model and 'rf' in model:
            w = model.get('weights', (0.6, 0.4))
            predictions = w[0]*model['xgb'].predict(df_processed) + w[1]*model['rf'].predict(df_processed)
        else:
            predictions = model.predict(df_processed)

        original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['uploaded_file'])
        try:
            df_original = pd.read_csv(original_filepath, encoding='utf-8')
        except UnicodeDecodeError:
            df_original = pd.read_csv(original_filepath, encoding='latin-1')
        df_results  = df_original.copy()
        if len(predictions) < len(df_original):
            df_results['predicted_price'] = np.nan
            df_results.loc[:len(predictions)-1, 'predicted_price'] = predictions
        else:
            df_results['predicted_price'] = predictions

        results_filename = f"results_{model_key}_{session['uploaded_file']}"
        results_filepath = os.path.join(app.config['RESULTS_FOLDER'], results_filename)
        df_results.to_csv(results_filepath, index=False)
        session['results_file'] = results_filename

        valid_preds = predictions[~np.isnan(predictions)]

        # Basic statistics
        stats = {
            'count':  len(valid_preds),
            'mean':   float(np.mean(valid_preds)),
            'median': float(np.median(valid_preds)),
            'min':    float(np.min(valid_preds)),
            'max':    float(np.max(valid_preds)),
            'std':    float(np.std(valid_preds)),
            'q25':    float(np.percentile(valid_preds, 25)),
            'q75':    float(np.percentile(valid_preds, 75)),
            'range':  float(np.max(valid_preds) - np.min(valid_preds)),
            'cv':     float(np.std(valid_preds) / np.mean(valid_preds) * 100) if np.mean(valid_preds) > 0 else 0
        }

        # Price distribution for histogram (bins)
        hist, bin_edges = np.histogram(valid_preds, bins=20)
        histogram_data = {
            'counts': hist.tolist(),
            'bins': bin_edges.tolist()
        }

        # Price ranges distribution
        price_ranges = {
            'below_500k': int(np.sum(valid_preds < 500000)),
            '500k_1m': int(np.sum((valid_preds >= 500000) & (valid_preds < 1000000))),
            '1m_2m': int(np.sum((valid_preds >= 1000000) & (valid_preds < 2000000))),
            '2m_5m': int(np.sum((valid_preds >= 2000000) & (valid_preds < 5000000))),
            'above_5m': int(np.sum(valid_preds >= 5000000))
        }

        # Top predictions
        top_indices = np.argsort(predictions)[-10:][::-1]
        top_predictions = df_results.iloc[top_indices][['predicted_price']].to_dict('records')

        # Add index/row info to identify properties
        for i, idx in enumerate(top_indices):
            top_predictions[i]['index'] = int(idx)
            if 'title' in df_original.columns:
                top_predictions[i]['title'] = str(df_original.iloc[idx]['title'])[:50]
            if 'location' in df_original.columns:
                top_predictions[i]['location'] = str(df_original.iloc[idx]['location'])

        metrics_path = os.path.join(MODEL_DIR, 'models_metrics.json')
        model_metrics = None
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                all_m = json.load(f)
            model_metrics = all_m.get(model_key)

        return jsonify({
            'success': True,
            'message': f'Predictions completed using {model_info["label"]}',
            'data': {
                'predictions':            df_results.head(100).to_dict('records'),
                'all_predictions':        valid_preds.tolist(),
                'statistics':             stats,
                'histogram':              histogram_data,
                'price_ranges':           price_ranges,
                'top_predictions':        top_predictions,
                'total_rows':             len(df_results),
                'successful_predictions': len(valid_preds),
                'model_used':             model_info['label'],
                'model_key':              model_key,
                'model_metrics':          model_metrics
            }
        })
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}\n{traceback.format_exc()}'}), 500


@app.route('/download')
def download_results():
    try:
        if 'results_file' not in session:
            return jsonify({'error': 'No results available'}), 400
        results_filepath = os.path.join(app.config['RESULTS_FOLDER'], session['results_file'])
        if not os.path.exists(results_filepath):
            return jsonify({'error': 'Results file not found'}), 404
        return send_file(
            results_filepath, mimetype='text/csv', as_attachment=True,
            download_name=f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    except Exception as e:
        return jsonify({'error': f'Download error: {str(e)}'}), 500


@app.route('/model-info')
def model_info_route():
    try:
        model_key  = request.args.get('model', DEFAULT_MODEL)
        minfo      = MODEL_REGISTRY.get(model_key, {})
        model_path = os.path.join(MODEL_DIR, minfo.get('file', ''))
        model_exists = os.path.exists(model_path)
        info = {'model_key': model_key, 'label': minfo.get('label','Unknown'),
                'model_exists': model_exists, 'model_path': model_path}
        if model_exists:
            m = joblib.load(model_path)
            info['model_type'] = 'Ensemble (XGBoost + RF)' if isinstance(m, dict) else type(m).__name__
            if hasattr(m, 'n_features_in_'):
                info['n_features'] = m.n_features_in_
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': f'Error getting model info: {str(e)}'}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413


if __name__ == '__main__':
    print("="*60)
    print("Housing Price Prediction – Multi-Model Edition")
    print("="*60)
    for key, info in MODEL_REGISTRY.items():
        path   = os.path.join(MODEL_DIR, info['file'])
        status = "✅" if os.path.exists(path) else "❌ (not trained)"
        print(f"  {status}  {key}: {info['label']}")
    print("\nStarting server on http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)