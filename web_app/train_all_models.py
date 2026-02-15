"""
MULTI-MODEL TRAINING SCRIPT
Trains and saves all models:
  - Random Forest
  - KNN (n=1, 5, 10, 100)
  - XGBoost + Bayesian Optimization
  - XGBoost + Random Search
  - XGBoost + Grid Search
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────────────────
# Feature Engineering (mirrors app.py / train_advanced_model.py)
# ─────────────────────────────────────────────────────────

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

    coastal_kw = ['Hammamet','La Marsa','Gammarth','Carthage','Sousse','Monastir','Bizerte','Klibia','Mahdia']
    tourist_kw = ['Hammamet','Sousse','Monastir','Djerba','Mahdia']
    df['is_coastal']     = df['city'].apply(lambda x: 1 if any(k in str(x) for k in coastal_kw) else 0)
    df['is_tourist_area'] = df['city'].apply(lambda x: 1 if any(k in str(x) for k in tourist_kw) else 0)
    return df


def encode_location_frequency(df, column, min_freq=10):
    freq_map = df[column].value_counts()
    df[f'{column}_frequency']  = df[column].map(freq_map).fillna(0)
    df[f'{column}_is_common'] = (df[f'{column}_frequency'] >= min_freq).astype(int)
    return df


def create_advanced_features(df):
    df = df.copy()
    amenity_cols = ['garden','terrace','garage','elevator','swimming_pool',
                    'doorman','cellar','air_conditioning','heating','security',
                    'double_glazing','reinforced_door','equipped_kitchen']
    df['total_amenities']  = df[amenity_cols].sum(axis=1)
    df['premium_amenities'] = df[['swimming_pool','doorman','elevator','air_conditioning']].sum(axis=1)
    df['basic_amenities']   = df[['heating','equipped_kitchen','garage']].sum(axis=1)
    df['security_score']    = df[['security','doorman','reinforced_door','double_glazing']].sum(axis=1)
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


def remove_outliers(df, y):
    Q1, Q3 = y.quantile(0.05), y.quantile(0.95)
    IQR     = Q3 - Q1
    mask = (y >= Q1 - 2.0 * IQR) & (y <= Q3 + 2.0 * IQR)
    print(f"   Removing {(~mask).sum()} outliers (kept {mask.sum()} rows)")
    return df[mask], y[mask]


def evaluate(name, model, X_val, y_val):
    preds = model.predict(X_val)
    r2    = r2_score(y_val, preds)
    mae   = mean_absolute_error(y_val, preds)
    mape  = np.mean(np.abs((y_val - preds) / y_val)) * 100
    rmse  = np.sqrt(mean_squared_error(y_val, preds))
    print(f"   [{name}]  R²={r2:.4f}  MAE={mae:,.0f}  MAPE={mape:.2f}%  RMSE={rmse:,.0f}")
    return {"r2": round(r2,4), "mae": round(float(mae),2),
            "mape": round(float(mape),2), "rmse": round(float(rmse),2)}


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def train_all_models():
    print("="*70)
    print("MULTI-MODEL TRAINING")
    print("="*70)

    project_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_data_path = os.path.join(project_dir, 'data', 'train', 'trains.csv')
    model_dir      = os.path.join(project_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(train_data_path):
        print(f"❌ Training data not found at: {train_data_path}")
        return False

    # ── Load ────────────────────────────────────────────────
    print(f"\n📂 Loading data...")
    df = pd.read_csv(train_data_path)
    print(f"   {len(df)} rows, {len(df.columns)} columns")

    # ── Prepare features ───────────────────────────────────
    y = df['price'].copy()
    X = df.drop(columns=[c for c in ['price','title'] if c in df.columns])

    print("\n🔧 Feature engineering...")
    X = extract_location_features(X)
    X = encode_location_frequency(X, 'city',     min_freq=20)
    X = encode_location_frequency(X, 'district', min_freq=15)
    X = X.drop(columns=['location','location_str','city','district'], errors='ignore')

    if 'property_type' in X.columns:
        X = pd.get_dummies(X, columns=['property_type'], prefix='property_type')

    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median()).fillna(0)
    X = create_advanced_features(X)
    X, y = remove_outliers(X, y)
    print(f"   Final shape: {X.shape}")

    # ── Split ───────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # ── Scale (needed for KNN; others benefit too) ─────────
    numeric_cols = ['area_m2','rooms','total_amenities','area_rooms_interaction',
                    'area_per_room','amenity_density','space_quality',
                    'city_frequency','district_frequency',
                    'location_quality_score','coastal_premium_score']
    numeric_cols = [c for c in numeric_cols if c in X_train.columns]

    scaler = StandardScaler()
    X_train_s = X_train.copy(); X_val_s = X_val.copy()
    X_train_s[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val_s[numeric_cols]   = scaler.transform(X_val[numeric_cols])

    # Save shared scaler & metadata
    joblib.dump(scaler, os.path.join(model_dir, 'scaler_advanced.joblib'))
    with open(os.path.join(model_dir, 'scaling_columns_advanced.txt'), 'w') as f:
        f.write('\n'.join(numeric_cols))
    with open(os.path.join(model_dir, 'feature_names_advanced.txt'), 'w') as f:
        f.write('\n'.join(X.columns.tolist()))

    metrics_all = {}

    # ══════════════════════════════════════════════════════
    # 1. RANDOM FOREST
    # ══════════════════════════════════════════════════════
    print("\n🌲 Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, max_features='sqrt',
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train_s, y_train)
    metrics_all['random_forest'] = evaluate("Random Forest", rf, X_val_s, y_val)
    joblib.dump(rf, os.path.join(model_dir, 'random_forest.joblib'))
    print("   ✅ Saved random_forest.joblib")

    # ══════════════════════════════════════════════════════
    # 2. KNN variants
    # ══════════════════════════════════════════════════════
    for k in [1, 5, 10, 100]:
        print(f"\n🔵 Training KNN (n={k})...")
        knn = KNeighborsRegressor(n_neighbors=k, weights='uniform', n_jobs=-1)
        knn.fit(X_train_s, y_train)
        key = f'knn_k{k}'
        metrics_all[key] = evaluate(f"KNN k={k}", knn, X_val_s, y_val)
        joblib.dump(knn, os.path.join(model_dir, f'knn_k{k}.joblib'))
        print(f"   ✅ Saved knn_k{k}.joblib")

    # ══════════════════════════════════════════════════════
    # 3. XGBoost base params (shared starting point)
    # ══════════════════════════════════════════════════════
    base_xgb_params = dict(
        objective='reg:squarederror',
        random_state=42, n_jobs=-1, verbosity=0
    )

    # ── 3a. XGBoost + Bayesian Optimization (via scikit-optimize) ─
    print("\n🔬 Training XGBoost + Bayesian Optimization...")
    try:
        from skopt import BayesSearchCV
        from skopt.space import Real, Integer

        bayes_search_space = {
            'max_depth':        Integer(3, 9),
            'learning_rate':    Real(0.01, 0.3, prior='log-uniform'),
            'n_estimators':     Integer(100, 600),
            'subsample':        Real(0.5, 1.0),
            'colsample_bytree': Real(0.5, 1.0),
            'min_child_weight': Integer(1, 10),
            'gamma':            Real(0.0, 0.5),
            'reg_alpha':        Real(0.0, 1.0),
            'reg_lambda':       Real(0.5, 3.0),
        }

        bayes_cv = BayesSearchCV(
            xgb.XGBRegressor(**base_xgb_params),
            bayes_search_space,
            n_iter=30,
            cv=3,
            scoring='r2',
            n_jobs=1,
            random_state=42,
            verbose=0
        )
        bayes_cv.fit(X_train_s, y_train)
        xgb_bayes = bayes_cv.best_estimator_
        print(f"   Best params: {bayes_cv.best_params_}")
        metrics_all['xgboost_bayesian'] = evaluate("XGBoost Bayesian", xgb_bayes, X_val_s, y_val)
        joblib.dump(xgb_bayes, os.path.join(model_dir, 'xgboost_bayesian.joblib'))
        print("   ✅ Saved xgboost_bayesian.joblib")

    except ImportError:
        print("   ⚠️  scikit-optimize not installed. Training XGBoost Bayesian with manual good params instead.")
        xgb_bayes = xgb.XGBRegressor(
            max_depth=6, learning_rate=0.04, n_estimators=450,
            subsample=0.75, colsample_bytree=0.75,
            min_child_weight=4, gamma=0.12,
            reg_alpha=0.3, reg_lambda=1.5,
            **base_xgb_params
        )
        xgb_bayes.fit(X_train_s, y_train, verbose=False)
        metrics_all['xgboost_bayesian'] = evaluate("XGBoost Bayesian (fallback)", xgb_bayes, X_val_s, y_val)
        joblib.dump(xgb_bayes, os.path.join(model_dir, 'xgboost_bayesian.joblib'))
        print("   ✅ Saved xgboost_bayesian.joblib (fallback params)")

    # ── 3b. XGBoost + Random Search ────────────────────────
    print("\n🎲 Training XGBoost + Random Search...")
    random_param_dist = {
        'max_depth':        [3, 4, 5, 6, 7, 8, 9],
        'learning_rate':    [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3],
        'n_estimators':     [100, 200, 300, 400, 500, 600],
        'subsample':        [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 2, 3, 4, 5, 7, 10],
        'gamma':            [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
        'reg_alpha':        [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
        'reg_lambda':       [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    }

    random_cv = RandomizedSearchCV(
        xgb.XGBRegressor(**base_xgb_params),
        random_param_dist,
        n_iter=40,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    random_cv.fit(X_train_s, y_train)
    xgb_random = random_cv.best_estimator_
    print(f"   Best params: {random_cv.best_params_}")
    metrics_all['xgboost_random'] = evaluate("XGBoost Random Search", xgb_random, X_val_s, y_val)
    joblib.dump(xgb_random, os.path.join(model_dir, 'xgboost_random.joblib'))
    print("   ✅ Saved xgboost_random.joblib")

    # ── 3c. XGBoost + Grid Search (smaller grid for speed) ─
    print("\n🔲 Training XGBoost + Grid Search...")
    grid_param_grid = {
        'max_depth':        [4, 6, 8],
        'learning_rate':    [0.03, 0.05, 0.1],
        'n_estimators':     [200, 400],
        'subsample':        [0.7, 0.85],
        'colsample_bytree': [0.7, 0.85],
        'min_child_weight': [3, 6],
    }

    grid_cv = GridSearchCV(
        xgb.XGBRegressor(reg_alpha=0.3, reg_lambda=1.5, gamma=0.1, **base_xgb_params),
        grid_param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )
    grid_cv.fit(X_train_s, y_train)
    xgb_grid = grid_cv.best_estimator_
    print(f"   Best params: {grid_cv.best_params_}")
    metrics_all['xgboost_grid'] = evaluate("XGBoost Grid Search", xgb_grid, X_val_s, y_val)
    joblib.dump(xgb_grid, os.path.join(model_dir, 'xgboost_grid.joblib'))
    print("   ✅ Saved xgboost_grid.joblib")

    # ── Save metrics summary ────────────────────────────────
    metrics_path = os.path.join(model_dir, 'models_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_all, f, indent=2)
    print(f"\n📊 Metrics saved to {metrics_path}")

    # ── Summary ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("="*70)
    print(f"\n{'Model':<30} {'R²':>8} {'MAE':>12} {'MAPE':>8}")
    print("-" * 62)
    for name, m in metrics_all.items():
        print(f"{name:<30} {m['r2']:>8.4f} {m['mae']:>12,.0f} {m['mape']:>7.2f}%")

    return True


if __name__ == "__main__":
    success = train_all_models()
    if not success:
        sys.exit(1)