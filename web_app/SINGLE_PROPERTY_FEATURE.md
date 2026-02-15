# Single Property Prediction Feature - Documentation

## 🎯 Overview

A new feature has been added to the Housing Price Prediction application that allows users to **dynamically input a single property's details** and get an instant price prediction, without needing to upload a CSV file.

## ✨ Features Added

### 1. **Dual Input Method Selection**
Users now see two options on the home page:
- **📁 Bulk Upload (CSV)**: Traditional batch prediction for multiple properties
- **🏠 Single Property**: New feature for instant single property prediction

### 2. **Comprehensive Property Input Form**

#### Required Fields:
- **Property Type** (Dropdown):
  - Appartement
  - Villa
  - Maison
  - Studio
  - Duplex
  - Riad

- **Area (m²)**: Numeric input for property size
- **Rooms**: Number of rooms

#### Optional Fields:
- **Title/Description**: Property title or description
- **Location**: City/district information

#### 13 Amenities/Features (Checkboxes):
- 🏊 Swimming Pool
- 🌳 Garden
- 🌅 Terrace
- 🚗 Garage
- 🛗 Elevator
- ❄️ Air Conditioning
- 🔥 Heating
- 🍳 Equipped Kitchen
- 🔒 Security
- 👮 Doorman
- 📦 Cellar
- 🪟 Double Glazing
- 🚪 Reinforced Door

### 3. **Enhanced Results Display**

For single property predictions, the results page shows:
- **Property Summary Card**: Beautiful gradient card with:
  - Property title
  - Location
  - Type, area, and rooms
  - **Large predicted price display**

- **Model Performance Metrics** (if available):
  - R² Score
  - MAPE
  - MAE
  - RMSE

- **Property Features Summary**: Visual display of selected amenities with emoji icons

- **Simplified View**: Charts and bulk statistics are hidden for single property mode

## 🔧 Technical Implementation

### Frontend Changes (`index.html`)

#### 1. **Redesigned Step 1**
```html
<!-- Choice between CSV and Single Property -->
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
    <!-- CSV Upload Card -->
    <!-- Single Property Card -->
</div>
```

#### 2. **Single Property Form**
- Responsive grid layout
- Styled inputs with border-radius and color scheme
- Checkbox grid for amenities (3 columns)
- Form validation for required fields

#### 3. **JavaScript Functions Added**

**`showUploadMethod(method)`**
- Controls which input method is displayed
- Parameters: 'csv', 'single', or 'choice'

**`resetSinglePropertyForm()`**
- Clears all form inputs

**`submitSingleProperty()`**
- Validates required fields
- Collects form data into JSON object
- Sends POST request to `/predict_single` endpoint
- Handles response and displays results

**`displaySinglePropertyResult(data, propertyData)`**
- Creates beautiful property info card
- Shows predicted price prominently
- Displays model metrics
- Shows selected amenities
- Hides unnecessary sections (charts, bulk tables)

### Backend Changes (`app.py`)

#### New Endpoint: `/predict_single`

**Method**: POST  
**Content-Type**: application/json

**Request Body**:
```json
{
    "title": "Villa luxueuse avec piscine",
    "location": "Tunis, La Marsa",
    "property_type": "Villa",
    "area_m2": 250,
    "rooms": 5,
    "swimming_pool": 1,
    "garden": 1,
    "terrace": 1,
    "garage": 1,
    "elevator": 0,
    "air_conditioning": 1,
    "heating": 1,
    "equipped_kitchen": 1,
    "security": 1,
    "doorman": 0,
    "cellar": 0,
    "double_glazing": 1,
    "reinforced_door": 1
}
```

**Response**:
```json
{
    "success": true,
    "message": "Prediction completed using XGBoost (Grid Search)",
    "data": {
        "predicted_price": 2450000.50,
        "model_used": "XGBoost (Grid Search)",
        "model_key": "xgboost_grid",
        "model_metrics": {
            "r2": "0.92",
            "mape": "12.5",
            "mae": "150000",
            "rmse": "200000"
        }
    }
}
```

**Processing Steps**:
1. Receives JSON property data
2. Converts to pandas DataFrame
3. Applies same preprocessing pipeline as CSV upload
4. Loads default model (XGBoost)
5. Applies scaling if scaler exists
6. Aligns features with training data
7. Makes prediction
8. Returns predicted price with model metrics

## 📊 User Flow

### Single Property Mode:
```
User lands on homepage
    ↓
Clicks "Single Property" option
    ↓
Fills in property details form
    ↓
Clicks "🎯 Predict Price" button
    ↓
Form validation
    ↓
Data sent to backend via /predict_single
    ↓
Backend preprocesses single property
    ↓
Model makes prediction
    ↓
Results displayed with:
    - Property summary card
    - Predicted price (large display)
    - Model metrics
    - Feature summary
    ↓
User can:
    - Start new prediction (reload)
    - Go back to input form
```

### CSV Mode (Original):
```
User lands on homepage
    ↓
Clicks "Bulk Upload (CSV)" option
    ↓
Uploads CSV file
    ↓
Reviews data preview
    ↓
Proceeds to preprocessing
    ↓
Selects model
    ↓
Views comprehensive results with charts
```

## 🎨 Design Features

### Color Scheme:
- **Primary Gradient**: `#667eea` → `#764ba2` (Blue-purple)
- **Background**: White with subtle shadows
- **Borders**: `#e8ebff` (Light blue)
- **Hover States**: Enhanced borders and shadows

### Layout:
- **Responsive Grid**: Adapts to mobile, tablet, desktop
- **Card-Based Design**: Clean, modern appearance
- **Checkbox Grid**: 3 columns for amenities
- **Form Inputs**: Consistent styling with 10px padding and 8px border-radius

### Typography:
- **Headings**: Bold, color-coded
- **Labels**: Font-weight 600
- **Price Display**: 2.5em, bold, prominent

## 🚀 Usage Examples

### Example 1: Luxury Villa
```
Title: Villa moderne avec vue sur mer
Location: Tunis, La Marsa
Type: Villa
Area: 350 m²
Rooms: 6
Features: Pool, Garden, Terrace, Garage, AC, Heating, Kitchen, Security
→ Predicted Price: 4,250,000 DT
```

### Example 2: City Apartment
```
Title: Appartement moderne
Location: Tunis, Centre Ville
Type: Appartement
Area: 85 m²
Rooms: 2
Features: Elevator, AC, Equipped Kitchen, Security
→ Predicted Price: 950,000 DT
```

### Example 3: Budget Studio
```
Title: Studio étudiant
Location: Sousse, Sahloul
Type: Studio
Area: 35 m²
Rooms: 1
Features: Heating
→ Predicted Price: 380,000 DT
```

## ✅ Benefits

1. **Instant Predictions**: No need to create CSV files for single properties
2. **User-Friendly**: Intuitive form with clear labels
3. **Visual Feedback**: Beautiful result cards with gradient backgrounds
4. **Complete Information**: All property details captured
5. **Professional Display**: Suitable for client presentations
6. **Flexible**: Users can choose between bulk and single modes
7. **Same Accuracy**: Uses same preprocessing and model as batch predictions

## 🔄 Future Enhancements

Potential improvements:
- Add property image upload
- Show comparable properties
- Display price confidence intervals
- Add map visualization for location
- Show price history if property ID is provided
- Add "Save Property" feature
- Export single property report as PDF
- Add currency conversion options
- Include price per m² calculation
- Show investment ROI calculator

## 📝 Validation Rules

- **Property Type**: Required, must be selected from dropdown
- **Area**: Required, must be numeric, minimum 1 m²
- **Rooms**: Required, must be numeric, minimum 1
- **Title**: Optional, defaults to "Property"
- **Location**: Optional, defaults to "Unknown"
- **Amenities**: All optional, default to 0 (not present)

## 🛡️ Error Handling

The system handles:
- Missing required fields (shows error alert)
- Invalid numeric inputs (HTML5 validation)
- Preprocessing errors (shows error message)
- Model not trained (shows error with suggestion)
- Network errors (shows error alert)
- Empty results after preprocessing (shows warning)

## 📱 Responsive Design

- **Desktop**: Two-column layout, side-by-side cards
- **Tablet**: Responsive grid adjusts to screen size
- **Mobile**: Stacks vertically, maintains usability

---

**Created**: February 15, 2026  
**Version**: 3.0  
**Files Modified**:
- `templates/index.html` (Frontend)
- `app.py` (Backend endpoint)

**Impact**: Users can now predict prices for single properties without CSV files! 🏠🎯

