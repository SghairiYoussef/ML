@echo off
echo ============================================================
echo Housing Price Prediction Web Application
echo ============================================================
echo.

REM Set Python path
set PYTHON_PATH=C:\Users\Youssef\AppData\Local\Programs\Python\Python312\python.exe

REM Check if Python is installed
%PYTHON_PATH% --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found at %PYTHON_PATH%
    echo Trying system Python...
    set PYTHON_PATH=python
    python --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python is not installed or not in PATH
        echo Please install Python 3.8 or higher from python.org
        pause
        exit /b 1
    )
)

echo Using Python: %PYTHON_PATH%
echo.

echo [1/4] Checking dependencies...
%PYTHON_PATH% -m pip show flask >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    cd ..
    %PYTHON_PATH% -m pip install -r requierements.txt
    cd web_app
)

echo.
echo [2/4] Checking for trained models...

REM Check for basic model
if not exist "..\model\xgboost.joblib" (
    echo Basic model not found. Training basic model...
    %PYTHON_PATH% train_model.py
    if errorlevel 1 (
        echo ERROR: Basic model training failed
        pause
        exit /b 1
    )
) else (
    echo Basic model found!
)

REM Check for advanced ensemble model (required as default)
if not exist "..\model\xgboost_advanced.joblib" (
    echo.
    echo Advanced ensemble model not found. Training advanced model...
    echo This may take 5-10 minutes. Please wait...
    echo.
    %PYTHON_PATH% train_advanced_model.py
    if errorlevel 1 (
        echo WARNING: Advanced model training failed
        echo The application will use a fallback model
        echo.
    ) else (
        echo Advanced model trained successfully!
        echo.
        echo IMPORTANT: Please update model\models_metrics.json with the metrics shown above
        echo.
    )
) else (
    echo Advanced ensemble model found!
)

echo.
echo [3/4] Creating necessary directories...
if not exist "uploads" mkdir uploads
if not exist "results" mkdir results
if not exist "..\model" mkdir ..\model

echo.
echo [4/4] Starting web application...
echo.
echo ============================================================
echo Application starting at http://localhost:5000
echo Press Ctrl+C to stop the server
echo ============================================================
echo.

%PYTHON_PATH% app.py

pause


