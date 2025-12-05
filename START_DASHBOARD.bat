@echo off
REM ============================================================================
REM BRAIN TUMOR CLASSIFICATION - PROFESSIONAL ENHANCED DASHBOARD
REM ============================================================================
REM Runs in ISOLATED virtual environment (dashboard_venv)
REM ZERO effects on main training environment
REM Professional design with medical imagery and enhanced UI

cd /d "C:\Users\parne\OneDrive\Documents\265 Final project 2\archive (2)"

echo.
echo ============================================================================
echo BRAIN TUMOR CLASSIFICATION - PROFESSIONAL DASHBOARD
echo Enhanced Medical AI Interface
echo ============================================================================
echo.
echo ISOLATION: Using separate venv (dashboard_venv)
echo STATUS: Starting enhanced dashboard...
echo.
echo FEATURES:
echo   - Professional medical design
echo   - Brain and tumor imagery
echo   - Upload MRI images
echo   - Real-time predictions
echo   - Model comparison
echo   - Comprehensive analytics
echo   - Formal UI/UX
echo.
echo URL: http://localhost:8501
echo Press Ctrl+C to stop
echo.
echo ============================================================================
echo.

REM Set environment variables
set TF_CPP_MIN_LOG_LEVEL=3

REM Use separate dashboard venv (NOT main .venv)
call .\dashboard_venv\Scripts\activate.bat

REM Open browser
timeout /t 2
start http://localhost:8501

REM Run clean dashboard from isolated environment
streamlit run dashboard_app/app_clean.py --logger.level=error --server.port=8501 --client.showErrorDetails=false

pause
echo.
echo Dashboard closed.
echo.

streamlit run dashboard.py

pause
