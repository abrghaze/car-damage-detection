@echo off
echo ============================================
echo    Car Damage Detection - Application Start
echo ============================================
echo.

:: Check if Python virtual environment exists
if not exist "venv" (
    echo [INFO] Creating Python virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate

:: Install Python dependencies if needed
echo [INFO] Checking Python dependencies...
pip install -r requirements.txt -q

:: Start Backend in new terminal
echo [INFO] Starting Backend API server...
start cmd /k "cd /d %~dp0 && call venv\Scripts\activate && cd backend && python app.py"

:: Wait for backend to initialize
timeout /t 3 /nobreak > nul

:: Check if node_modules exists
if not exist "frontend\node_modules" (
    echo [INFO] Installing frontend dependencies...
    cd frontend
    npm install
    cd ..
)

:: Start Frontend
echo [INFO] Starting Frontend...
start cmd /k "cd /d %~dp0\frontend && npm run dev"

echo.
echo ============================================
echo    Application Started Successfully!
echo ============================================
echo.
echo    Backend API: http://localhost:8000
echo    API Docs:    http://localhost:8000/docs
echo    Frontend:    http://localhost:3000
echo.
echo    Press any key to open the application...
pause > nul
start http://localhost:3000
