@echo off
REM Crawl2Answer - Start Script for Windows
REM This script sets up the environment and starts the API server

echo === Crawl2Answer Startup Script ===
echo Crawl. Retrieve. Answer.
echo.

REM Configuration
set VENV_NAME=crawl2answer_env
set API_HOST=0.0.0.0
set API_PORT=8000

REM Check if Python is installed
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version') do set PYTHON_VER=%%i
echo Python %PYTHON_VER% found

REM Create virtual environment if it doesn't exist
if not exist "%VENV_NAME%" (
    echo Creating virtual environment...
    python -m venv %VENV_NAME%
    if errorlevel 1 (
        echo Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)

REM Activate virtual environment
echo Activating virtual environment...
call %VENV_NAME%\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements...
if exist "requirements.txt" (
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install requirements
        pause
        exit /b 1
    )
    echo Requirements installed successfully
) else (
    echo requirements.txt not found
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo No .env file found. Creating from .env.example...
    if exist ".env.example" (
        copy .env.example .env
        echo Please edit .env file with your configuration
    ) else (
        echo .env.example not found
        pause
        exit /b 1
    )
)

REM Create data directories
echo Creating data directories...
mkdir data\raw 2>nul
mkdir data\processed 2>nul
mkdir data\embeddings 2>nul

REM Set Python path
set PYTHONPATH=%PYTHONPATH%;%cd%

REM Start the API server
echo Starting Crawl2Answer API server...
echo API will be available at: http://localhost:%API_PORT%
echo API Documentation: http://localhost:%API_PORT%/docs
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start with uvicorn
python -m uvicorn api.main:app --host %API_HOST% --port %API_PORT% --reload

REM Deactivate virtual environment when done
call deactivate