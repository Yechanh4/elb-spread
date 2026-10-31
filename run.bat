@echo off
REM Windows: first run creates the venv and installs, then starts http://localhost:8780
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3.12 -m venv .venv || python -m venv .venv
  .venv\Scripts\pip install -r requirements.txt
)
if "%PORT%"=="" set PORT=8780
.venv\Scripts\uvicorn app.main:app --port %PORT%
