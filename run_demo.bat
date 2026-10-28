@echo off
setlocal
cd /d "%~dp0"

echo Seeding the ResolveAI database...
py -m data.seed_data
if errorlevel 1 goto :failed

echo Ingesting the knowledge base...
py -m kb.ingest
if errorlevel 1 goto :failed

echo Starting the FastAPI server in a new window...
start "ResolveAI API" /D "%~dp0" py -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

echo Waiting 5 seconds for the API to start...
timeout /t 5 /nobreak >nul

echo Starting the Streamlit chat...
py -m streamlit run ui/chat_app.py
exit /b %errorlevel%

:failed
echo ResolveAI setup failed. Check the error above.
pause
exit /b 1