@echo off
echo Iniciando Jira Dashboard...
echo.
cd /d "%~dp0"
python -m streamlit run app.py
pause
