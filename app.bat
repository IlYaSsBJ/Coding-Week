@echo off
setlocal

cd /d "%~dp0"
set SCRIPT_PATH=views\ui_components.py

:: Run Streamlit app
streamlit run %SCRIPT_PATH%

endlocal
