@echo off
setlocal

cd /d "%~dp0"

cd ..


streamlit run views\ui_components.py

endlocal
