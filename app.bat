@echo off
cd /d "%~dp0"  # Automatically set the directory to where the script is
streamlit run views/ui_components.py
