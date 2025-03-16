@echo off
setlocal

:: Find the script's directory
cd /d "%~dp0"

:: Navigate to the root of the project (assumes the BAT is inside a subfolder)
cd ..

:: Check if requirements.txt exists
if exist requirements.txt (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
) else (
    echo ERROR: requirements.txt not found!
    echo Make sure the BAT file is in the correct project structure.
    pause
)

endlocal
