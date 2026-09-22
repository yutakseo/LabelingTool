@echo off
setlocal

rem Always run from this project's directory, regardless of where this file is opened from.
cd /d "%~dp0"

rem Prefer a project-local virtual environment when one exists.
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py %*
) else (
    py -3 main.py %*
    if errorlevel 9009 python main.py %*
)

if errorlevel 1 (
    echo.
    echo The labeler could not start. Install dependencies with:
    echo   python -m pip install -r requirements.txt
    pause
)

endlocal
