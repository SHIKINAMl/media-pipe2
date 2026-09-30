@echo off
cd /d "%~dp0"
python -m venv .venv || goto :error
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt || goto :error
echo.
echo Setup finished. Run run.bat to start the game.
pause
exit /b 0

:error
echo.
echo Setup failed. Check that Python is installed and on PATH.
pause
exit /b 1
