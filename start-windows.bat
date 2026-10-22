@echo off
REM HIVEMIND launcher for Windows. Double-click to set up (first time) and start.
REM Everything it creates stays inside this folder: delete the folder to remove it all.
setlocal
cd /d "%~dp0"
title HIVEMIND
set "PIP_NO_CACHE_DIR=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

echo.
echo   HIVEMIND
echo   Everything lives inside this folder. To remove it all: close this window, then delete the folder.
echo.

set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1 && set "PY=py -3"
if defined PY goto :havepy
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1 && set "PY=python"
if defined PY goto :havepy

echo Python 3.10 or newer is needed. Trying to install it now...
winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
echo.
echo When the install has finished, close this window and double-click start-windows.bat again.
echo If it did not install, get Python from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" on the first install screen.
goto :fail_quiet

:havepy
if exist ".env" goto :haveenv
echo First-time setup. Paste each key (right-click to paste) and press Enter.
echo Leave a key empty to skip it (without keys the hive runs in demo mode).
echo.
set "AK="
set "PK="
set "SK="
set "BUD="
set /p "AK=Anthropic API key (sk-ant-...): "
set /p "PK=Alpaca PAPER API key (PK...): "
set /p "SK=Alpaca PAPER secret key: "
set /p "BUD=Stop Claude spending after how many dollars? [20]: "
if "%BUD%"=="" set "BUD=20"
>".env" echo ANTHROPIC_API_KEY=%AK%
>>".env" echo ALPACA_API_KEY=%PK%
>>".env" echo ALPACA_SECRET_KEY=%SK%
>>".env" echo HIVE_LLM_BUDGET_USD=%BUD%
echo.
echo Saved to the .env file in this folder (open it with Notepad to change keys later).
echo.

:haveenv
if exist ".venv\Scripts\python.exe" goto :haveenvdir
echo Creating a private Python environment (.venv) inside this folder...
%PY% -m venv .venv
if errorlevel 1 goto :fail

:haveenvdir
fc /b requirements.txt .venv\requirements.installed >nul 2>&1
if not errorlevel 1 goto :run
echo Installing what HIVEMIND needs (first run takes a few minutes)...
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail
copy /y requirements.txt .venv\requirements.installed >nul

:run
echo.
echo Starting HIVEMIND. The dashboard opens in your browser (http://127.0.0.1:8000).
echo Keep this window open while it runs. Press Ctrl+C or close the window to stop it.
echo.
set "HIVE_OPEN_BROWSER=1"
".venv\Scripts\python.exe" -m hive
echo.
echo HIVEMIND stopped.
pause
exit /b 0

:fail
echo.
echo Something went wrong (see the messages above). Check your internet connection and try again.
:fail_quiet
echo.
pause
exit /b 1
