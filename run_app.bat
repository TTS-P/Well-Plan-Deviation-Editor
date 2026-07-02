@echo off
REM ============================================================================
REM  Well Plan Deviation Editor - run from source (no .exe needed).
REM  Double-click this file. It installs the required Python packages the first
REM  time, then launches the app in your web browser.
REM
REM  Requires Python 3 installed and on PATH (https://www.python.org/downloads/;
REM  tick "Add python.exe to PATH" in the installer).
REM ============================================================================
setlocal
cd /d "%~dp0"

REM --- Find a Python launcher: prefer the "py" launcher, fall back to python ---
set "PY=py"
where py >nul 2>nul || set "PY=python"
where %PY% >nul 2>nul || goto :nopython

echo Checking dependencies (first run installs them; this can take a minute)...
%PY% -m pip install --quiet --disable-pip-version-check -r requirements.txt || goto :pipfail

echo Starting the app - your browser will open shortly.
echo Leave this window open while you use the app. Close it to stop.
echo.
%PY% -m streamlit run app.py --browser.gatherUsageStats=false
goto :eof

:nopython
echo.
echo ERROR: Python was not found.
echo Install Python 3 from https://www.python.org/downloads/ and be sure to
echo tick "Add python.exe to PATH" during setup, then run this file again.
echo.
pause
exit /b 1

:pipfail
echo.
echo ERROR: Could not install the required packages. Check your internet
echo connection and that Python/pip are working, then try again.
echo.
pause
exit /b 1
