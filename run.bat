@echo off
rem Launch Probabilistic Monkey in a local web page.
setlocal
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (where python >nul 2>nul && set "PY=python")
if not defined PY (
  echo Python 3.9 or newer is required: https://www.python.org/downloads/
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo First run: creating a virtual environment in .venv ...
  %PY% -m venv .venv
  if errorlevel 1 ( pause & exit /b 1 )
)
".venv\Scripts\python.exe" -m pip install --quiet --disable-pip-version-check -r app\requirements.txt
if errorlevel 1 ( pause & exit /b 1 )
".venv\Scripts\python.exe" app\server.py %*
pause
