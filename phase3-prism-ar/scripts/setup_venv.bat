@echo off
REM Create a clean virtual environment for PRISM-AR
REM Run this from the project root

python -m venv C:\prismar_venv
C:\prismar_venv\Scripts\python.exe -m pip install --upgrade pip
C:\prismar_venv\Scripts\python.exe -m pip install -r requirements.txt

echo Virtual environment ready at C:\prismar_venv
