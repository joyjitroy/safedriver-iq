@echo off
REM Run the complete PRISM-AR real-data pipeline, ablation, robustness, and figure generation.

set PYTHON=C:\prismar_venv\Scripts\python.exe
set PROJECT=C:\Personal\EB1A\1. Project Description\American Center for Mobility Project\03_Conference_ASCE2027\09_PRISM-AR - IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY

cd /d "%PROJECT%"

echo Running tests...
%PYTHON% -m pytest prism_ar/tests/ -v

echo Running full real-data pipeline...
%PYTHON% run_prism_ar_real_data.py --max_scenes 50

echo Running ablation study...
%PYTHON% run_ablation_study.py

echo Running robustness study...
%PYTHON% run_robustness_study.py

echo Generating figures...
%PYTHON% generate_prism_ar_figures.py

echo Generating consolidated report...
%PYTHON% generate_prism_ar_report.py

echo Done. Outputs are in %PROJECT%\results
