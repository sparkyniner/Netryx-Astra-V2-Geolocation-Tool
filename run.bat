@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python test_super.py
pause
