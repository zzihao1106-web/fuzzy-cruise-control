@echo off
cd /d "%~dp0"
python tool_gui.py
if errorlevel 1 pause
