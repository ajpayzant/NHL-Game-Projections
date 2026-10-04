@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
rem open the browser once the server has had a few seconds to start
start "" /b powershell -NoProfile -Command "Start-Sleep 6; Start-Process 'http://localhost:8670'"
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8670 --server.headless true
