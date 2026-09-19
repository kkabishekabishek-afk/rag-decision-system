@echo off
echo ========================================================
echo   Launching Self-Adaptive Multi-Agent RAG Decision Engine
echo ========================================================
echo.

cd /d "%~dp0"
call .venv\Scripts\activate.bat
streamlit run app.py --server.port=8501 --server.address=0.0.0.0

pause
