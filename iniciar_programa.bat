@echo off
rem Abre el programa en tu navegador (http://localhost:8501). Cierra esta ventana para detenerlo.
cd /d "%~dp0"
python -m pip install -q -r requirements.txt
python -m streamlit run app_web.py
pause
