@echo off
title FastAPI Backend - Handwritten Character Recognition
cd /d "c:\Users\rreji\Downloads\hand written\ml-service"
"C:\Users\rreji\AppData\Local\Programs\Python\Python311\python.exe" main.py
if %errorlevel% neq 0 (
    echo.
    echo Backend server exited with error code %errorlevel%
    pause
)
