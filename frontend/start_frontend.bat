@echo off
title Vite Frontend - Handwritten Character Recognition
cd /d "c:\Users\rreji\Downloads\hand written\frontend"
npm run dev
if %errorlevel% neq 0 (
    echo.
    echo Frontend server exited with error code %errorlevel%
    pause
)
