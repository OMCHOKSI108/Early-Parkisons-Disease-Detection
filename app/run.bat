@echo off
:: Simple Windows batch runner for development
:: Usage (from repository root):
::    app\run.bat










POPDpython main.py
nPUSHD %~dp0IF "%ALLOW_SQLITE_FALLBACK%"=="" SET ALLOW_SQLITE_FALLBACK=true
nSET ALLOW_SQLITE_FALLBACK=%ALLOW_SQLITE_FALLBACK%IF "%HOST%"=="" SET HOST=0.0.0.0SET HOST=%HOST%IF "%PORT%"=="" SET PORT=8000nSET PORT=%PORT%