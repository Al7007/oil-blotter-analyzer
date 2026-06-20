@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
title Анализ капельного теста масла

REM Для работы нужен Python 3.10+ и пакеты из requirements.txt.
REM Скрипт bootstrap.ps1 установит их автоматически и сообщит, что скачивается.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap.ps1"
exit /b %ERRORLEVEL%
