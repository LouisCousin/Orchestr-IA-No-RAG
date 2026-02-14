@echo off
title Orchestr'IA
echo ========================================
echo        Lancement d'Orchestr'IA
echo ========================================
echo.

:: Chercher Python
where python >nul 2>nul
if %errorlevel%==0 (
    set PYTHON=python
) else (
    where python3 >nul 2>nul
    if %errorlevel%==0 (
        set PYTHON=python3
    ) else (
        echo ERREUR : Python n'est pas installe ou n'est pas dans le PATH.
        echo Installez Python depuis https://www.python.org/downloads/
        pause
        exit /b 1
    )
)

:: Se placer dans le dossier du script
cd /d "%~dp0"

:: Verifier que les dependances sont installees
%PYTHON% -c "import streamlit" >nul 2>nul
if %errorlevel% neq 0 (
    echo Installation des dependances...
    %PYTHON% -m pip install -r requirements.txt
    echo.
)

:: Lancer l'application
echo Demarrage du serveur Streamlit...
echo L'application va s'ouvrir dans votre navigateur.
echo Pour arreter : fermez cette fenetre ou appuyez sur Ctrl+C.
echo.
%PYTHON% -m streamlit run src/app.py
pause
