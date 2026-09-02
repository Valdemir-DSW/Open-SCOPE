@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================
echo OpenScope - Windows build
echo ============================================================

set "PY_CMD="
where py.exe >nul 2>nul && set "PY_CMD=py -3"
if not defined PY_CMD (
    where python.exe >nul 2>nul && set "PY_CMD=python"
)
if not defined PY_CMD (
    echo [ERROR] Python 3 was not found.
    exit /b 1
)

echo [1/4] Checking Python...
%PY_CMD% -c "import sys; assert sys.version_info >= (3,9); print(sys.version)"
if errorlevel 1 exit /b 1

echo [2/4] Installing/updating build dependencies...
%PY_CMD% -m pip install --upgrade pip
if errorlevel 1 exit /b 1
%PY_CMD% -m pip install -r requirements.txt -r requirements-build.txt
if errorlevel 1 exit /b 1

set "ICON_ARG="
if exist "resources\OpenScope.ico" set "ICON_ARG=--windows-icon-from-ico=resources\OpenScope.ico"

echo [3/4] Building OpenScope with Nuitka...
%PY_CMD% -m nuitka ^
  --standalone ^
  --enable-plugin=pyqt5 ^
  --assume-yes-for-downloads ^
  --windows-console-mode=disable ^
  --remove-output ^
  --output-dir=build ^
  --output-filename=OpenScope.exe ^
  --company-name=Falido ^
  --product-name=OpenScope ^
  --file-description="OpenScope Oscilloscope" ^
  --file-version=1.0.0.0 ^
  --product-version=1.0.0.0 ^
  --include-package=pyqtgraph ^
  --include-data-dir=resources=resources ^
  %ICON_ARG% ^
  main.py
if errorlevel 1 exit /b 1

if not exist "build\main.dist\OpenScope.exe" (
    echo [ERROR] Nuitka finished but build\main.dist\OpenScope.exe was not found.
    exit /b 1
)

echo [4/4] Searching for Inno Setup Compiler ISCC.exe...
set "ISCC="
for /f "delims=" %%I in ('where ISCC.exe 2^>nul') do if not defined ISCC set "ISCC=%%I"

if not defined ISCC if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles(x86)%\Inno Setup 5\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 5\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles%\Inno Setup 5\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 5\ISCC.exe"

if not defined ISCC (
    for /f "tokens=2,*" %%A in ('reg query "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1" /v InstallLocation 2^>nul ^| find /I "InstallLocation"') do if exist "%%B\ISCC.exe" set "ISCC=%%B\ISCC.exe"
)
if not defined ISCC (
    for /f "tokens=2,*" %%A in ('reg query "HKLM\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1" /v InstallLocation 2^>nul ^| find /I "InstallLocation"') do if exist "%%B\ISCC.exe" set "ISCC=%%B\ISCC.exe"
)
if not defined ISCC (
    for /f "tokens=2,*" %%A in ('reg query "HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1" /v InstallLocation 2^>nul ^| find /I "InstallLocation"') do if exist "%%B\ISCC.exe" set "ISCC=%%B\ISCC.exe"
)

if not defined ISCC if exist "%ProgramFiles(x86)%\" (
    for /f "delims=" %%I in ('where /R "%ProgramFiles(x86)%" ISCC.exe 2^>nul') do if not defined ISCC set "ISCC=%%I"
)
if not defined ISCC if exist "%ProgramFiles%\" (
    for /f "delims=" %%I in ('where /R "%ProgramFiles%" ISCC.exe 2^>nul') do if not defined ISCC set "ISCC=%%I"
)

if not defined ISCC (
    echo.
    echo [WARNING] OpenScope was compiled successfully, but Inno Setup was not found.
    echo Install Inno Setup 6 and run this build again to create the installer.
    echo Standalone build: build\main.dist\OpenScope.exe
    exit /b 2
)

echo Found: !ISCC!
set "ISS_ICON_ARG="
if exist "resources\OpenScope.ico" set "ISS_ICON_ARG=--define=OpenScopeIcon=1"
"!ISCC!" !ISS_ICON_ARG! "installer\OpenScope.iss"
if errorlevel 1 exit /b 1

echo.
echo Build complete.
echo Standalone: build\main.dist\OpenScope.exe
echo Installer:  release\OpenScope-Setup.exe
exit /b 0
