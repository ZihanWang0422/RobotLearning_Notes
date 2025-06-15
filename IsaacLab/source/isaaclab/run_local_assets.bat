@echo off
REM IsaacLab本地资产工具
REM 此脚本帮助用户使用本地资产替代Nucleus服务器资产

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set LOCAL_ASSETS_DIR=%SCRIPT_DIR%local_assets
set PYTHON_SCRIPT=%SCRIPT_DIR%launch_with_local_assets.py

:MENU
cls
echo.
echo  IsaacLab本地资产工具
echo  ------------------------------
echo  1. 下载蚂蚁模型到本地目录
echo  2. 使用本地资产路径启动IsaacLab（带界面）
echo  3. 使用本地资产路径启动IsaacLab（无界面）
echo  4. 退出
echo.
set /p choice=请选择操作(1-4): 

if "%choice%"=="1" goto SETUP_ANT
if "%choice%"=="2" goto START_GUI
if "%choice%"=="3" goto START_HEADLESS
if "%choice%"=="4" goto EXIT
goto MENU

:SETUP_ANT
echo 创建本地目录结构并设置蚂蚁模型...
if not exist "%LOCAL_ASSETS_DIR%" mkdir "%LOCAL_ASSETS_DIR%"
if not exist "%LOCAL_ASSETS_DIR%\Isaac\Robots\Ant" mkdir "%LOCAL_ASSETS_DIR%\Isaac\Robots\Ant"

REM 启动Python脚本，只用于创建资产
python %PYTHON_SCRIPT% --headless --assets_path="%LOCAL_ASSETS_DIR%" --script=""

echo.
echo 蚂蚁模型已设置在: %LOCAL_ASSETS_DIR%\Isaac\Robots\Ant\ant_instanceable.usd
echo.
pause
goto MENU

:START_GUI
echo 使用本地资产启动IsaacLab（带界面）...
python %PYTHON_SCRIPT% --assets_path="%LOCAL_ASSETS_DIR%"
goto MENU

:START_HEADLESS
echo 使用本地资产启动IsaacLab（无界面）...
python %PYTHON_SCRIPT% --headless --assets_path="%LOCAL_ASSETS_DIR%"
pause
goto MENU

:EXIT
echo 退出程序...
exit /b 0 