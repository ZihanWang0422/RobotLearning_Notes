#!/bin/bash
# IsaacLab本地资产工具
# 此脚本帮助用户使用本地资产替代Nucleus服务器资产

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
LOCAL_ASSETS_DIR="${SCRIPT_DIR}/local_assets"
PYTHON_SCRIPT="${SCRIPT_DIR}/launch_with_local_assets.py"

function show_menu() {
    clear
    echo
    echo " IsaacLab本地资产工具"
    echo " ------------------------------"
    echo " 1. 下载蚂蚁模型到本地目录"
    echo " 2. 使用本地资产路径启动IsaacLab（带界面）"
    echo " 3. 使用本地资产路径启动IsaacLab（无界面）"
    echo " 4. 退出"
    echo
    read -p "请选择操作(1-4): " choice
    
    case $choice in
        1) setup_ant ;;
        2) start_gui ;;
        3) start_headless ;;
        4) exit 0 ;;
        *) show_menu ;;
    esac
}

function setup_ant() {
    echo "创建本地目录结构并设置蚂蚁模型..."
    mkdir -p "${LOCAL_ASSETS_DIR}/Isaac/Robots/Ant"
    
    # 启动Python脚本，只用于创建资产
    python "${PYTHON_SCRIPT}" --headless --assets_path="${LOCAL_ASSETS_DIR}" --script=""
    
    echo
    echo "蚂蚁模型已设置在: ${LOCAL_ASSETS_DIR}/Isaac/Robots/Ant/ant_instanceable.usd"
    echo
    read -p "按回车键继续..."
    show_menu
}

function start_gui() {
    echo "使用本地资产启动IsaacLab（带界面）..."
    python "${PYTHON_SCRIPT}" --assets_path="${LOCAL_ASSETS_DIR}"
    show_menu
}

function start_headless() {
    echo "使用本地资产启动IsaacLab（无界面）..."
    python "${PYTHON_SCRIPT}" --headless --assets_path="${LOCAL_ASSETS_DIR}"
    read -p "按回车键继续..."
    show_menu
}

# 主程序入口
show_menu 