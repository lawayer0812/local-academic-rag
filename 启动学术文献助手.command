#!/bin/bash

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "没有找到虚拟环境 .venv"
    echo "请先运行安装程序。"
    read -p "按回车键退出..."
    exit 1
fi

source .venv/bin/activate

python -m streamlit run web_app.py --server.headless true
