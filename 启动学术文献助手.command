#!/bin/bash

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "没有找到虚拟环境 .venv"
    echo "请先运行安装程序。"
    read -p "按回车键退出..."
    exit 1
fi

source .venv/bin/activate

RESTART_FLAG=".rag_restart_requested"

while true; do
    rm -f "$RESTART_FLAG"

    python -m streamlit run web_app.py --server.headless true
    EXIT_CODE=$?

    if [ "$EXIT_CODE" -eq 42 ] || [ -f "$RESTART_FLAG" ]; then
        rm -f "$RESTART_FLAG"
        echo
        echo "知识库已更新，正在自动重启服务..."
        sleep 1
        continue
    fi

    exit "$EXIT_CODE"
done
