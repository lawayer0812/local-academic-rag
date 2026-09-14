cat > "启动学术文献助手.command" <<'EOF'
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
EOF

chmod +x "启动学术文献助手.command"