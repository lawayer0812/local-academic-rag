#!/bin/bash

cd "$(dirname "$0")"

echo "========================================"
echo "  本地学术文献助手 - Mac 一键安装"
echo "========================================"
echo

echo "1. 检查 Python..."
if ! command -v python3 >/dev/null 2>&1; then
    echo "未检测到 Python 3。"
    echo "请先安装 Python 3，然后重新运行本安装程序。"
    read -p "按回车键退出..."
    exit 1
fi

echo "已检测到：$(python3 --version)"
echo

echo "2. 创建虚拟环境..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
else
    echo ".venv 已存在，跳过创建。"
fi

source .venv/bin/activate

echo
echo "3. 更新 pip..."
python -m pip install --upgrade pip

echo
echo "4. 安装项目依赖..."
python -m pip install -r requirements.txt

echo
echo "5. 创建必要文件夹..."
mkdir -p papers
mkdir -p chroma_db

echo
echo "6. 初始化环境变量..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "已根据 .env.example 创建 .env"
    else
        echo "DEEPSEEK_API_KEY=your_deepseek_api_key_here" > .env
        echo "已创建默认 .env"
    fi
else
    echo ".env 已存在，跳过创建。"
fi

echo
echo "========================================"
echo "安装完成！"
echo
echo "下一步："
echo "1. 打开项目文件夹中的 .env 文件"
echo "2. 将 your_deepseek_api_key_here 替换成你自己的 DeepSeek API Key"
echo "3. 双击“启动学术文献助手.command”启动程序"
echo "========================================"
echo

read -p "按回车键退出..."
