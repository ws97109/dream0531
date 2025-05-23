#!/bin/bash

# 設置環境變量
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 檢查 Fooocus 路徑
FOOOCUS_PATH="path/to/fooocus"  # 請替換為您的 Fooocus 實際路徑

# 啟動 Fooocus API 服務
echo "啟動 Fooocus API 服務..."
cd $FOOOCUS_PATH
python launch.py --listen --port 8888 --api &
FOOOCUS_PID=$!

# 等待 Fooocus API 啟動
echo "等待 Fooocus API 啟動..."
sleep 10

# 返回原始目錄
cd -

# 啟動夢境分析網頁應用
echo "啟動夢境分析網頁應用..."
python app.py &
APP_PID=$!

# 捕獲 SIGINT 和 SIGTERM 訊號
trap "echo '關閉服務...'; kill $FOOOCUS_PID; kill $APP_PID; exit" SIGINT SIGTERM

# 等待所有後台進程完成
wait