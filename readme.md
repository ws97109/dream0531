# 夢境分析與視覺化系統

這是一個整合了多個AI模型的夢境分析系統，可以將使用者的夢境描述轉換成故事、圖像，並提供心理分析。

## 主要功能

1. **夢境故事生成**：使用Ollama的Qwen模型將零散的夢境元素編織成故事
2. **故事翻譯**：將生成的中文故事翻譯成英文
3. **圖像生成**：使用Fooocus基於翻譯後的故事生成視覺圖像
4. **心理分析**：使用Ollama的Qwen模型對夢境進行心理解析

## 系統需求

- Python 3.8+
- Docker & Docker Compose
- NVIDIA GPU (建議用於圖像生成)
- 至少8GB RAM

## 安裝指南

### 使用Docker (推薦)

1. 確保已安裝Docker和Docker Compose
2. 克隆本儲存庫：
   ```
   git clone https://github.com/ws97109/dream-analyzer.git
   cd dream-analyzer
   ```
3. 啟動服務：
   ```
   docker-compose up -d
   ```
4. 在瀏覽器中訪問：http://localhost:5000

### 手動安裝

1. 確保已安裝Python 3.8+
2. 克隆本儲存庫：
   ```
   git clone https://github.com/ws97109/dream-analyzer.git
   cd dream-analyzer
   ```
3. 安裝依賴：
   ```
   pip install -r requirements.txt
   ```
4. 確保已安裝Ollama並下載Qwen模型：
   ```
   ollama pull qwen:7b
   ```
5. 下載並設置Fooocus
6. 運行應用：
   ```
   python app.py
   ```
7. 在瀏覽器中訪問：http://localhost:5000

## 使用方法

1. 在文本框中輸入您的夢境描述（至少10個字）
2. 點擊「開始分析」按鈕
3. 等待系統處理（這可能需要幾分鐘，取決於您的硬件）
4. 查看生成的故事、圖像和心理分析

## 技術架構

- **前端**：HTML, CSS, JavaScript, Bootstrap 5
- **後端**：Python Flask
- **AI模型**：
  - Ollama的Qwen模型：用於故事生成、翻譯和心理分析
  - Fooocus：用於圖像生成

## 自訂配置

您可以在`docker-compose.yml`或環境變數中調整以下設置：

- `DEBUG`：設置為`True`以啟用調試模式
- `PORT`：應用的端口號
- `OLLAMA_HOST`：Ollama服務的主機地址

## 問題排解

- **圖像生成失敗**：檢查Fooocus安裝和GPU驅動
- **模型回應緩慢**：考慮使用更強大的GPU或減少步驟數
- **連接到Ollama失敗**：確保Ollama服務正在運行並可訪問

## 授權

MIT

## 貢獻

歡迎提交問題和合併請求！