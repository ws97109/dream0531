#!/usr/bin/env python3
"""
FramePack 設置腳本
用於自動下載和配置 FramePack 依賴項目
"""

import os
import sys
import subprocess
import requests
import zipfile
import shutil
from pathlib import Path

def run_command(command, check=True):
    """執行命令並處理錯誤"""
    print(f"🔄 執行: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ 命令執行失敗: {e}")
        if e.stderr:
            print(f"錯誤訊息: {e.stderr}")
        return False

def download_file(url, destination):
    """下載文件"""
    print(f"📥 下載: {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ 下載完成: {destination}")
        return True
    except Exception as e:
        print(f"❌ 下載失敗: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """解壓 ZIP 文件"""
    print(f"📦 解壓: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ 解壓完成: {extract_to}")
        return True
    except Exception as e:
        print(f"❌ 解壓失敗: {e}")
        return False

def check_gpu():
    """檢查 GPU 可用性"""
    print("🔍 檢查 GPU 狀態...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            print(f"✅ CUDA 可用")
            print(f"   GPU 數量: {gpu_count}")
            print(f"   GPU 型號: {gpu_name}")
            print(f"   顯存容量: {memory_gb:.1f} GB")
            
            if memory_gb >= 6:
                print("✅ 顯存容量足夠 (≥6GB)")
                return True
            else:
                print("⚠️ 顯存容量可能不足 (<6GB)")
                return False
                
        elif torch.backends.mps.is_available():
            print("✅ MPS (Apple Silicon) 可用")
            print("⚠️ MPS 支援可能有限，建議使用 CUDA")
            return True
        else:
            print("❌ 未檢測到 GPU 支援")
            return False
            
    except ImportError:
        print("❌ PyTorch 未安裝")
        return False

def install_dependencies():
    """安裝必要的依賴項目"""
    print("📦 安裝 Python 依賴項目...")
    
    # 基本依賴項目
    basic_deps = [
        "torch>=2.0.0",
        "torchvision",
        "torchaudio",
        "diffusers>=0.33.0",
        "transformers>=4.46.0",
        "accelerate>=1.6.0",
        "sentencepiece",
        "safetensors",
        "einops",
        "opencv-contrib-python",
        "pillow",
        "numpy",
        "scipy"
    ]
    
    for dep in basic_deps:
        if not run_command(f"pip install {dep}"):
            print(f"⚠️ 安裝 {dep} 失敗，可能需要手動安裝")
    
    # 可選的高級依賴項目
    print("\n🔧 安裝可選的加速庫...")
    
    # xformers (NVIDIA GPU 加速)
    if run_command("pip install xformers", check=False):
        print("✅ xformers 安裝成功")
    else:
        print("⚠️ xformers 安裝失敗（可選）")
    
    # flash-attn (進階注意力機制)
    if run_command("pip install flash-attn", check=False):
        print("✅ flash-attn 安裝成功")
    else:
        print("⚠️ flash-attn 安裝失敗（可選）")

def setup_framepack():
    """設置 FramePack"""
    print("🎬 設置 FramePack...")
    
    current_dir = Path.cwd()
    framepack_dir = current_dir / "FramePack"
    
    # 檢查是否已存在 FramePack 目錄
    if framepack_dir.exists():
        print("✅ FramePack 目錄已存在")
        response = input("是否要重新下載 FramePack？(y/N): ").lower().strip()
        if response != 'y':
            return True
        else:
            shutil.rmtree(framepack_dir)
    
    # 下載 FramePack
    framepack_url = "https://github.com/lllyasviel/FramePack/archive/refs/heads/main.zip"
    zip_path = current_dir / "framepack.zip"
    
    if not download_file(framepack_url, zip_path):
        return False
    
    # 解壓 FramePack
    if not extract_zip(zip_path, current_dir):
        return False
    
    # 重命名目錄
    extracted_dir = current_dir / "FramePack-main"
    if extracted_dir.exists():
        extracted_dir.rename(framepack_dir)
    
    # 清理下載的 ZIP 文件
    zip_path.unlink()
    
    # 創建必要的目錄
    (framepack_dir / "hf_download").mkdir(exist_ok=True)
    (framepack_dir / "outputs").mkdir(exist_ok=True)
    
    print("✅ FramePack 設置完成")
    return True

def test_framepack():
    """測試 FramePack 功能"""
    print("🧪 測試 FramePack 功能...")
    
    try:
        # 測試導入
        sys.path.insert(0, str(Path.cwd() / "FramePack"))
        from framepack_video_generator import get_video_generator
        
        # 測試初始化
        generator = get_video_generator()
        
        if generator.is_available():
            print("✅ FramePack 測試成功")
            return True
        else:
            print("⚠️ FramePack 可用但可能有限制")
            return False
            
    except Exception as e:
        print(f"❌ FramePack 測試失敗: {e}")
        return False

def create_requirements_file():
    """創建完整的 requirements.txt"""
    print("📝 創建 requirements.txt...")
    
    requirements = """
# 基本依賴
torch>=2.0.0
torchvision
torchaudio
diffusers>=0.33.0
transformers>=4.46.0
accelerate>=1.6.0
sentencepiece>=0.2.0
safetensors
einops
opencv-contrib-python
pillow>=11.0.0
numpy>=1.26.0
scipy>=1.12.0

# Web 應用
flask
requests

# FramePack 特定依賴
gradio>=5.20.0
av>=12.0.0
torchsde>=0.2.6

# 可選加速庫（需要相容的 GPU）
# xformers  # 取消註釋以啟用
# flash-attn  # 取消註釋以啟用
# sageattention  # 取消註釋以啟用
""".strip()
    
    with open("requirements_framepack.txt", "w", encoding="utf-8") as f:
        f.write(requirements)
    
    print("✅ requirements_framepack.txt 已創建")

def main():
    """主要設置流程"""
    print("🚀 FramePack 設置腳本")
    print("=" * 50)
    
    # 檢查 Python 版本
    if sys.version_info < (3, 8):
        print("❌ 需要 Python 3.8 或更高版本")
        sys.exit(1)
    
    print(f"✅ Python 版本: {sys.version}")
    
    # 步驟 1: 檢查 GPU
    gpu_available = check_gpu()
    if not gpu_available:
        response = input("⚠️ 未檢測到足夠的 GPU 資源，是否繼續？(y/N): ").lower().strip()
        if response != 'y':
            print("❌ 設置已取消")
            sys.exit(1)
    
    # 步驟 2: 安裝依賴項目
    print("\n" + "=" * 50)
    install_dependencies()
    
    # 步驟 3: 設置 FramePack
    print("\n" + "=" * 50)
    if not setup_framepack():
        print("❌ FramePack 設置失敗")
        sys.exit(1)
    
    # 步驟 4: 測試功能
    print("\n" + "=" * 50)
    if not test_framepack():
        print("⚠️ FramePack 測試未通過，但設置已完成")
        print("   請檢查 GPU 驅動和依賴項目")
    
    # 步驟 5: 創建 requirements 文件
    print("\n" + "=" * 50)
    create_requirements_file()
    
    # 完成
    print("\n" + "=" * 50)
    print("🎉 FramePack 設置完成！")
    print("\n下一步:")
    print("1. 確保 Ollama 服務正在運行")
    print("2. 運行 'python app.py' 啟動夢境分析應用")
    print("3. 訪問 http://localhost:5002 使用應用")
    print("\n注意事項:")
    print("- 首次使用會自動下載 30GB+ 的模型文件")
    print("- 確保有足夠的磁碟空間和網路連接")
    print("- 影片生成需要 6GB+ 顯存")
    
    if gpu_available:
        print("✅ 您的系統支援 GPU 加速")
    else:
        print("⚠️ 建議使用具有足夠顯存的 NVIDIA GPU")

if __name__ == "__main__":
    main()