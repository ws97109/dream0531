#!/usr/bin/env python3
# 測試 Fooocus API 是否正常工作

import requests
import base64
import os
import time
import argparse

def test_fooocus_api(prompt="A beautiful dreamlike forest with glowing mushrooms and a small cottage"):
    """測試 Fooocus API"""
    print(f"測試 Fooocus API，提示詞：'{prompt}'")
    
    api_url = "http://localhost:8888/v1/generation/text-to-image"
    
    payload = {
        "prompt": prompt,
        "negative_prompt": "blurry, low quality, deformed",
        "style_name": "dreamlike",
        "performance_selection": "speed",
        "aspect_ratios_selection": "1024×1024",
        "image_number": 1,
        "image_quality": 95
    }
    
    try:
        print("發送請求到 Fooocus API...")
        response = requests.post(api_url, json=payload, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            if "images" in result and len(result["images"]) > 0:
                # 從Base64解碼並保存圖片
                image_data = base64.b64decode(result["images"][0])
                
                # 生成輸出檔名
                timestamp = int(time.time())
                output_filename = f"test_output_{timestamp}.png"
                
                # 保存圖片
                with open(output_filename, "wb") as f:
                    f.write(image_data)
                
                print(f"測試成功！圖像已保存為 {output_filename}")
                return True
            else:
                print(f"API返回了結果，但沒有找到圖像: {result}")
        else:
            print(f"API請求失敗: HTTP {response.status_code}")
            print(f"回應內容: {response.text}")
        
        return False
    
    except Exception as e:
        print(f"測試過程中發生錯誤: {str(e)}")
        print("請確保 Fooocus API 服務正在運行。")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='測試 Fooocus API')
    parser.add_argument('--prompt', type=str, default="A beautiful dreamlike forest with glowing mushrooms and a small cottage",
                        help='用於生成圖像的提示詞')
    
    args = parser.parse_args()
    test_fooocus_api(args.prompt)