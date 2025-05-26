import os
import time
import subprocess
import tempfile
import uuid
import json
import logging
import sys
from flask import Flask, request, jsonify, render_template, url_for
import requests
import base64
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import numpy as np

# 新增的導入
import torch
from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline
import cv2

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app_root = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(app_root, 'static')

app = Flask(__name__, 
            template_folder=os.path.join(app_root, 'templates'),
            static_folder=static_dir)

# ==================== 配置設定 ====================
# Ollama API（僅用於文本生成）
OLLAMA_API = "http://localhost:11434/api/generate"

# 全局變數
image_pipe = None
video_pipe = None
models_loaded = False

# ==================== 本地模型初始化 ====================

def initialize_local_models():
    """初始化本地 Diffusers 模型"""
    global image_pipe, video_pipe, models_loaded
    
    if models_loaded:
        return True
    
    try:
        logger.info("初始化本地 Diffusers 模型...")
        
        # 檢測設備
        if torch.backends.mps.is_available():
            device = "mps"
            torch_dtype = torch.float16
        elif torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
        else:
            device = "cpu"
            torch_dtype = torch.float32
        
        logger.info(f"使用設備: {device}")
        
        # 文字轉圖像模型 - 使用較小的模型
        logger.info("載入圖像生成模型...")
        image_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch_dtype,
            use_safetensors=True,
            variant="fp16" if torch_dtype == torch.float16 else None,
            safety_checker=None,          # 禁用安全檢查器
            requires_safety_checker=False  # 不需要安全檢查器
        )
        image_pipe = image_pipe.to(device)
        
        # 啟用記憶體優化
        if device != "mps":  # MPS 暫不支援某些優化
            image_pipe.enable_model_cpu_offload()
        image_pipe.enable_attention_slicing()
        
        # 圖像轉視頻模型 - 使用輕量版本
        logger.info("載入視頻生成模型...")
        video_pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch_dtype,
            variant="fp16" if torch_dtype == torch.float16 else None
        )
        video_pipe = video_pipe.to(device)
        
        # 視頻模型記憶體優化
        if device != "mps":
            video_pipe.enable_model_cpu_offload()
        
        models_loaded = True
        logger.info("✅ 本地模型初始化完成")
        return True
        
    except Exception as e:
        logger.error(f"模型初始化失敗: {str(e)}")
        return False

# ==================== 故事轉圖像提示詞 ====================

def story_to_image_prompt(story_text):
    """將故事內容轉換為適合圖像生成的提示詞"""
    try:
        system_prompt = """你是一位專業的AI繪畫提示詞專家。請將用戶提供的中文故事內容轉換為適合Stable Diffusion圖像生成的英文提示詞。

要求：
1. 提取故事中的核心視覺元素
2. 轉換為簡潔的英文關鍵詞
3. 按重要性排序，最重要的放前面
4. 包含畫面風格、色彩、情境等描述
5. 避免過於複雜的句子，使用逗號分隔的關鍵詞
6. 長度控制在50-80個英文單詞內

格式範例：
主要對象, 動作/狀態, 環境/背景, 色彩風格, 畫面質量詞

請只返回英文提示詞，不要解釋。"""

        user_prompt = f"""
        故事內容：{story_text}
        
        請轉換為Stable Diffusion圖像生成提示詞：
        """
        
        # 使用 Ollama 進行轉換
        converted_prompt = ollama_generate(system_prompt, user_prompt, "qwen2.5:14b")
        
        if not converted_prompt:
            # 如果 Ollama 失敗，使用簡單的關鍵詞提取
            converted_prompt = extract_visual_keywords(story_text)
        
        # 添加質量增強詞
        enhanced_prompt = f"{converted_prompt}, masterpiece, best quality, highly detailed, cinematic lighting, beautiful composition"
        
        logger.info(f"故事轉換為提示詞: {enhanced_prompt[:100]}...")
        return enhanced_prompt
        
    except Exception as e:
        logger.error(f"故事轉提示詞失敗: {str(e)}")
        return extract_visual_keywords(story_text)

def extract_visual_keywords(text):
    """簡單的視覺關鍵詞提取（備用方案）"""
    # 中英對照的關鍵詞映射
    keyword_map = {
        # 人物
        '女孩': 'girl', '男孩': 'boy', '女人': 'woman', '男人': 'man',
        '公主': 'princess', '王子': 'prince', '天使': 'angel',
        
        # 動作
        '飛行': 'flying', '漂浮': 'floating', '跳舞': 'dancing', '奔跑': 'running',
        '游泳': 'swimming', '行走': 'walking', '坐著': 'sitting',
        
        # 環境
        '海洋': 'ocean', '大海': 'sea', '天空': 'sky', '雲朵': 'clouds',
        '森林': 'forest', '山': 'mountain', '城堡': 'castle', '花園': 'garden',
        '房間': 'room', '橋': 'bridge', '島嶼': 'island',
        
        # 色彩
        '藍色': 'blue', '紅色': 'red', '綠色': 'green', '黃色': 'yellow',
        '紫色': 'purple', '白色': 'white', '黑色': 'black', '金色': 'golden',
        '彩虹': 'rainbow', '閃光': 'glowing', '明亮': 'bright',
        
        # 情境
        '夢境': 'dreamlike', '幻想': 'fantasy', '魔法': 'magical', '神秘': 'mysterious',
        '美麗': 'beautiful', '優雅': 'elegant', '浪漫': 'romantic',
        '日落': 'sunset', '夜晚': 'night', '星空': 'starry sky'
    }
    
    extracted_keywords = []
    text_lower = text.lower()
    
    for chinese, english in keyword_map.items():
        if chinese in text:
            extracted_keywords.append(english)
    
    if not extracted_keywords:
        extracted_keywords = ['dreamlike scene', 'fantasy', 'beautiful']
    
    return ', '.join(extracted_keywords[:8])  # 最多8個關鍵詞

# ==================== 快速圖像生成 ====================

def generate_image_fast_local(story_text):
    """快速本地圖像生成"""
    try:
        # 確保模型已載入
        if not initialize_local_models():
            logger.error("本地模型初始化失敗")
            return create_default_image(f"error_{int(time.time())}.png", story_text)
        
        # 將故事轉換為圖像提示詞
        image_prompt = story_to_image_prompt(story_text)
        
        logger.info(f"開始生成圖像，提示詞: {image_prompt[:50]}...")
        
        # 生成參數（針對速度優化）
        generation_params = {
            "prompt": image_prompt,
            "negative_prompt": "blurry, low quality, distorted, ugly, bad anatomy, deformed, watermark, signature",
            "num_inference_steps": 20,  # 減少步數提升速度
            "guidance_scale": 7.5,
            "width": 512,  # 標準尺寸
            "height": 512,
            "generator": torch.Generator().manual_seed(42)  # 固定種子確保一致性
        }
        
        # 生成圖像
        start_time = time.time()
        result = image_pipe(**generation_params)
        generation_time = time.time() - start_time
        
        logger.info(f"圖像生成完成，耗時: {generation_time:.2f}秒")
        
        # 保存圖像
        timestamp = int(time.time())
        random_id = str(uuid.uuid4())[:8]
        output_filename = f"dream_{timestamp}_{random_id}.png"
        
        output_dir = os.path.join(static_dir, 'images')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        
        result.images[0].save(output_path)
        
        logger.info(f"✅ 圖像保存成功: {output_filename}")
        return os.path.join('images', output_filename)
        
    except Exception as e:
        logger.error(f"快速圖像生成失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return create_default_image(f"error_{int(time.time())}.png", story_text)

# ==================== 快速視頻生成 ====================

def generate_video_fast_local(image_path, story_text):
    """快速本地視頻生成"""
    try:
        # 確保模型已載入
        if not initialize_local_models():
            logger.error("本地模型初始化失敗")
            return None
        
        # 載入圖像
        full_image_path = os.path.join(static_dir, image_path)
        input_image = Image.open(full_image_path)
        
        # 調整圖像尺寸（SVD 需要特定尺寸）
        input_image = input_image.resize((1024, 576), Image.Resampling.LANCZOS)
        
        logger.info("開始生成視頻...")
        
        # 生成參數（針對速度優化）
        video_params = {
            "image": input_image,
            "decode_chunk_size": 8,  # 減少記憶體使用
            "generator": torch.Generator().manual_seed(42),
            "motion_bucket_id": 127,  # 中等運動強度
            "noise_aug_strength": 0.1,  # 較低的噪聲增強
            "num_frames": 14,  # 較少幀數（約0.6秒@25fps）
        }
        
        start_time = time.time()
        frames = video_pipe(**video_params).frames[0]
        generation_time = time.time() - start_time
        
        logger.info(f"視頻幀生成完成，耗時: {generation_time:.2f}秒")
        
        # 保存視頻
        timestamp = int(time.time())
        random_id = str(uuid.uuid4())[:8]
        output_filename = f"dream_video_{timestamp}_{random_id}.mp4"
        
        output_dir = os.path.join(static_dir, 'videos')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        
        # 使用 OpenCV 保存視頻（更快）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 8.0, (1024, 576))  # 8 FPS
        
        for frame in frames:
            frame_array = np.array(frame)
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        
        logger.info(f"✅ 視頻保存成功: {output_filename}")
        return os.path.join('videos', output_filename)
        
    except Exception as e:
        logger.error(f"快速視頻生成失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ==================== 其他輔助函數 ====================

def check_local_services():
    """檢查本地服務狀態"""
    try:
        # 檢查 Ollama
        ollama_response = requests.get("http://localhost:11434/api/tags", timeout=5)
        ollama_status = ollama_response.status_code == 200
        logger.info(f"Ollama API 狀態: {'正常' if ollama_status else '異常'}")
        
        # 檢查本地模型狀態
        local_models_status = models_loaded or initialize_local_models()
        
        return ollama_status, True, local_models_status  # fooocus_status 設為 True（不再使用）
    except Exception as e:
        logger.error(f"服務檢查失敗: {str(e)}")
        return False, False, False

def dream_weaver(prompt):
    """使用Ollama的qwen模型處理夢境故事生成"""
    try:
        system_planner = """請用台灣習慣的中文回覆。你是一個專業的繁體中文故事大綱規劃專家，特別擅長分析夢境元素並創建連貫的故事架構。
        當用戶提供零散的夢境片段時，你的任務是分析每個元素的象徵意義，產生多種可能的故事架構，
        並根據元素之間的潛在聯繫選擇最佳架構。請用台灣習慣的中文回覆。"""

        system_reflector = """請用台灣習慣的中文回覆。你是一位繁體中文的批判性分析專家，專門評估故事的質量和連貫性。
        分析這個基於夢境片段創作的故事，找出它的優點和不足之處，並提供具體的改進建議。請用台灣習慣的中文回覆。"""

        system_writer = """請用台灣習慣的中文回覆。你是一位繁體中文專精於夢境故事創作的作家，能將零散的夢境元素編織成引人入勝的敘事。
        你的故事應該融合所有提供的夢境元素，捕捉夢境特有的超現實性，並符合指定的情緒氛圍。請用台灣習慣的中文回覆。"""

        # 分析夢境片段並規劃故事大綱
        cot_prompt = f"""
        使用者提供了以下夢境片段：{prompt}

        請思考如何將這些元素組織成故事：
        1. 識別所有關鍵元素與可能的象徵意義
        2. 考慮多種可能的故事架構（至少三種）
        3. 評估每個架構的優缺點
        4. 選擇最佳故事大綱
        5. 使用15-30字數總結故事大綱
        6. 請用台灣習慣的中文回覆
        """

        story_outline = ollama_generate(system_planner, cot_prompt, "qwen2.5:14b")
        if not story_outline:
            return "無法生成故事大綱", "無法提供反饋", "無法生成最終故事"

        # 故事創作
        story_prompt = f"""
        基於以下思考過程和大綱：

        {story_outline}

        請創作一個完整的夢境故事，融合所有提供的夢境元素：「{prompt}」。
        故事應展現夢境的超現實性和流動感，長度50-100字，使用繁體中文回答。
        """

        initial_story = ollama_generate(system_writer, story_prompt, "qwen2.5:14b")
        if not initial_story:
            initial_story = "無法生成初始故事"

        # 評估故事並提出改進建議
        reflection_prompt = f"""
        以下是基於用戶夢境片段「{prompt}」創作的初步故事：

        {initial_story}

        請評估這個故事並提出具體改進建議。關注：
        1. 是否所有夢境元素都得到了恰當融合？
        2. 故事的超現實性和夢境感如何？
        3. 哪些部分可以增強以使故事更加引人入勝？
        4. 故事結構和連貫性如何？
        5. 給出10-25字數的總結性評價和建議。
        6. 使用繁體中文回答
        """
        
        story_feedback = ollama_generate(system_reflector, reflection_prompt, "qwen2.5:14b")
        if not story_feedback:
            story_feedback = "無法生成故事反饋"

        # 根據反思優化故事
        final_prompt = f"""
        請根據以下反饋意見改進故事：

        {story_feedback}

        原始故事：
        {initial_story}

        請使用繁體中文創作更加完善的最終版本，確保融合所有夢境元素並增強其超現實性，全部都要使用繁體中文回答。
        """
        
        final_story = ollama_generate(system_writer, final_prompt, "qwen2.5:14b")
        if not final_story:
            final_story = initial_story if initial_story != "無法生成初始故事" else "無法生成最終故事"

        return initial_story, story_feedback, final_story
    
    except Exception as e:
        logger.error(f"夢境編織過程中發生錯誤: {str(e)}")
        return "故事生成失敗", "反饋生成失敗", "最終故事生成失敗"

def translate_to_english(text):
    """使用Ollama的qwen模型將文本翻譯成英文"""
    try:
        if not text or text.strip() == "":
            return text

        # 簡單檢查是否已經是英文
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        if chinese_chars < len(text) * 0.3:  # 如果中文字符少於30%，可能已經是英文
            return text

        system_prompt = "你是一位翻譯專家。請將用戶輸入的任何語言翻譯成英文，只返回翻譯結果，不要添加任何解釋。請將描述轉換為適合圖像生成的簡潔英文提示詞。"
        user_prompt = f"將以下夢境故事翻譯成簡潔的英文圖像描述: {text}"
        
        translation = ollama_generate(system_prompt, user_prompt, "qwen2.5:14b")
        
        # 如果翻譯失敗或太長，使用簡化版本
        if not translation or len(translation) > 500:
            return "dreamlike floating scene, blue ocean, golden sunlight, surreal clouds, ethereal atmosphere"
        
        return translation
    
    except Exception as e:
        logger.error(f"翻譯過程中發生錯誤: {str(e)}")
        return "dreamlike scene, floating, ocean, sunlight, surreal atmosphere"

def analyze_dream(image_path, video_path, text):
    """使用Ollama分析夢境的心理意義"""
    try:
        system_prompt = """請用台灣習慣的中文回覆。你是一位專業的夢境與心理分析專家，擅長解讀夢境的象徵意義和潛在的心理訊息。
        請根據使用者描述的夢境提供深入的心理分析和建議。請用台灣習慣的中文回覆，避免使用過多心理學專業術語，確保回答通俗易懂。"""
        
        user_prompt = f"""
        以下是使用者描述的夢境：
        
        夢境描述: {text}
        
        請分析這個夢境可能揭示的心理狀態、潛意識願望或恐懼，以及可能的象徵意義。提供心理學觀點的解讀，
        以及對使用者當前生活狀態的可能啟示和建議。分析長度控制在150-200字左右。請使用溫和、支持性的語調。
        """
        
        analysis = ollama_generate(system_prompt, user_prompt, "qwen2.5:14b")
        
        return analysis if analysis else "暫時無法進行心理分析，請稍後再試。"
    
    except Exception as e:
        logger.error(f"夢境分析過程中發生錯誤: {str(e)}")
        return "心理分析功能暫時不可用，但您的夢境描述很有趣，建議您記錄下來以便日後回顧。"

def ollama_generate(system_prompt, user_prompt, model="qwen2.5:14b"):
    """使用Ollama API生成文本"""
    try:
        data = {
            "model": model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 500,
                "stop": ["Human:", "Assistant:", "用戶:", "助手:"]
            }
        }
        
        logger.info(f"發送Ollama請求，模型: {model}")
        response = requests.post(OLLAMA_API, json=data, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("response", "").strip()
            if generated_text:
                logger.info(f"Ollama成功生成文本，長度: {len(generated_text)}")
                return generated_text
            else:
                logger.error("Ollama返回空文本")
                return ""
        else:
            logger.error(f"Ollama API錯誤: {response.status_code}, {response.text}")
            return ""
    except requests.exceptions.Timeout:
        logger.error("Ollama請求超時")
        return ""
    except requests.exceptions.ConnectionError:
        logger.error("無法連接到Ollama服務")
        return ""
    except Exception as e:
        logger.error(f"Ollama請求錯誤: {str(e)}")
        return ""

def create_default_image(filename, prompt_text=""):
    """創建預設圖像"""
    try:
        output_dir = os.path.join(static_dir, 'images')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        # 創建漸變背景圖像
        img = Image.new('RGB', (512, 512), color=(70, 130, 180))
        
        # 創建漸變效果
        for y in range(512):
            for x in range(512):
                # 從中心向外的漸變
                center_x, center_y = 256, 256
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                max_distance = 256 * 1.414  # 對角線長度
                
                # 根據距離調整顏色
                factor = min(distance / max_distance, 1.0)
                r = int(70 + (120 - 70) * factor)
                g = int(130 + (80 - 130) * factor)
                b = int(180 + (200 - 180) * factor)
                
                img.putpixel((x, y), (r, g, b))
        
        # 添加文字
        try:
            draw = ImageDraw.Draw(img)
            
            # 嘗試載入字體
            font_size = 32
            font = None
            
            # 嘗試載入字體
            font_paths = [
                # macOS
                "/System/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/PingFang.ttc",
                # Windows
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/calibri.ttf",
                # Linux
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            ]
            
            for font_path in font_paths:
                try:
                    if os.path.exists(font_path):
                        font = ImageFont.truetype(font_path, font_size)
                        break
                except:
                    continue
            
            if not font:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
            
            if font:
                # 主標題
                title_text = "夢境視覺化"
                bbox = draw.textbbox((0, 0), title_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (512 - text_width) // 2
                y = 200
                
                # 添加文字陰影
                draw.text((x+2, y+2), title_text, fill=(0, 0, 0, 128), font=font)
                draw.text((x, y), title_text, fill=(255, 255, 255), font=font)
                
                # 底部提示
                info_text = "AI 生成中..."
                try:
                    info_font = ImageFont.truetype(font_paths[0], 18) if font_paths else font
                except:
                    info_font = font
                
                if info_font:
                    bbox = draw.textbbox((0, 0), info_text, font=info_font)
                    text_width = bbox[2] - bbox[0]
                    x = (512 - text_width) // 2
                    y = 350
                    
                    draw.text((x+1, y+1), info_text, fill=(0, 0, 0, 64), font=info_font)
                    draw.text((x, y), info_text, fill=(200, 200, 200), font=info_font)
        
        except Exception as text_error:
            logger.info(f"添加文字時發生錯誤: {str(text_error)}")
        
        img.save(output_path, 'PNG')
        logger.info(f"創建預設圖像: {filename}")
        return os.path.join('images', filename)
    
    except Exception as e:
        logger.error(f"創建預設圖像失敗: {str(e)}")
        return "images/default_dream.png"

def save_dream_result(data):
    """保存夢境分析結果以便分享"""
    try:
        # 創建唯一ID
        share_id = str(uuid.uuid4())
        
        # 創建存儲目錄
        share_dir = os.path.join(static_dir, 'shares')
        os.makedirs(share_dir, exist_ok=True)
        
        # 包含時間戳以方便排序
        timestamp = int(time.time())
        
        # 構建保存數據
        share_data = {
            'id': share_id,
            'timestamp': timestamp,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'initialStory': data.get('initialStory', ''),
            'storyFeedback': data.get('storyFeedback', ''),
            'finalStory': data.get('finalStory', ''),
            'translation': data.get('translation', ''),
            'imagePath': data.get('imagePath', ''),
            'videoPath': data.get('videoPath', ''),
            'psychologyAnalysis': data.get('psychologyAnalysis', '')
        }
        
        # 保存到文件
        share_file = os.path.join(share_dir, f"{share_id}.json")
        with open(share_file, 'w', encoding='utf-8') as f:
            json.dump(share_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存分享數據成功: {share_id}")
        return share_id
    except Exception as e:
        logger.error(f"保存分享數據時出錯: {str(e)}")
        return None

# ==================== 記憶體管理 ====================

def clear_model_memory():
    """清理模型記憶體"""
    global image_pipe, video_pipe, models_loaded
    
    import gc
    
    if image_pipe is not None:
        del image_pipe
        image_pipe = None
    
    if video_pipe is not None:
        del video_pipe  
        video_pipe = None
    
    models_loaded = False
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    logger.info("模型記憶體已清理")

# ==================== 路由定義 ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """檢查服務狀態"""
    ollama_status, fooocus_status, local_models_status = check_local_services()
    
    return jsonify({
        'ollama': ollama_status,
        'fooocus': fooocus_status,  # 保持兼容性
        'framepack': local_models_status,  # 現在指向本地模型
        'local_models': local_models_status,
        'timestamp': int(time.time())
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    dream_text = data.get('dream', '')
    
    if not dream_text or len(dream_text.strip()) < 10:
        return jsonify({'error': '請輸入至少10個字的夢境描述'}), 400
    
    if len(dream_text.strip()) > 2000:
        return jsonify({'error': '夢境描述過長，請控制在2000字以內'}), 400
    
    try:
        logger.info(f"開始處理夢境分析請求，輸入長度: {len(dream_text)}")
        
        # 檢查服務狀態
        ollama_status, _, local_models_status = check_local_services()
        if not ollama_status:
            return jsonify({'error': 'Ollama服務不可用，請確認服務是否正常運行在 localhost:11434'}), 503
        
        # 步驟1: 使用dream_weaver處理夢境故事
        logger.info("步驟1: 開始夢境故事生成")
        initial_story, story_feedback, final_story = dream_weaver(dream_text)
        
        # 步驟2: 翻譯故事以便更好地生成圖像（保留備用）
        logger.info("步驟2: 開始翻譯故事")
        translation = translate_to_english(final_story)
        
        # 步驟3: 使用本地快速模型生成圖像
        logger.info("步驟3: 開始生成圖像")
        if local_models_status:
            image_path = generate_image_fast_local(final_story)
        else:
            logger.warning("本地模型不可用，使用預設圖像")
            timestamp = int(time.time())
            image_path = create_default_image(f"default_{timestamp}.png", final_story)
        
        # 步驟4: 使用本地快速模型生成視頻
        logger.info("步驟4: 開始生成視頻")
        video_path = None
        if local_models_status and image_path:
            video_path = generate_video_fast_local(image_path, final_story)
            if video_path:
                logger.info("✅ 視頻生成成功")
            else:
                logger.warning("⚠️ 視頻生成失敗，但不影響其他功能")
        else:
            logger.warning("本地模型不可用，跳過視頻生成")
        
        # 步驟5: 心理分析
        logger.info("步驟5: 開始心理分析")
        psychology_analysis = analyze_dream(image_path, video_path, dream_text)
        
        # 準備響應
        response = {
            'initialStory': initial_story,
            'storyFeedback': story_feedback,
            'finalStory': final_story,
            'translation': translation,
            'imagePath': '/static/' + image_path if image_path else None,
            'videoPath': '/static/' + video_path if video_path else None,
            'psychologyAnalysis': psychology_analysis,
            'apiStatus': {
                'ollama': ollama_status,
                'fooocus': True,  # 保持兼容性
                'framepack': local_models_status,
                'local_models': local_models_status
            },
            'processingInfo': {
                'timestamp': int(time.time()),
                'inputLength': len(dream_text),
                'storyLength': len(final_story) if final_story else 0,
                'useLocalModels': True
            }
        }
        
        logger.info("夢境分析完成，準備返回結果")
        return jsonify(response)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"處理過程中發生錯誤: {str(e)}\n{error_details}")
        return jsonify({
            'error': f'處理過程中發生錯誤: {str(e)}',
            'details': error_details if app.debug else None
        }), 500

@app.route('/api/share', methods=['POST'])
def share_result():
    """創建可分享的夢境分析結果"""
    data = request.json
    
    if not data or 'finalStory' not in data:
        return jsonify({'error': '缺少必要的夢境分析數據'}), 400
    
    try:
        # 保存分享數據
        share_id = save_dream_result(data)
        
        if not share_id:
            return jsonify({'error': '創建分享失敗'}), 500
        
        # 創建分享URL
        share_url = url_for('view_shared', share_id=share_id, _external=True)
        
        return jsonify({
            'shareId': share_id, 
            'shareUrl': share_url,
            'timestamp': int(time.time())
        })
    
    except Exception as e:
        logger.error(f'處理分享請求時發生錯誤: {str(e)}')
        return jsonify({'error': f'處理分享請求時發生錯誤: {str(e)}'}), 500

@app.route('/share/<share_id>')
def view_shared(share_id):
    """查看分享的夢境分析結果"""
    try:
        # 檢查分享ID格式
        if not share_id or not all(c.isalnum() or c == '-' for c in share_id):
            return jsonify({'error': '無效的分享ID'}), 400
        
        # 讀取分享數據
        share_file = os.path.join(static_dir, 'shares', f"{share_id}.json")
        
        if not os.path.exists(share_file):
            return jsonify({'error': '找不到該分享內容'}), 404
        
        with open(share_file, 'r', encoding='utf-8') as f:
            share_data = json.load(f)
        
        # 渲染分享頁面
        try:
            return render_template('shared.html', data=share_data)
        except:
            return render_template('index.html', shared_data=share_data)
    
    except Exception as e:
        logger.error(f'載入分享內容時發生錯誤: {str(e)}')
        return jsonify({'error': f'載入分享內容時發生錯誤'}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': '頁面不存在'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': '服務器內部錯誤'}), 500

# ==================== 主程式入口 ====================

if __name__ == '__main__':
    try:
        # 在程式結束時清理記憶體
        import atexit
        atexit.register(clear_model_memory)
        
        # 確保必要的目錄存在
        directories = [
            os.path.join(static_dir, 'images'),
            os.path.join(static_dir, 'videos'),
            os.path.join(static_dir, 'shares')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"確保目錄存在: {directory}")
        
        # 確保有預設圖像
        default_image_path = os.path.join(static_dir, 'images', 'default_dream.png')
        if not os.path.exists(default_image_path):
            try:
                logger.info("創建預設圖像...")
                create_default_image('default_dream.png', '夢境編織者系統')
                logger.info("預設圖像創建成功")
            except Exception as e:
                logger.error(f"無法創建預設圖像: {str(e)}")
        
        # 檢查服務狀態
        logger.info("檢查服務狀態...")
        ollama_status, _, local_models_status = check_local_services()
        
        # 輸出狀態報告
        print("=" * 80)
        print("夢境編織者系統 - 本地快速生成版本 啟動狀態報告")
        print("=" * 80)
        print(f"Ollama API (localhost:11434): {'✅ 正常' if ollama_status else '❌ 異常'}")
        print(f"本地圖像生成模型: {'✅ 可用' if local_models_status else '❌ 不可用'}")
        print(f"本地視頻生成模型: {'✅ 可用' if local_models_status else '❌ 不可用'}")
        print(f"靜態檔案目錄: {static_dir}")
        
        # 檢查 PyTorch 和設備支持
        if torch.backends.mps.is_available():
            print("✅ 已啟用 Metal Performance Shaders (MPS) 加速")
            device_info = "MPS (Apple Silicon 優化)"
        elif torch.cuda.is_available():
            print("✅ 已啟用 CUDA 加速")
            device_info = "CUDA"
        else:
            print("⚠️  使用 CPU 模式，速度可能較慢")
            device_info = "CPU"
        
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"使用設備: {device_info}")
        print("=" * 80)
        
        # 詳細的狀態說明和建議
        if not ollama_status:
            print("❌ 警告: Ollama API 無法連接")
            print("   請確認 Ollama 服務是否運行在 localhost:11434")
            print("   啟動命令: ollama serve")
            print()
        
        if not local_models_status:
            print("❌ 警告: 本地生成模型不可用")
            print("   首次運行時會自動下載模型（約 4-6GB）")
            print("   請確保網路連接正常且有足夠的儲存空間")
            print("   模型下載完成後即可離線使用")
            print()
        
        # 系統功能說明
        print("🔧 系統功能狀態:")
        print(f"   • 故事生成: {'✅ 可用 (Ollama)' if ollama_status else '❌ 不可用'}")
        print(f"   • 文本翻譯: {'✅ 可用 (Ollama)' if ollama_status else '❌ 不可用'}")
        print(f"   • 圖像生成: {'✅ 可用 (Stable Diffusion v1.5)' if local_models_status else '⚠️  預設圖像'}")
        print(f"   • 視頻生成: {'✅ 可用 (Stable Video Diffusion)' if local_models_status else '❌ 不可用'}")
        print(f"   • 心理分析: {'✅ 可用 (Ollama)' if ollama_status else '❌ 不可用'}")
        print(f"   • 故事轉提示詞: {'✅ 可用 (智能轉換)' if ollama_status else '⚠️  關鍵詞提取'}")
        print()
        
        # 特殊說明
        print("🚀 本地快速生成特性:")
        if local_models_status:
            print("   • 圖像生成: 10-25 秒（512x512）")
            print("   • 視頻生成: 30 秒-2 分鐘（0.6 秒@8fps）")
            print("   • 智能故事轉圖像提示詞")
            print("   • 完全離線運行（首次下載後）")
            print("   • 自動設備優化（MPS/CUDA/CPU）")
        print("   • 所有功能通過統一界面使用")
        print("   • 完整的錯誤處理和日誌記錄")
        print("   • 自動記憶體管理")
        print()
        
        # 性能預期
        print("⚡ 性能預期:")
        if device_info == "MPS (Apple Silicon 優化)":
            print("   • M1/M2/M3/M4 優化，速度較快")
            print("   • 記憶體使用: 4-8GB")
        elif device_info == "CUDA":
            print("   • GPU 加速，速度最快")
            print("   • 記憶體使用: 4-6GB VRAM")
        else:
            print("   • CPU 模式，速度較慢但功能完整")
            print("   • 記憶體使用: 6-12GB RAM")
        print()
        
        # 最低運行要求
        if ollama_status:
            print("✅ 系統可以基本運行（至少需要 Ollama）")
        else:
            print("❌ 系統無法正常運行，需要至少安裝 Ollama")
        
        print("=" * 80)
        print("系統準備就緒，啟動 Flask 應用程式...")
        print("訪問地址: http://localhost:5002")
        print("=" * 80)
        
        # 啟動Flask應用
        app.run(debug=True, host='0.0.0.0', port=5002, threaded=True)
        
    except KeyboardInterrupt:
        logger.info("用戶中斷，正在關閉系統...")
        clear_model_memory()
    except Exception as e:
        logger.error(f"系統啟動失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        clear_model_memory()