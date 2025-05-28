import os
import time
import uuid
import json
from flask import Flask, request, jsonify, render_template, url_for
import requests
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import gc
import random

app_root = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(app_root, 'static')

app = Flask(__name__, 
            template_folder=os.path.join(app_root, 'templates'),
            static_folder=static_dir)

OLLAMA_API = "http://localhost:11434/api/generate"

# 全局變數
image_pipe = None
models_loaded = False
current_device = None
torch_dtype = None

# 定義可用的模型清單
models = {
    "stable-diffusion-v1-5": "runwayml/stable-diffusion-v1-5",
}

def translate_to_english(text):
    """使用本地 Ollama qwen2.5:14b 模型將任何語言翻譯成英文"""
    # 如果文本為空，直接返回
    if not text or text.strip() == "":
        return text

    # 檢查是否是英文
    if all('\u4e00' > char or char > '\u9fff' for char in text):
        return text

    try:
        system_prompt = "你是一位英文翻譯專家。請將用戶輸入的任何語言翻譯成英文，只返回翻譯成英文的結果，不要包含任何額外說明。"
        user_prompt = f"將以下文本翻譯成英文: {text}"
        
        translation = ollama_generate(system_prompt, user_prompt)
        
        if translation:
            print(f"已翻譯: {text} -> {translation}")
            return translation.strip()
        else:
            return text

    except Exception as e:
        print(f"翻譯錯誤: {str(e)}")
        return text

def initialize_local_models():
    global image_pipe, models_loaded, current_device, torch_dtype
    
    if models_loaded:
        return True
    
    try:
        if torch.cuda.is_available():
            current_device = "cuda"
            torch_dtype = torch.float16
        elif torch.backends.mps.is_available():
            current_device = "mps"
            torch_dtype = torch.float32
        else:
            current_device = "cpu"
            torch_dtype = torch.float32
        
        model_id = "runwayml/stable-diffusion-v1-5"
        
        # 使用類似右邊代碼的載入方式
        image_pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
        ).to(current_device)
        
        image_pipe.scheduler = UniPCMultistepScheduler.from_config(image_pipe.scheduler.config)
        
        # 優化設定
        if current_device == "cuda":
            image_pipe.enable_model_cpu_offload()
            image_pipe.enable_attention_slicing()
            image_pipe.enable_vae_slicing()
        elif current_device == "mps":
            image_pipe.enable_attention_slicing(1)
        else:
            image_pipe.enable_attention_slicing()
        
        models_loaded = True
        return True
        
    except Exception as e:
        print(f"模型初始化錯誤: {e}")
        return False

def load_model(model_key):
    """加載不同模型的函數"""
    global image_pipe, models_loaded
    
    try:
        model_name = models.get(model_key, "runwayml/stable-diffusion-v1-5")
        
        # 清理舊模型
        if image_pipe is not None:
            del image_pipe
            torch.cuda.empty_cache()
            gc.collect()
        
        # 載入新模型
        image_pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
        ).to(current_device)
        
        image_pipe.scheduler = UniPCMultistepScheduler.from_config(image_pipe.scheduler.config)
        
        models_loaded = True
        return True
        
    except Exception as e:
        print(f"模型載入錯誤: {e}")
        return False

def story_to_image_prompt(dream_text):
    """將故事轉換為圖像提示詞並翻譯成英文"""
    system_prompt = """You are a Stable Diffusion prompt expert. Convert the user's story into English image generation prompts. Requirements:
1. Extract core visual elements, especially characters and animals
2. If the story mentions "我" (I/me),"他"(he),"她"(she), include "person" or "human" in the prompt
3. Use English keywords only
4. Limit to 40 English words
5. Return only the English prompt, ensure all important elements are included"""

    user_prompt = f"Story: {dream_text}\nConvert to image prompt:"
    
    converted_prompt = ollama_generate(system_prompt, user_prompt)
    
    if not converted_prompt:
        return None
    
    # 直接使用生成的英文提示詞，不再翻譯
    final_prompt = converted_prompt.strip()
    print(f"✨ 直接生成英文提示詞: {final_prompt}")
    
    if not final_prompt:
        return None
    
    # 檢查是否包含人物元素，如果故事中有"我"但提示詞中沒有人物，則添加
    if ('我' in dream_text or '自己' in dream_text) and 'person' not in final_prompt.lower() and 'human' not in final_prompt.lower() and 'figure' not in final_prompt.lower():
        final_prompt = f" {final_prompt}"
    
    base_prompt = final_prompt
    if len(base_prompt) > 80:
        base_prompt = base_prompt[:80].rsplit(',', 1)[0]
    
    return f"{base_prompt}, vibrant colors, detailed, cinematic"

def extract_visual_keywords(text):
    keyword_map = {
        '飛行': 'flying', '漂浮': 'floating', '海洋': 'ocean', '天空': 'sky', 
        '城堡': 'castle', '森林': 'forest', '夢境': 'dreamlike', '美麗': 'beautiful',
        '藍色': 'blue', '金色': 'golden', '明亮': 'bright', '神秘': 'mysterious'
    }
    
    keywords = []
    for chinese, english in keyword_map.items():
        if chinese in text:
            keywords.append(english)
    
    if not keywords:
        keywords = ['dreamlike scene', 'fantasy', 'beautiful']
    
    return ', '.join(keywords[:5]) + ', detailed'

def generate_image_fast_local(story_text, model_name="stable-diffusion-v1-5"):
    """使用改進的圖像生成邏輯"""
    if not initialize_local_models() or image_pipe is None:
        return None
    
    try:
        # 生成並翻譯提示詞
        image_prompt = story_to_image_prompt(story_text)
        
        if not image_prompt:
            return None
        
        # 根據故事內容調整負面提示詞
        # 如果故事包含人物相關詞彙，則不排除人物
        story_lower = story_text.lower()
        person_keywords = ['我', '人', '自己', '夢見我']
        
        if any(keyword in story_text for keyword in person_keywords):
            # 包含人物的負面提示詞
            negative_prompt = "ugly, blurry, distorted, deformed, low quality, bad anatomy, multiple heads, extra limbs"
        else:
            # 不包含人物的負面提示詞
            negative_prompt = "human, person, ugly, blurry, distorted, deformed, low quality, bad anatomy"
        
        # 使用隨機種子
        seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(current_device).manual_seed(seed)
        
        # 設定生成參數
        generation_params = {
            "prompt": image_prompt,
            "negative_prompt": negative_prompt,
            "height": 512,
            "width": 512,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "generator": generator
        }
        
        # 根據設備調整參數
        if current_device == "cpu":
            generation_params["num_inference_steps"] = 15
        elif current_device == "cuda":
            generation_params["num_inference_steps"] = 25
        
        # 清理記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # 生成圖像
        with torch.no_grad():
            result = image_pipe(**generation_params)
            
            if not result.images or len(result.images) == 0:
                return None
            
            generated_image = result.images[0]
        
        # 確保圖像格式正確
        if generated_image.mode != 'RGB':
            generated_image = generated_image.convert('RGB')
        
        # 檢查並調整亮度
        img_array = np.array(generated_image)
        avg_brightness = np.mean(img_array)
        
        if avg_brightness < 30:
            img_array = np.clip(img_array * 1.3 + 20, 0, 255).astype(np.uint8)
            generated_image = Image.fromarray(img_array)
        
        # 保存圖像
        timestamp = int(time.time())
        random_id = str(uuid.uuid4())[:8]
        output_filename = f"dream_{timestamp}_{random_id}.png"
        
        output_dir = os.path.join(static_dir, 'images')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        
        generated_image.save(output_path, format='PNG', quality=95)
        
        return os.path.join('images', output_filename)
        
    except Exception as e:
        print(f"圖像生成失敗: {e}")
        return None

def dream_weaver(prompt):
    system_writer = """你是夢境故事創作專家，要求：
1. 直接開始故事，無問候語
2. 融合夢境元素
3. 使用第一人稱
4. 150-200字完整故事
5. 繁體中文"""

    story_prompt = f"基於夢境片段創作故事：「{prompt}」"
    
    final_story = ollama_generate(system_writer, story_prompt)
    
    if not final_story:
        return "無法生成夢境故事"
    
    return clean_story_content(final_story)

def clean_story_content(story):
    unwanted_phrases = [
        "好的，根據您的建議", "根據您的要求", "以下是故事", "故事如下",
        "###", "**", "故事名稱：", "夢境故事：", "完整故事："
    ]
    
    cleaned_story = story.strip()
    
    for phrase in unwanted_phrases:
        if cleaned_story.startswith(phrase):
            cleaned_story = cleaned_story[len(phrase):].strip()
        if phrase in cleaned_story:
            parts = cleaned_story.split(phrase)
            if len(parts) > 1:
                cleaned_story = parts[-1].strip()
    
    if cleaned_story.startswith('"') and cleaned_story.endswith('"'):
        cleaned_story = cleaned_story[1:-1].strip()
    if cleaned_story.startswith('「') and cleaned_story.endswith('」'):
        cleaned_story = cleaned_story[1:-1].strip()
    
    cleaned_story = cleaned_story.replace('*', '').replace('#', '').strip()
    
    return cleaned_story

def analyze_dream(text):
    system_prompt = """你是夢境心理分析專家，分析夢境的象徵意義和心理狀態，
提供150-200字的分析，使用溫和支持性語調，繁體中文回答。"""
    
    user_prompt = f"夢境描述: {text}\n請提供心理分析："
    
    analysis = ollama_generate(system_prompt, user_prompt)
    
    return analysis if analysis else "暫時無法進行心理分析。"

def ollama_generate(system_prompt, user_prompt):
    try:
        data = {
            "model": "qwen2.5:14b",
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 500
            }
        }
        
        response = requests.post(OLLAMA_API, json=data, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            return ""
    except Exception:
        return ""

def save_dream_result(data):
    try:
        share_id = str(uuid.uuid4())
        share_dir = os.path.join(static_dir, 'shares')
        os.makedirs(share_dir, exist_ok=True)
        
        share_data = {
            'id': share_id,
            'timestamp': int(time.time()),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'finalStory': data.get('finalStory', ''),
            'imagePath': data.get('imagePath', ''),
            'psychologyAnalysis': data.get('psychologyAnalysis', '')
        }
        
        share_file = os.path.join(share_dir, f"{share_id}.json")
        with open(share_file, 'w', encoding='utf-8') as f:
            json.dump(share_data, f, ensure_ascii=False, indent=2)
        
        return share_id
    except Exception:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    try:
        ollama_response = requests.get("http://localhost:11434/api/tags", timeout=5)
        ollama_status = ollama_response.status_code == 200
    except:
        ollama_status = False
    
    local_models_status = models_loaded or initialize_local_models()
    
    return jsonify({
        'ollama': ollama_status,
        'local_models': local_models_status,
        'device': current_device,
        'available_models': list(models.keys()),
        'timestamp': int(time.time())
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    dream_text = data.get('dream', '')
    selected_model = data.get('model', 'stable-diffusion-v1-5')
    
    if not dream_text or len(dream_text.strip()) < 10:
        return jsonify({'error': '請輸入至少10個字的夢境描述'}), 400
    
    if len(dream_text.strip()) > 2000:
        return jsonify({'error': '夢境描述過長，請控制在2000字以內'}), 400
    
    try:
        # 檢查 Ollama 服務
        try:
            ollama_response = requests.get("http://localhost:11434/api/tags", timeout=5)
            ollama_status = ollama_response.status_code == 200
        except:
            ollama_status = False
        
        if not ollama_status:
            return jsonify({'error': 'Ollama服務不可用'}), 503
        
        # 生成故事
        final_story = dream_weaver(dream_text)
        
        # 載入指定模型並生成圖像
        local_models_status = models_loaded or initialize_local_models()
        if local_models_status and selected_model in models:
            load_model(selected_model)
        
        image_path = None
        if local_models_status:
            image_path = generate_image_fast_local(dream_text, selected_model)
        
        # 心理分析
        psychology_analysis = analyze_dream(dream_text)
        
        response = {
            'finalStory': final_story,
            'imagePath': '/static/' + image_path if image_path else None,
            'psychologyAnalysis': psychology_analysis,
            'apiStatus': {
                'ollama': ollama_status,
                'local_models': local_models_status,
                'device': current_device,
                'current_model': selected_model
            },
            'processingInfo': {
                'timestamp': int(time.time()),
                'inputLength': len(dream_text),
                'storyLength': len(final_story) if final_story else 0
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"分析錯誤: {e}")
        return jsonify({'error': '處理過程中發生錯誤'}), 500

@app.route('/api/share', methods=['POST'])
def share_result():
    data = request.json
    
    if not data or 'finalStory' not in data:
        return jsonify({'error': '缺少必要的夢境分析數據'}), 400
    
    try:
        share_id = save_dream_result(data)
        
        if not share_id:
            return jsonify({'error': '創建分享失敗'}), 500
        
        share_url = url_for('view_shared', share_id=share_id, _external=True)
        
        return jsonify({
            'shareId': share_id, 
            'shareUrl': share_url,
            'timestamp': int(time.time())
        })
    
    except Exception:
        return jsonify({'error': '處理分享請求時發生錯誤'}), 500

@app.route('/share/<share_id>')
def view_shared(share_id):
    try:
        share_file = os.path.join(static_dir, 'shares', f"{share_id}.json")
        
        if not os.path.exists(share_file):
            return jsonify({'error': '找不到該分享內容'}), 404
        
        with open(share_file, 'r', encoding='utf-8') as f:
            share_data = json.load(f)
        
        return render_template('shared.html', data=share_data)
    
    except Exception:
        return jsonify({'error': '載入分享內容時發生錯誤'}), 500

if __name__ == '__main__':
    directories = [
        os.path.join(static_dir, 'images'),
        os.path.join(static_dir, 'shares')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    app.run(debug=False, host='0.0.0.0', port=5002, threaded=True)