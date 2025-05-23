import os
import time
import subprocess
import tempfile
import uuid
import json
from flask import Flask, request, jsonify, render_template
import requests
import base64
from PIL import Image
from io import BytesIO


app_root = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(app_root, 'static')

app = Flask(__name__, 
            template_folder=os.path.join(app_root, 'templates'),
            static_folder=static_dir)

# 設定Ollama API端點
OLLAMA_API = "http://localhost:11434/api/generate"
# 設定Fooocus API端點
FOOOCUS_API = "http://localhost:8888/v1/generation/text-to-image"
# 設定FramePack路徑 (請替換為您的FramePack實際路徑)
FRAMEPACK_PATH = "path/to/framepack"

def dream_weaver(prompt):
    """使用Ollama的qwen模型處理夢境故事生成，返回初始故事、反饋和最終故事"""
    
    # 系統提示詞
    system_planner = """請用台灣習慣的中文回覆。你是一個專業的繁體中文故事大綱規劃專家，特別擅長分析夢境元素並創建連貫的故事架構。
    當用戶提供零散的夢境片段時，你的任務是分析每個元素的象徵意義，產生多種可能的故事架構，
    並根據元素之間的潛在聯繫選擇最佳架構。請展示你的完整思考過程。請用台灣習慣的中文回覆。"""

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

    # 故事創作
    story_prompt = f"""
    基於以下思考過程和大綱：

    {story_outline}

    請創作一個完整的夢境故事，融合所有提供的夢境元素：「{prompt}」。
    故事應展現夢境的超現實性和流動感，長度50-100字，使用繁體中文回答。
    """

    initial_story = ollama_generate(system_writer, story_prompt, "qwen2.5:14b")

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

    # 根據反思優化故事
    final_prompt = f"""
    請根據以下反饋意見改進故事：

    {story_feedback}

    原始故事：
    {initial_story}

    請使用繁體中文創作更加完善的最終版本，確保融合所有夢境元素並增強其超現實性，全部都要使用繁體中文回答。
    """
    
    final_story = ollama_generate(system_writer, final_prompt, "qwen2.5:14b")

    return initial_story, story_feedback, final_story

def translate_to_english(text):
    """使用Ollama的qwen模型將文本翻譯成英文"""
    
    if not text or text.strip() == "":
        return text

    # 檢查是否是英文
    if all('\u4e00' > char or char > '\u9fff' for char in text):
        return text

    system_prompt = "你是一位翻譯專家。請將用戶輸入的任何語言翻譯成英文，只返回翻譯結果。"
    user_prompt = f"將以下文本翻譯成英文: {text}"
    
    translation = ollama_generate(system_prompt, user_prompt, "qwen2.5:14b")
    
    return translation

def generate_image(prompt):
    """使用Fooocus API生成圖像"""
    
    try:
        # 創建輸出目錄
        output_dir = os.path.join(static_dir, 'images')
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成唯一的檔案名
        timestamp = int(time.time())
        output_filename = f"dream_{timestamp}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # 增強提示詞以獲得更好的夢境效果
        enhanced_prompt = f"{prompt}, dreamlike, ethereal, surreal, fantasy landscape, intricate details, magical atmosphere, dream visualization"
        
        # 準備Fooocus API請求數據
        payload = {
            "prompt": enhanced_prompt,
            "negative_prompt": "blurry, low quality, deformed, distorted face, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, out of focus, long neck, long body",
            "style_name": "dreamlike",  # 適合夢境的風格
            "performance_selection": "speed",
            "aspect_ratios_selection": "1024×1024",
            "image_number": 1,
            "image_quality": 95,
            "sampler_name": "dpmpp_2m_sde_gpu",
            "scheduler_name": "karras",
            "steps": 30,
            "cfg": 7.0,
            "seed": -1  # 隨機種子
        }
        
        # 發送POST請求到Fooocus API
        print(f"發送請求到Fooocus API，提示詞：{enhanced_prompt[:50]}...")
        response = requests.post(FOOOCUS_API, json=payload, timeout=300)  # 設置較長的超時時間
        
        if response.status_code == 200:
            result = response.json()
            if "images" in result and len(result["images"]) > 0:
                # 從Base64解碼並保存圖片
                image_data = base64.b64decode(result["images"][0])
                with open(output_path, "wb") as f:
                    f.write(image_data)
                print(f"Fooocus API成功生成圖像: {output_filename}")
                return os.path.join('images', output_filename)
            else:
                print(f"Fooocus API返回無效結果: {result}")
        else:
            print(f"Fooocus API錯誤: {response.status_code}, {response.text}")
        
        # 如果API調用失敗，返回預設圖像
        return "images/default_dream.png"
    
    except Exception as e:
        print(f"生成圖像時出錯: {str(e)}")
        # 如果出錯，返回預設圖片
        return "images/default_dream.png"

def generate_video(image_path, prompt):
    """使用FramePack生成視頻"""
    
    try:
        # 創建輸出目錄
        output_dir = os.path.join(static_dir, 'videos')
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成唯一的檔案名
        timestamp = int(time.time())
        output_filename = f"dream_video_{timestamp}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # 完整的圖片路徑
        full_image_path = os.path.join(static_dir, image_path)
        
        # 構建命令調用FramePack (請根據實際FramePack的命令行參數調整)
        cmd = [
            "python", f"{FRAMEPACK_PATH}/run.py",
            "--input", full_image_path,
            "--prompt", prompt,
            "--output", output_path,
            "--duration", "15",  # 15秒的視頻
            "--fps", "30"
        ]
        
        # 執行命令 (實際部署時取消註釋)
        # subprocess.run(cmd, check=True)
        print(f"將執行FramePack命令生成視頻: {output_filename}")
        
        # 在實際應用中，FramePack會生成視頻到指定路徑
        # 為了演示，這裡假設視頻已經生成
        
        return os.path.join('videos', output_filename)
    
    except Exception as e:
        print(f"生成視頻時出錯: {str(e)}")
        return None

def analyze_dream(image_path, video_path, text):
    """使用Ollama的qwen模型分析夢境的心理意義"""
    
    system_prompt = """請用台灣習慣的中文回覆。你是一位專業的夢境與心理分析專家，擅長解讀夢境的象徵意義和潛在的心理訊息。
    請根據使用者描述的夢境、生成的圖像和視頻，提供深入的心理分析和建議。請用台灣習慣的中文回覆，避免使用心理學專業術語過多，確保回答通俗易懂。"""
    
    user_prompt = f"""
    以下是使用者描述的夢境以及根據夢境生成的圖像和視頻：
    
    夢境描述: {text}
    
    請分析這個夢境可能揭示的心理狀態、潛意識願望或恐懼，以及可能的象徵意義。提供心理學觀點的解讀，
    以及對使用者當前生活狀態的可能啟示和建議。分析長度控制在200字左右。
    """
    
    analysis = ollama_generate(system_prompt, user_prompt, "qwen2.5:14b")
    
    return analysis

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
        
        return share_id
    except Exception as e:
        print(f"保存分享數據時出錯: {str(e)}")
        return None

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
        
        return jsonify({'shareId': share_id, 'shareUrl': share_url})
    
    except Exception as e:
        return jsonify({'error': f'處理分享請求時發生錯誤: {str(e)}'}), 500

@app.route('/share/<share_id>')
def view_shared(share_id):
    """查看分享的夢境分析結果"""
    try:
        # 檢查分享ID格式
        if not share_id or not all(c.isalnum() or c == '-' for c in share_id):
            return render_template('error.html', message='無效的分享ID'), 400
        
        # 讀取分享數據
        share_file = os.path.join(static_dir, 'shares', f"{share_id}.json")
        
        if not os.path.exists(share_file):
            return render_template('error.html', message='找不到該分享內容'), 404
        
        with open(share_file, 'r', encoding='utf-8') as f:
            share_data = json.load(f)
        
        # 渲染分享頁面
        return render_template('shared.html', data=share_data)
    
    except Exception as e:
        return render_template('error.html', message=f'載入分享內容時發生錯誤: {str(e)}'), 500

def ollama_generate(system_prompt, user_prompt, model="qwen2.5:14b"):
    """使用Ollama API生成文本"""
    
    data = {
        "model": model,
        "prompt": user_prompt,
        "system": system_prompt,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_API, json=data)
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "")
        else:
            print(f"Ollama API錯誤: {response.status_code}")
            return "生成失敗，請稍後再試。"
    except Exception as e:
        print(f"Ollama請求錯誤: {str(e)}")
        return "連接失敗，請確認Ollama服務運行正常。"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    dream_text = data.get('dream', '')
    
    if not dream_text or len(dream_text.strip()) < 10:
        return jsonify({'error': '請輸入至少10個字的夢境描述'}), 400
    
    try:
        # 步驟1: 使用dream_weaver處理夢境故事
        initial_story, story_feedback, final_story = dream_weaver(dream_text)
        
        # 步驟2: 翻譯故事以便更好地生成圖像
        translation = translate_to_english(final_story)
        
        # 步驟3: 使用翻譯生成圖像
        image_path = generate_image(translation)
        
        # 步驟4: 使用圖像和翻譯生成視頻
        video_path = generate_video(image_path, translation)
        
        # 步驟5: 心理分析
        psychology_analysis = analyze_dream(image_path, video_path, dream_text)
        
        # 準備響應
        response = {
            'initialStory': initial_story,
            'storyFeedback': story_feedback,
            'finalStory': final_story,
            'translation': translation,
            'imagePath': '/static/' + image_path if image_path else None,
            'videoPath': '/static/' + video_path if video_path else None,
            'psychologyAnalysis': psychology_analysis
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"處理過程中發生錯誤: {str(e)}\n{error_details}")
        return jsonify({'error': f'處理過程中發生錯誤: {str(e)}'}), 500

if __name__ == '__main__':
    # 確保圖像和視頻目錄存在
    os.makedirs('static/images', exist_ok=True)
    os.makedirs('static/videos', exist_ok=True)
    
    # 確保有預設圖像
    default_image_path = os.path.join(static_dir, 'images', 'default_dream.png')
    if not os.path.exists(default_image_path):
        # 創建一個簡單的預設圖像
        try:
            img = Image.new('RGB', (800, 600), color=(73, 109, 137))
            img.save(default_image_path)
        except Exception as e:
            print(f"無法創建預設圖像: {str(e)}")
    
    app.run(debug=True, host='0.0.0.0', port=5002)