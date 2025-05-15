import os
import subprocess
import tempfile
import json
from flask import Flask, request, jsonify, render_template
import requests
from PIL import Image
from io import BytesIO


app_root = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(app_root, 'static')

app = Flask(__name__, 
            template_folder=os.path.join(app_root, 'templates'),
            static_folder=static_dir)

# 設定Ollama API端點
OLLAMA_API = "http://localhost:11434/api/generate"

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
    """使用Fooocus生成圖像"""
    
    # 這裡我們模擬與Fooocus的互動
    # 在真實環境中，你需要根據Fooocus的API或命令行接口進行調整
    
    try:
        # 假設Fooocus提供了REST API或命令行接口
        # 這裡使用臨時文件來存儲生成的圖像
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_file = tmp.name
        
        # 構建命令調用Fooocus (這是模擬的命令，實際命令可能不同)
        cmd = [
            "python", "path/to/fooocus/run.py",
            "--prompt", prompt,
            "--output", output_file,
            "--sampler", "dpmpp_2m",
            "--steps", "30"
        ]
        
        # 執行命令
        # subprocess.run(cmd, check=True)
        
        # 讀取生成的圖像
        # image = Image.open(output_file)
        
        # 在實際應用中，你會返回這個圖像
        # 但在這個示例中，我們只返回一個假的路徑
        return output_file
    
    except Exception as e:
        print(f"生成圖像時出錯: {str(e)}")
        return None

def analyze_dream(image_path, text):
    """使用Ollama的qwen模型分析夢境的心理意義"""
    
    system_prompt = """請用台灣習慣的中文回覆。你是一位專業的夢境與心理分析專家，擅長解讀夢境的象徵意義和潛在的心理訊息。
    請根據使用者描述的夢境和相關圖像，提供深入的心理分析和建議。請用台灣習慣的中文回覆，避免使用心理學專業術語過多，確保回答通俗易懂。"""
    
    user_prompt = f"""
    以下是使用者描述的夢境以及根據夢境生成的圖像：
    
    夢境描述: {text}
    
    請分析這個夢境可能揭示的心理狀態、潛意識願望或恐懼，以及可能的象徵意義。提供心理學觀點的解讀，
    以及對使用者當前生活狀態的可能啟示和建議。分析長度控制在200字左右。
    """
    
    analysis = ollama_generate(system_prompt, user_prompt, "qwen2.5:14b")
    
    return analysis

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
            print(f"API錯誤: {response.status_code}")
            return "生成失敗，請稍後再試。"
    except Exception as e:
        print(f"請求錯誤: {str(e)}")
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
        
        # 步驟2: 翻譯故事
        translation = translate_to_english(final_story)
        
        # 步驟3: 使用翻譯生成圖像
        image_path = generate_image(translation)
        
        # 步驟4: 心理分析
        psychology_analysis = analyze_dream(image_path, dream_text)
        
        # 準備響應
        response = {
            'initialStory': initial_story,
            'storyFeedback': story_feedback,
            'finalStory': final_story,
            'translation': translation,
            'imagePath': '/images/' + os.path.basename(image_path) if image_path else None,
            'psychologyAnalysis': psychology_analysis
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': f'處理過程中發生錯誤: {str(e)}'}), 500

if __name__ == '__main__':
    # 確保圖像目錄存在
    os.makedirs('static/images', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5002)