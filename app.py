import os
import time
import uuid
import json
import random
from flask import Flask, request, jsonify, render_template, url_for
import requests
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import gc

# 導入 FramePack 影片生成器
from framepack_video_generator import generate_dream_video, get_video_generator

class DreamAnalyzer:
    def __init__(self):
        self.app_root = os.path.dirname(os.path.abspath(__file__))
        self.static_dir = os.path.join(self.app_root, 'static')
        self.app = Flask(__name__, 
                        template_folder=os.path.join(self.app_root, 'templates'),
                        static_folder=self.static_dir)
        
        # 配置
        self.OLLAMA_API = "http://localhost:11434/api/generate"
        self.OLLAMA_MODEL = "qwen2.5:14b"
        
        # 模型狀態
        self.image_pipe = None
        self.models_loaded = False
        self.current_device = None
        self.torch_dtype = None
        
        # FramePack 影片生成器
        self.video_generator = get_video_generator(self.static_dir)
        
        # 可用模型
        self.models = {
            "stable-diffusion-v1-5": "runwayml/stable-diffusion-v1-5",
        }
        
        # 防重複提交
        self.processing_requests = set()  # 正在處理的請求ID
        self.request_lock = False  # 全局鎖
        
        self._setup_routes()
        self._create_directories()

    def _create_directories(self):
        """創建必要的目錄"""
        directories = [
            os.path.join(self.static_dir, 'images'),
            os.path.join(self.static_dir, 'videos'),  # 新增影片目錄
            os.path.join(self.static_dir, 'shares')
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def _setup_routes(self):
        """設定路由"""
        self.app.route('/')(self.index)
        self.app.route('/api/status')(self.api_status)
        self.app.route('/api/analyze', methods=['POST'])(self.analyze)
        self.app.route('/api/share', methods=['POST'])(self.share_result)
        self.app.route('/share/<share_id>')(self.view_shared)

    def _initialize_device(self):
        """初始化設備設定"""
        if torch.cuda.is_available():
            self.current_device = "cuda"
            self.torch_dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.current_device = "mps"
            self.torch_dtype = torch.float32
        else:
            self.current_device = "cpu"
            self.torch_dtype = torch.float32

    def _load_image_model(self):
        """載入圖像生成模型"""
        if self.models_loaded:
            return True
        
        try:
            self._initialize_device()
            model_id = "runwayml/stable-diffusion-v1-5"
            
            self.image_pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.current_device)
            
            self.image_pipe.scheduler = UniPCMultistepScheduler.from_config(
                self.image_pipe.scheduler.config
            )
            
            # 優化設定
            if self.current_device == "cuda":
                self.image_pipe.enable_model_cpu_offload()
                self.image_pipe.enable_attention_slicing()
                self.image_pipe.enable_vae_slicing()
            elif self.current_device == "mps":
                self.image_pipe.enable_attention_slicing(1)
            else:
                self.image_pipe.enable_attention_slicing()
            
            self.models_loaded = True
            print(f"✅ 模型載入成功，使用設備: {self.current_device}")
            return True
            
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            return False

    def _check_ollama_status(self):
        """檢查 Ollama 服務狀態"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def _check_framepack_status(self):
        """檢查 FramePack 影片生成可用性"""
        try:
            return self.video_generator.is_available()
        except:
            return False

    def _call_ollama(self, system_prompt, user_prompt, temperature=0.7):
        """調用 Ollama API"""
        try:
            data = {
                "model": self.OLLAMA_MODEL,
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_predict": 500
                }
            }
            
            response = requests.post(self.OLLAMA_API, json=data, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            return ""
        except Exception as e:
            print(f"❌ Ollama 調用失敗: {e}")
            return ""

    def _generate_story(self, dream_text):
        """生成夢境故事"""
        system_prompt = """你是夢境故事創作專家，要求：
1. 直接開始故事，無問候語
2. 融合夢境元素
3. 使用第一人稱
4. 150-200字完整故事
5. 繁體中文"""

        user_prompt = f"基於夢境片段創作故事：「{dream_text}」"
        
        story = self._call_ollama(system_prompt, user_prompt)
        return self._clean_story_content(story) if story else "無法生成夢境故事"

    def _clean_story_content(self, story):
        """清理故事內容"""
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
        
        # 移除引號
        if cleaned_story.startswith('"') and cleaned_story.endswith('"'):
            cleaned_story = cleaned_story[1:-1].strip()
        if cleaned_story.startswith('「') and cleaned_story.endswith('」'):
            cleaned_story = cleaned_story[1:-1].strip()
        
        return cleaned_story.replace('*', '').replace('#', '').strip()

    def _generate_image_prompt(self, dream_text):
        """將夢境轉換為圖像生成提示詞"""
        system_prompt = """You are a Stable Diffusion prompt expert. Convert user input into English image generation prompts. Requirements:
        1. PRESERVE and translate ALL original elements from the input
        2. If input is a story/emotion, extract the MOST VISUAL and EMOTIONAL scene
        3. ENHANCE with additional quality details, don't replace original content
        4. Use English keywords only
        5. The main characters are mostly Asian
        6. Focus on the most dramatic or emotional moment in the story
        7. Include facial expressions and emotions from the original text

        Example:
        Input: "他常夢見前任結婚了，每次都笑著祝福，然後哭著醒來"
        Better output: "Asian man dreaming, wedding scene, forced smile, tears, emotional pain, bittersweet expression, dream-like atmosphere, cinematic lighting, detailed"

        Always capture the EMOTIONAL CORE and most visual elements."""
        user_prompt = f"Story: {dream_text}\nConvert to image prompt:"
        
        raw_prompt = self._call_ollama(system_prompt, user_prompt, temperature=0.5)
        
        if not raw_prompt:
            return None
        
        # 清理並處理提示詞
        clean_prompt = raw_prompt.strip()
        
        # 檢查是否包含人物元素
        if ('我' in dream_text or '自己' in dream_text) and \
           not any(word in clean_prompt.lower() for word in ['person', 'human', 'figure']):
            clean_prompt = f" I {clean_prompt}"
        
        final_prompt = f"{clean_prompt}, vibrant colors, detailed, cinematic"
        
        print(f"✨ 生成圖像提示詞: {final_prompt}")
        return final_prompt

    def _generate_image(self, dream_text):
        """生成圖像"""
        if not self._load_image_model() or self.image_pipe is None:
            print("❌ 圖像模型未載入")
            return None
        
        try:
            # 生成提示詞
            image_prompt = self._generate_image_prompt(dream_text)
            if not image_prompt:
                print("❌ 無法生成圖像提示詞")
                return None
            
            # 設定負面提示詞
            person_keywords = ['我', '人', '自己', '夢見我']
            if any(keyword in dream_text for keyword in person_keywords):
                negative_prompt = "ugly, blurry, distorted, deformed, low quality, bad anatomy, multiple heads, extra limbs"
            else:
                negative_prompt = "human, person, ugly, blurry, distorted, deformed, low quality, bad anatomy"
            
            # 生成參數
            seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator(self.current_device).manual_seed(seed)
            
            generation_params = {
                "prompt": image_prompt,
                "negative_prompt": negative_prompt,
                "height": 512,
                "width": 512,
                "num_inference_steps": 20 if self.current_device != "cpu" else 15,
                "guidance_scale": 7.5,
                "generator": generator
            }
            
            # 清理記憶體
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # 生成圖像
            print("🎨 開始生成圖像...")
            with torch.no_grad():
                result = self.image_pipe(**generation_params)
                
                if not result.images or len(result.images) == 0:
                    print("❌ 圖像生成失敗")
                    return None
                
                generated_image = result.images[0]
            
            # 處理圖像
            if generated_image.mode != 'RGB':
                generated_image = generated_image.convert('RGB')
            
            # 調整亮度
            img_array = np.array(generated_image)
            avg_brightness = np.mean(img_array)
            
            if avg_brightness < 30:
                img_array = np.clip(img_array * 1.3 + 20, 0, 255).astype(np.uint8)
                generated_image = Image.fromarray(img_array)
            
            # 保存圖像
            timestamp = int(time.time())
            random_id = str(uuid.uuid4())[:8]
            output_filename = f"dream_{timestamp}_{random_id}.png"
            
            output_dir = os.path.join(self.static_dir, 'images')
            output_path = os.path.join(output_dir, output_filename)
            
            generated_image.save(output_path, format='PNG', quality=95)
            print(f"✅ 圖像已保存: {output_filename}")
            
            return os.path.join('images', output_filename)
            
        except Exception as e:
            print(f"❌ 圖像生成錯誤: {e}")
            return None

    def _generate_video(self, image_path, dream_text, progress_callback=None):
        """生成夢境影片（原始方法，保持向後兼容）"""
        return self._generate_video_with_settings(
            image_path, dream_text, 5, 'standard', progress_callback
        )[0]  # 只返回路徑

    def _generate_video_with_settings(self, image_path, dream_text, video_length, video_quality, progress_callback=None):
        """
        根據設定生成夢境影片
        
        Args:
            image_path: 圖像路徑
            dream_text: 夢境描述
            video_length: 影片長度（秒）
            video_quality: 影片品質 ('fast', 'standard', 'high')
            progress_callback: 進度回調函數
        
        Returns:
            tuple: (影片路徑, 生成資訊)
        """
        if not self._check_framepack_status():
            print("❌ FramePack 影片生成不可用")
            if progress_callback:
                progress_callback("❌ 影片生成功能不可用")
            return None, None
        
        try:
            # 完整的圖像路徑
            full_image_path = os.path.join(self.static_dir, image_path)
            
            if not os.path.exists(full_image_path):
                print(f"❌ 圖像文件不存在: {full_image_path}")
                return None, None
            
            print(f"🎬 開始生成影片，輸入圖像: {full_image_path}")
            print(f"📊 影片設定: {video_length}秒, {video_quality}品質")
            
            # 根據品質設定調整生成參數
            generation_params = self._get_video_generation_params(video_quality)
            
            # 生成影片
            start_time = time.time()
            video_path = self._generate_video_with_params(
                image_path=full_image_path,
                dream_text=dream_text,
                video_length=video_length,
                generation_params=generation_params,
                progress_callback=progress_callback
            )
            generation_time = time.time() - start_time
            
            if video_path:
                # 準備生成資訊
                video_info = {
                    'length': video_length,
                    'quality': video_quality,
                    'generationTime': round(generation_time, 1),
                    'parameters': generation_params
                }
                
                print(f"✅ 影片生成成功: {video_path}")
                print(f"⏱️ 生成時間: {generation_time:.1f} 秒")
                return video_path, video_info
            else:
                print("❌ 影片生成失敗")
                return None, None
                
        except Exception as e:
            print(f"❌ 影片生成錯誤: {e}")
            if progress_callback:
                progress_callback(f"❌ 影片生成失敗: {str(e)}")
            return None, None

    def _get_video_generation_params(self, video_quality):
        """
        根據品質設定獲取生成參數
        
        Args:
            video_quality: 'fast', 'standard', 'high'
        
        Returns:
            dict: 生成參數
        """
        base_params = {
            'use_teacache': True,
            'num_inference_steps': 25,
            'guidance_scale': 10.0,
            'fps': 30
        }
        
        if video_quality == 'fast':
            # 快速模式：降低步數，啟用所有加速功能
            return {
                **base_params,
                'use_teacache': True,
                'num_inference_steps': 15,
                'guidance_scale': 8.0,
                'optimization_level': 'aggressive'
            }
        elif video_quality == 'high':
            # 高品質模式：增加步數，關閉一些加速功能
            return {
                **base_params,
                'use_teacache': False,  # 關閉 TeaCache 以獲得更好品質
                'num_inference_steps': 35,
                'guidance_scale': 12.0,
                'optimization_level': 'quality'
            }
        else:
            # 標準模式：平衡設定
            return {
                **base_params,
                'optimization_level': 'balanced'
            }

    def _generate_video_with_params(self, image_path, dream_text, video_length, generation_params, progress_callback=None):
        """
        使用指定參數生成影片
        """
        try:
            # 導入 FramePack 生成函數
            from framepack_video_generator import generate_dream_video
            
            # 從生成參數中提取品質設定
            optimization_level = generation_params.get('optimization_level', 'balanced')
            
            # 將優化等級映射到品質設定
            quality_mapping = {
                'aggressive': 'fast',
                'balanced': 'standard', 
                'quality': 'high'
            }
            video_quality = quality_mapping.get(optimization_level, 'standard')
            
            # 生成影片
            video_path = generate_dream_video(
                image_path=image_path,
                dream_text=dream_text,
                video_length=video_length,
                video_quality=video_quality,
                static_dir=self.static_dir,
                progress_callback=progress_callback
            )
            
            return video_path
            
        except Exception as e:
            print(f"❌ 參數化影片生成錯誤: {e}")
            # 降級到原始方法
            return self._generate_video(image_path.replace(self.static_dir + '/', ''), dream_text, progress_callback)

    def _analyze_psychology(self, dream_text):
        """心理分析"""
        system_prompt = """你是夢境心理分析專家，分析夢境的象徵意義和心理狀態，
提供150-200字的分析，使用溫和支持性語調，繁體中文回答。"""
        
        user_prompt = f"夢境描述: {dream_text}\n請提供心理分析："
        
        analysis = self._call_ollama(system_prompt, user_prompt)
        return analysis if analysis else "暫時無法進行心理分析。"

    def _save_dream_result(self, data):
        """保存夢境分析結果"""
        try:
            share_id = str(uuid.uuid4())
            share_dir = os.path.join(self.static_dir, 'shares')
            
            share_data = {
                'id': share_id,
                'timestamp': int(time.time()),
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'finalStory': data.get('finalStory', ''),
                'imagePath': data.get('imagePath', ''),
                'videoPath': data.get('videoPath', ''),  # 新增影片路徑
                'psychologyAnalysis': data.get('psychologyAnalysis', '')
            }
            
            share_file = os.path.join(share_dir, f"{share_id}.json")
            with open(share_file, 'w', encoding='utf-8') as f:
                json.dump(share_data, f, ensure_ascii=False, indent=2)
            
            return share_id
        except Exception as e:
            print(f"❌ 保存分享結果失敗: {e}")
            return None

    # 路由處理函數
    def index(self):
        return render_template('index.html')

    def api_status(self):
        ollama_status = self._check_ollama_status()
        local_models_status = self.models_loaded or self._load_image_model()
        framepack_status = self._check_framepack_status()
        
        return jsonify({
            'ollama': ollama_status,
            'local_models': local_models_status,
            'framepack_video': framepack_status,  # 新增 FramePack 狀態
            'device': self.current_device,
            'available_models': list(self.models.keys()),
            'timestamp': int(time.time())
        })

    def analyze(self):
        data = request.json
        dream_text = data.get('dream', '')
        selected_model = data.get('model', 'stable-diffusion-v1-5')
        generate_video = data.get('generateVideo', False)  # 是否生成影片
        video_length = data.get('videoLength', 5)  # 影片長度（秒）
        video_quality = data.get('videoQuality', 'standard')  # 影片品質
        
        # 輸入驗證
        if not dream_text or len(dream_text.strip()) < 10:
            return jsonify({'error': '請輸入至少10個字的夢境描述'}), 400
        
        if len(dream_text.strip()) > 2000:
            return jsonify({'error': '夢境描述過長，請控制在2000字以內'}), 400
        
        # 影片參數驗證
        if generate_video:
            if not isinstance(video_length, int) or video_length < 3 or video_length > 15:
                return jsonify({'error': '影片長度必須在 3-15 秒之間'}), 400
            
            if video_quality not in ['fast', 'standard', 'high']:
                return jsonify({'error': '無效的影片品質設定'}), 400
        
        # 防重複提交檢查
        request_id = f"{dream_text[:50]}_{int(time.time())}"
        
        if self.request_lock:
            print("⚠️  有其他請求正在處理中，請稍候...")
            return jsonify({'error': '系統正在處理其他請求，請稍候再試'}), 429
        
        if request_id in self.processing_requests:
            print("⚠️  相同請求已在處理中...")
            return jsonify({'error': '相同的請求正在處理中'}), 429
        
        # 設定處理狀態
        self.request_lock = True
        self.processing_requests.add(request_id)
        
        try:
            print(f"🌙 開始分析夢境 [ID: {request_id[:20]}...]: {dream_text[:50]}...")
            if generate_video:
                print(f"🎬 此次分析將包含 {video_length} 秒 {video_quality} 品質影片生成")
            
            # 檢查服務狀態
            ollama_status = self._check_ollama_status()
            if not ollama_status:
                return jsonify({'error': 'Ollama服務不可用'}), 503
            
            # 生成故事
            print("📖 生成夢境故事...")
            final_story = self._generate_story(dream_text)
            
            # 生成圖像
            print("🎨 生成夢境圖像...")
            local_models_status = self._load_image_model()
            image_path = None
            if local_models_status:
                image_path = self._generate_image(dream_text)
            
            # 生成影片（如果請求）
            video_path = None
            video_generation_info = None
            if generate_video and image_path:
                print(f"🎬 生成 {video_length} 秒夢境影片（{video_quality} 品質）...")
                
                def video_progress_callback(message):
                    print(f"📹 影片生成進度: {message}")
                
                video_path, video_generation_info = self._generate_video_with_settings(
                    image_path, dream_text, video_length, video_quality, video_progress_callback
                )
            
            # 心理分析
            print("🧠 進行心理分析...")
            psychology_analysis = self._analyze_psychology(dream_text)
            
            response = {
                'finalStory': final_story,
                'imagePath': '/static/' + image_path if image_path else None,
                'videoPath': '/static/' + video_path if video_path else None,
                'psychologyAnalysis': psychology_analysis,
                'apiStatus': {
                    'ollama': ollama_status,
                    'local_models': local_models_status,
                    'framepack_video': self._check_framepack_status(),
                    'device': self.current_device,
                    'current_model': selected_model
                },
                'processingInfo': {
                    'timestamp': int(time.time()),
                    'inputLength': len(dream_text),
                    'storyLength': len(final_story) if final_story else 0,
                    'videoGenerated': video_path is not None,
                    'videoInfo': video_generation_info,  # 包含影片詳細資訊
                    'requestId': request_id[:20]
                }
            }
            
            print(f"✅ 夢境分析完成 [ID: {request_id[:20]}...]")
            if video_path:
                print(f"🎬 影片已生成: {video_path} ({video_length}秒, {video_quality}品質)")
            
            return jsonify(response)
            
        except Exception as e:
            print(f"❌ 分析錯誤 [ID: {request_id[:20]}...]: {e}")
            return jsonify({'error': '處理過程中發生錯誤'}), 500
        
        finally:
            # 清理處理狀態
            self.request_lock = False
            self.processing_requests.discard(request_id)
            print(f"🔓 釋放請求鎖 [ID: {request_id[:20]}...]")

    def share_result(self):
        data = request.json
        
        if not data or 'finalStory' not in data:
            return jsonify({'error': '缺少必要的夢境分析數據'}), 400
        
        try:
            share_id = self._save_dream_result(data)
            
            if not share_id:
                return jsonify({'error': '創建分享失敗'}), 500
            
            share_url = url_for('view_shared', share_id=share_id, _external=True)
            
            return jsonify({
                'shareId': share_id, 
                'shareUrl': share_url,
                'timestamp': int(time.time())
            })
            
        except Exception as e:
            print(f"❌ 分享處理錯誤: {e}")
            return jsonify({'error': '處理分享請求時發生錯誤'}), 500

    def view_shared(self, share_id):
        try:
            share_file = os.path.join(self.static_dir, 'shares', f"{share_id}.json")
            
            if not os.path.exists(share_file):
                return jsonify({'error': '找不到該分享內容'}), 404
            
            with open(share_file, 'r', encoding='utf-8') as f:
                share_data = json.load(f)
            
            return render_template('shared.html', data=share_data)
            
        except Exception as e:
            print(f"❌ 載入分享內容錯誤: {e}")
            return jsonify({'error': '載入分享內容時發生錯誤'}), 500

    def run(self, debug=False, host='0.0.0.0', port=5002):
        """啟動應用"""
        print("🚀 啟動夢境分析應用...")
        print(f"📁 FramePack 影片生成: {'✅ 可用' if self._check_framepack_status() else '❌ 不可用'}")
        self.app.run(debug=debug, host=host, port=port, threaded=True)


# 主程式入口
if __name__ == '__main__':
    dream_analyzer = DreamAnalyzer()
    dream_analyzer.run(debug=False)