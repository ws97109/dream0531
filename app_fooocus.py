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

# 本地路徑設定 - 請根據您的實際情況修改
FOOOCUS_PATH = "/Users/lishengfeng/Desktop/淡江課程/生成式AI/期末報告/Fooocus"
FRAMEPACK_PATH = "/Users/lishengfeng/Desktop/淡江課程/生成式AI/期末報告/FramePack"

# FramePack 直接整合的全局變量
framepack_models_loaded = False
framepack_models = {}

# ==================== FramePack 直接整合 ====================

def get_device():
    """獲取適合 Mac M4 的設備"""
    import torch
    if torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def get_memory_info(device):
    """獲取記憶體信息（Mac 版本）"""
    if device.type == 'mps':
        # MPS 沒有直接的記憶體查詢API，使用估算值
        import psutil
        available_memory = psutil.virtual_memory().available
        # 轉換為 GB 並保守估計可用於 MPS 的記憶體
        return (available_memory / (1024 ** 3)) * 0.5  # 假設可用一半記憶體
    else:
        # CPU 模式
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)

def initialize_framepack_models():
    """初始化 FramePack 模型（僅在第一次使用時載入）"""
    global framepack_models_loaded, framepack_models
    
    if framepack_models_loaded:
        return True
    
    try:
        logger.info("開始初始化 FramePack 模型...")
        
        # 添加 FramePack 路徑到 Python 路徑
        if FRAMEPACK_PATH not in sys.path:
            sys.path.insert(0, FRAMEPACK_PATH)
        
        # 設定環境變量
        os.environ['HF_HOME'] = os.path.join(FRAMEPACK_PATH, 'hf_download')
        
        # 導入必要模塊
        import torch
        from diffusers import AutoencoderKLHunyuanVideo
        from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
        from transformers import SiglipImageProcessor, SiglipVisionModel
        
        # 導入 FramePack 特定模塊
        from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
        
        # 獲取設備
        device = get_device()
        logger.info(f"使用設備: {device}")
        
        # 檢查記憶體
        try:
            free_mem_gb = get_memory_info(device)
            high_vram = free_mem_gb > 20  # 降低需求
            logger.info(f"可用記憶體: {free_mem_gb:.2f} GB, 高記憶體模式: {high_vram}")
        except:
            logger.warning("無法檢測記憶體，使用 CPU 模式")
            high_vram = False
            device = torch.device('cpu')
        
        # 載入模型（使用較小的記憶體設定）
        logger.info("載入文本編碼器...")
        text_encoder = LlamaModel.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo", 
            subfolder='text_encoder', 
            torch_dtype=torch.float16
        ).cpu()
        
        text_encoder_2 = CLIPTextModel.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo", 
            subfolder='text_encoder_2', 
            torch_dtype=torch.float16
        ).cpu()
        
        logger.info("載入分詞器...")
        tokenizer = LlamaTokenizerFast.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo", 
            subfolder='tokenizer'
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo", 
            subfolder='tokenizer_2'
        )
        
        logger.info("載入 VAE...")
        vae = AutoencoderKLHunyuanVideo.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo", 
            subfolder='vae', 
            torch_dtype=torch.float16
        ).cpu()
        
        logger.info("載入圖像編碼器...")
        feature_extractor = SiglipImageProcessor.from_pretrained(
            "lllyasviel/flux_redux_bfl", 
            subfolder='feature_extractor'
        )
        image_encoder = SiglipVisionModel.from_pretrained(
            "lllyasviel/flux_redux_bfl", 
            subfolder='image_encoder', 
            torch_dtype=torch.float16
        ).cpu()
        
        logger.info("載入變換器...")
        transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
            'lllyasviel/FramePackI2V_HY', 
            torch_dtype=torch.bfloat16
        ).cpu()
        
        # 設定模型為評估模式
        vae.eval()
        text_encoder.eval()
        text_encoder_2.eval()
        image_encoder.eval()
        transformer.eval()
        
        # 啟用記憶體優化
        if not high_vram:
            vae.enable_slicing()
            vae.enable_tiling()
        
        # 設定高質量輸出
        transformer.high_quality_fp32_output_for_inference = True
        
        # 禁用梯度計算
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)  
        image_encoder.requires_grad_(False)
        transformer.requires_grad_(False)
        
        # 保存模型到全局變量
        framepack_models = {
            'text_encoder': text_encoder,
            'text_encoder_2': text_encoder_2,
            'tokenizer': tokenizer,
            'tokenizer_2': tokenizer_2,
            'vae': vae,
            'feature_extractor': feature_extractor,
            'image_encoder': image_encoder,
            'transformer': transformer,
            'high_vram': high_vram,
            'device': device
        }
        
        framepack_models_loaded = True
        logger.info("✅ FramePack 模型初始化完成")
        return True
        
    except ImportError as e:
        logger.error(f"❌ FramePack 模塊導入失敗: {str(e)}")
        logger.error("請確認 FramePack 依賴已正確安裝")
        return False
    except Exception as e:
        logger.error(f"❌ FramePack 模型初始化失敗: {str(e)}")
        return False

def generate_video_with_framepack_direct(image_path, prompt):
    """直接使用 FramePack 核心功能生成視頻"""
    try:
        # 確保模型已載入
        if not initialize_framepack_models():
            logger.error("FramePack 模型未能正確初始化")
            return None
        
        logger.info("開始使用 FramePack 直接生成視頻...")
        
        # 導入必要的處理函數
        import torch
        from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode
        from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, resize_and_center_crop, generate_timestamp
        from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
        from diffusers_helper.clip_vision import hf_clip_vision_encode
        from diffusers_helper.bucket_tools import find_nearest_bucket
        
        # 創建輸出目錄
        output_dir = os.path.join(static_dir, 'videos')
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成唯一檔案名
        timestamp = int(time.time())
        random_id = str(uuid.uuid4())[:8]
        output_filename = f"dream_video_{timestamp}_{random_id}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # 載入和處理輸入圖像
        full_image_path = os.path.join(static_dir, image_path)
        input_image = np.array(Image.open(full_image_path))
        
        logger.info(f"處理圖像: {input_image.shape}")
        
        # 調整圖像尺寸
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        
        # 轉換為 PyTorch 張量
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
        
        # 獲取模型和設備
        models = framepack_models
        device = models['device']
        
        # 文本編碼
        logger.info("進行文本編碼...")
        
        # 移動文本編碼器到設備（如果可用）
        if device.type != 'cpu':
            models['text_encoder'].to(device)
            models['text_encoder_2'].to(device)
        
        llama_vec, clip_l_pooler = encode_prompt_conds(
            prompt, models['text_encoder'], models['text_encoder_2'], 
            models['tokenizer'], models['tokenizer_2']
        )
        
        # 負面提示詞（空）
        llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
            "", models['text_encoder'], models['text_encoder_2'], 
            models['tokenizer'], models['tokenizer_2']
        )
        
        # 處理文本向量
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        
        # VAE 編碼
        logger.info("進行 VAE 編碼...")
        if device.type != 'cpu':
            models['vae'].to(device)
        
        start_latent = vae_encode(input_image_pt, models['vae'])
        
        # CLIP Vision 編碼
        logger.info("進行 CLIP Vision 編碼...")
        if device.type != 'cpu':
            models['image_encoder'].to(device)
        
        image_encoder_output = hf_clip_vision_encode(
            input_image_np, models['feature_extractor'], models['image_encoder']
        )
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        
        # 類型轉換
        transformer_dtype = models['transformer'].dtype
        llama_vec = llama_vec.to(transformer_dtype)
        llama_vec_n = llama_vec_n.to(transformer_dtype)
        clip_l_pooler = clip_l_pooler.to(transformer_dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer_dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer_dtype)
        
        # 採樣參數（簡化版）
        logger.info("開始視頻生成採樣...")
        
        if device.type != 'cpu':
            models['transformer'].to(device)
        
        # 簡化的採樣參數
        seed = 31337
        steps = 15  # 進一步減少步數以適應 Mac
        cfg = 1.0
        gs = 10.0
        rs = 0.0
        num_frames = 25  # 減少幀數以節省記憶體
        
        rnd = torch.Generator("cpu").manual_seed(seed)
        
        # 創建採樣所需的索引和潛在變量
        latent_window_size = 7  # 減少窗口大小
        indices = torch.arange(0, sum([1, 0, latent_window_size, 1, 2, 16])).unsqueeze(0)
        clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, 0, latent_window_size, 1, 2, 16], dim=1)
        clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
        
        # 準備清潔潛在變量
        clean_latents_pre = start_latent.to(device)
        clean_latents_post = torch.zeros(1, 16, 1, height // 8, width // 8, device=device, dtype=transformer_dtype)
        clean_latents_2x = torch.zeros(1, 16, 2, height // 8, width // 8, device=device, dtype=transformer_dtype)
        clean_latents_4x = torch.zeros(1, 16, 16, height // 8, width // 8, device=device, dtype=transformer_dtype)
        clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
        
        # 進行採樣
        generated_latents = sample_hunyuan(
            transformer=models['transformer'],
            sampler='unipc',
            width=width,
            height=height,
            frames=num_frames,
            real_guidance_scale=cfg,
            distilled_guidance_scale=gs,
            guidance_rescale=rs,
            num_inference_steps=steps,
            generator=rnd,
            prompt_embeds=llama_vec,
            prompt_embeds_mask=llama_attention_mask,
            prompt_poolers=clip_l_pooler,
            negative_prompt_embeds=llama_vec_n,
            negative_prompt_embeds_mask=llama_attention_mask_n,
            negative_prompt_poolers=clip_l_pooler_n,
            device=device,
            dtype=transformer_dtype,
            image_embeddings=image_encoder_last_hidden_state,
            latent_indices=latent_indices,
            clean_latents=clean_latents,
            clean_latent_indices=clean_latent_indices,
            clean_latents_2x=clean_latents_2x,
            clean_latent_2x_indices=clean_latent_2x_indices,
            clean_latents_4x=clean_latents_4x,
            clean_latent_4x_indices=clean_latent_4x_indices,
            callback=None,  # 簡化版本不使用回調
        )
        
        # 將起始潛在變量添加到生成的潛在變量
        final_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)
        
        # VAE 解碼
        logger.info("進行 VAE 解碼...")
        history_pixels = vae_decode(final_latents, models['vae']).cpu()
        
        # 保存為 MP4
        logger.info(f"保存視頻到: {output_path}")
        save_bcthw_as_mp4(history_pixels, output_path, fps=30, crf=16)
        
        # 移動模型回 CPU 以節省記憶體
        if not models['high_vram']:
            models['text_encoder'].cpu()
            models['text_encoder_2'].cpu()
            models['vae'].cpu()
            models['image_encoder'].cpu()
            models['transformer'].cpu()
        
        logger.info("✅ FramePack 視頻生成完成")
        return os.path.join('videos', output_filename)
        
    except ImportError as e:
        logger.error(f"FramePack 模塊導入錯誤: {str(e)}")
        return None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("記憶體不足，嘗試使用較小的參數或關閉其他應用程式")
        else:
            logger.error(f"FramePack 運行時錯誤: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"FramePack 視頻生成失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ==================== 其他輔助函數 ====================

def check_local_services():
    """檢查本地服務和路徑狀態"""
    try:
        # 檢查Ollama
        ollama_response = requests.get("http://localhost:11434/api/tags", timeout=5)
        ollama_status = ollama_response.status_code == 200
        logger.info(f"Ollama API 狀態: {'正常' if ollama_status else '異常'}")
        
        # 檢查Fooocus路徑和文件
        fooocus_status = os.path.exists(FOOOCUS_PATH)
        fooocus_executable = False
        if fooocus_status:
            fooocus_main_files = [
                os.path.join(FOOOCUS_PATH, "launch.py"),
                os.path.join(FOOOCUS_PATH, "webui.py"),
                os.path.join(FOOOCUS_PATH, "main.py"),
                os.path.join(FOOOCUS_PATH, "entry_with_update.py")
            ]
            for file_path in fooocus_main_files:
                if os.path.exists(file_path):
                    fooocus_executable = True
                    logger.info(f"找到Fooocus執行文件: {os.path.basename(file_path)}")
                    break
        
        # 檢查FramePack路徑和核心文件
        framepack_status = os.path.exists(FRAMEPACK_PATH)
        framepack_executable = False
        if framepack_status:
            framepack_main_file = os.path.join(FRAMEPACK_PATH, "demo_gradio.py")
            framepack_core_files = [
                os.path.join(FRAMEPACK_PATH, "diffusers_helper"),
                framepack_main_file
            ]
            
            if all(os.path.exists(f) for f in framepack_core_files):
                framepack_executable = True
                logger.info("找到FramePack核心文件: demo_gradio.py 和 diffusers_helper")
            else:
                logger.warning("FramePack文件不完整")
        
        return ollama_status, fooocus_status and fooocus_executable, framepack_status and framepack_executable
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

def generate_image_with_fooocus(prompt):
    """使用本地Fooocus生成圖像 - 修正版本"""
    try:
        # 創建輸出目錄
        output_dir = os.path.join(static_dir, 'images')
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成唯一的檔案名
        timestamp = int(time.time())
        random_id = str(uuid.uuid4())[:8]
        output_filename = f"dream_{timestamp}_{random_id}.png"
        
        # 簡化和清理提示詞
        clean_prompt = prompt.replace('\n', ' ').replace('\r', ' ')[:200]  # 限制長度
        enhanced_prompt = f"{clean_prompt}, dreamlike, surreal, fantasy, high quality"
        
        logger.info(f"使用本地Fooocus生成圖像，簡化提示詞：{enhanced_prompt[:50]}...")
        
        # 檢查Fooocus可執行文件
        main_file = None
        fooocus_main_files = [
            os.path.join(FOOOCUS_PATH, "launch.py"),
            os.path.join(FOOOCUS_PATH, "webui.py"),
            os.path.join(FOOOCUS_PATH, "main.py"),
            os.path.join(FOOOCUS_PATH, "entry_with_update.py")
        ]
        
        for file_path in fooocus_main_files:
            if os.path.exists(file_path):
                main_file = file_path
                break
        
        if not main_file:
            logger.error("找不到Fooocus主執行文件")
            return create_default_image(output_filename, prompt)
        
        # 創建臨時輸出目錄
        temp_output_dir = os.path.join(FOOOCUS_PATH, "outputs")
        os.makedirs(temp_output_dir, exist_ok=True)
        
        # 方法1：嘗試啟動Fooocus服務並通過API調用
        try:
            # 先檢查是否已經有Fooocus服務在運行
            try:
                api_response = requests.get("http://localhost:7865/", timeout=2)
                if api_response.status_code == 200:
                    logger.info("檢測到Fooocus服務正在運行，嘗試API調用")
                    return generate_via_fooocus_api(enhanced_prompt, output_filename)
            except:
                pass
            
            # 如果沒有服務運行，啟動Fooocus（僅啟動服務，不直接生成）
            logger.info("啟動Fooocus服務...")
            fooocus_cmd = [
                sys.executable, main_file,
                "--listen", "127.0.0.1",
                "--port", "7865",
                "--output-path", temp_output_dir
            ]
            
            # 在背景啟動Fooocus服務
            process = subprocess.Popen(fooocus_cmd, cwd=FOOOCUS_PATH, 
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 等待服務啟動（最多30秒）
            for i in range(30):
                try:
                    test_response = requests.get("http://localhost:7865/", timeout=1)
                    if test_response.status_code == 200:
                        logger.info(f"Fooocus服務啟動成功（{i+1}秒）")
                        break
                except:
                    time.sleep(1)
            else:
                logger.warning("Fooocus服務啟動超時，使用預設圖像")
                process.terminate()
                return create_default_image(output_filename, prompt)
            
            # 服務啟動成功，嘗試通過API生成圖像
            result = generate_via_fooocus_api(enhanced_prompt, output_filename)
            
            # 生成完成後關閉服務
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
            
            return result
            
        except Exception as e:
            logger.error(f"Fooocus服務啟動失敗: {str(e)}")
            return create_default_image(output_filename, prompt)
    
    except Exception as e:
        logger.error(f"Fooocus圖像生成過程發生錯誤: {str(e)}")
        return create_default_image(f"error_{int(time.time())}.png", prompt)

def generate_via_fooocus_api(prompt, output_filename):
    """通過Fooocus API生成圖像"""
    try:
        api_url = "http://localhost:7865/v1/generation/text-to-image"
        
        payload = {
            "prompt": prompt,
            "negative_prompt": "blurry, low quality, deformed",
            "style_selections": ["Fooocus V2"],
            "performance_selection": "Speed",
            "aspect_ratios_selection": "1152×896",
            "image_number": 1,
            "image_seed": -1,
            "sharpness": 2.0,
            "guidance_scale": 4.0,
            "base_model_name": "juggernautXL_v45.safetensors",
            "refiner_model_name": "None",
            "refiner_switch": 0.5,
            "loras": [],
            "advanced_params": {},
            "require_base64": True,
            "async_process": False
        }
        
        logger.info("發送API請求到Fooocus...")
        response = requests.post(api_url, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            if result and "images" in result and len(result["images"]) > 0:
                # 解碼base64圖像
                image_data = result["images"][0]
                if image_data.startswith('data:image/'):
                    image_data = image_data.split(',', 1)[1]
                
                # 保存圖像
                output_dir = os.path.join(static_dir, 'images')
                output_path = os.path.join(output_dir, output_filename)
                
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(image_data))
                
                logger.info(f"成功通過Fooocus API生成圖像: {output_filename}")
                return os.path.join('images', output_filename)
        
        logger.error(f"Fooocus API調用失敗: {response.status_code}")
        return create_default_image(output_filename, prompt)
        
    except Exception as e:
        logger.error(f"Fooocus API調用出錯: {str(e)}")
        return create_default_image(output_filename, prompt)

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

# ==================== 路由定義 ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """檢查服務狀態"""
    ollama_status, fooocus_status, framepack_status = check_local_services()
    
    return jsonify({
        'ollama': ollama_status,
        'fooocus': fooocus_status,
        'framepack': framepack_status,
        'fooocus_path': FOOOCUS_PATH,
        'framepack_path': FRAMEPACK_PATH,
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
        ollama_status, fooocus_status, framepack_status = check_local_services()
        if not ollama_status:
            return jsonify({'error': 'Ollama服務不可用，請確認服務是否正常運行在 localhost:11434'}), 503
        
        # 步驟1: 使用dream_weaver處理夢境故事
        logger.info("步驟1: 開始夢境故事生成")
        initial_story, story_feedback, final_story = dream_weaver(dream_text)
        
        # 步驟2: 翻譯故事以便更好地生成圖像
        logger.info("步驟2: 開始翻譯故事")
        translation = translate_to_english(final_story)
        
        # 步驟3: 使用本地Fooocus生成圖像
        logger.info("步驟3: 開始生成圖像")
        if fooocus_status:
            image_path = generate_image_with_fooocus(translation)
        else:
            logger.warning("Fooocus不可用，使用預設圖像")
            timestamp = int(time.time())
            image_path = create_default_image(f"default_{timestamp}.png", translation)
        
        # 步驟4: 使用直接整合的 FramePack 生成視頻
        logger.info("步驟4: 開始生成視頻（直接整合版）")
        video_path = None
        if framepack_status and image_path:
            video_path = generate_video_with_framepack_direct(image_path, translation)
            if video_path:
                logger.info("✅ 視頻生成成功")
            else:
                logger.warning("⚠️ 視頻生成失敗，但不影響其他功能")
        else:
            logger.warning("FramePack不可用，跳過視頻生成")
        
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
                'fooocus': fooocus_status,
                'framepack': framepack_status
            },
            'processingInfo': {
                'timestamp': int(time.time()),
                'inputLength': len(dream_text),
                'storyLength': len(final_story) if final_story else 0,
                'useDirectIntegration': True
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
        ollama_status, fooocus_status, framepack_status = check_local_services()
        
        # 輸出狀態報告
        print("=" * 80)
        print("夢境編織者系統 - Mac M4 直接整合版本 啟動狀態報告")
        print("=" * 80)
        print(f"Ollama API (localhost:11434): {'✅ 正常' if ollama_status else '❌ 異常'}")
        print(f"Fooocus 路徑: {FOOOCUS_PATH}")
        print(f"Fooocus 狀態: {'✅ 可用' if fooocus_status else '❌ 不可用'}")
        print(f"FramePack 路徑: {FRAMEPACK_PATH}")
        print(f"FramePack 狀態: {'✅ 可用（MPS/CPU直接整合）' if framepack_status else '❌ 不可用'}")
        print(f"靜態檔案目錄: {static_dir}")
        
        # 檢查 PyTorch 和設備支持
        import torch
        device = get_device()
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"使用設備: {device}")
        if device.type == 'mps':
            print("✅ 已啟用 Metal Performance Shaders (MPS) 加速")
        else:
            print("⚠️  使用 CPU 模式，速度可能較慢")
        
        print("=" * 80)
        
        # 詳細的狀態說明和建議
        if not ollama_status:
            print("❌ 警告: Ollama API 無法連接")
            print("   請確認 Ollama 服務是否運行在 localhost:11434")
            print("   啟動命令: ollama serve")
            print()
        
        if not fooocus_status:
            print("❌ 警告: Fooocus 不可用")
            print(f"   當前設定路徑: {FOOOCUS_PATH}")
            print("   圖像生成將使用預設圖像")
            print("   請檢查 Fooocus 安裝和路徑設定")
            print()
        
        if not framepack_status:
            print("❌ 警告: FramePack 不可用（MPS/CPU直接整合模式）")
            print(f"   當前設定路徑: {FRAMEPACK_PATH}")
            print("   視頻生成功能將不可用")
            print("   請檢查:")
            print("   1. FramePack 路徑是否正確")
            print("   2. 是否有 demo_gradio.py 和 diffusers_helper 目錄")
            print("   3. FramePack 依賴是否已安裝")
            print("   4. PyTorch 是否支持 MPS 或 CPU 模式")
            print()
        
        # 系統功能說明
        print("🔧 系統功能狀態:")
        print(f"   • 故事生成: {'✅ 可用 (Ollama)' if ollama_status else '❌ 不可用'}")
        print(f"   • 文本翻譯: {'✅ 可用 (Ollama)' if ollama_status else '❌ 不可用'}")
        print(f"   • 圖像生成: {'✅ 可用 (本地Fooocus)' if fooocus_status else '⚠️  預設圖像'}")
        print(f"   • 視頻生成: {'✅ 可用 (MPS/CPU整合FramePack)' if framepack_status else '❌ 不可用'}")
        print(f"   • 心理分析: {'✅ 可用 (Ollama)' if ollama_status else '❌ 不可用'}")
        print()
        
        # 特殊說明
        print("🚀 Mac M4 優化特性:")
        if framepack_status:
            print("   • FramePack 模型自動使用 MPS 或 CPU 加速")
            print("   • 針對 Mac 記憶體進行優化配置")
            print("   • 視頻生成完全整合，無需額外的 Web 界面")
            print("   • 自動記憶體管理和設備優化")
        print("   • 所有功能通過統一界面使用")
        print("   • 完整的錯誤處理和日誌記錄")
        print("   • 針對 Apple Silicon 進行優化")
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
    except Exception as e:
        logger.error(f"系統啟動失敗: {str(e)}")
        import traceback
        traceback.print_exc()