"""
FramePack 完整整合模組 - Mac M 晶片相容版本
直接使用 FramePack 的核心邏輯生成影片
"""
import os
import sys
import torch
import time
import uuid
import numpy as np
import traceback
from PIL import Image

# 設定 FramePack 環境
framepack_path = os.path.join(os.path.dirname(__file__), 'FramePack')
sys.path.insert(0, framepack_path)
os.environ['HF_HOME'] = os.path.join(framepack_path, 'hf_download')

# 導入 FramePack 核心組件
try:
    from diffusers import AutoencoderKLHunyuanVideo
    from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel
    from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode
    from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, generate_timestamp
    from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
    from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
    from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, unload_complete_models, load_model_as_complete
    from diffusers_helper.clip_vision import hf_clip_vision_encode
    from diffusers_helper.bucket_tools import find_nearest_bucket
    
    FRAMEPACK_AVAILABLE = True
    print("✅ FramePack 核心模組載入成功")
except ImportError as e:
    FRAMEPACK_AVAILABLE = False
    print(f"❌ FramePack 不可用: {e}")

# 全局模型儲存
_models = {}
_models_loaded = False
_high_vram = False

def is_framepack_available():
    """檢查 FramePack 是否可用（Mac 相容版本）"""
    if not FRAMEPACK_AVAILABLE:
        return False
    
    # Mac M 晶片檢查
    if torch.backends.mps.is_available():
        print("🍎 檢測到 Mac MPS 後端")
        return True
    elif torch.cuda.is_available():
        try:
            free_mem_gb = get_cuda_free_memory_gb(gpu)
            return free_mem_gb >= 6
        except:
            return False
    else:
        # CPU 模式也允許，雖然會很慢
        print("⚠️ 使用 CPU 模式，生成速度會較慢")
        return True

def load_framepack_models():
    """載入所有 FramePack 模型（Mac 相容版本）"""
    global _models, _models_loaded, _high_vram
    
    if _models_loaded:
        return True
    
    if not FRAMEPACK_AVAILABLE:
        print("❌ FramePack 模組不可用")
        return False
    
    try:
        print("🔄 開始載入 FramePack 模型...")
        
        # 檢查記憶體（適用於 Mac）
        free_mem_gb = get_cuda_free_memory_gb(gpu)
        _high_vram = free_mem_gb > 32  # Mac 通常有較多統一記憶體
        
        print(f'💾 可用記憶體: {free_mem_gb:.1f} GB')
        print(f'🚀 高記憶體模式: {_high_vram}')
        
        # 設定適合 Mac 的資料型別
        if gpu.type == 'mps':
            # MPS 對某些操作需要 float32
            text_dtype = torch.float32
            vae_dtype = torch.float32
            transformer_dtype = torch.float32
        else:
            text_dtype = torch.float16
            vae_dtype = torch.float16
            transformer_dtype = torch.bfloat16
        
        # 載入所有模型
        print("📝 載入文本編碼器...")
        _models['text_encoder'] = LlamaModel.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo", 
            subfolder='text_encoder', 
            torch_dtype=text_dtype
        ).cpu()
        
        _models['text_encoder_2'] = CLIPTextModel.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo", 
            subfolder='text_encoder_2', 
            torch_dtype=text_dtype
        ).cpu()
        
        print("🔤 載入分詞器...")
        _models['tokenizer'] = LlamaTokenizerFast.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo", 
            subfolder='tokenizer'
        )
        
        _models['tokenizer_2'] = CLIPTokenizer.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo", 
            subfolder='tokenizer_2'
        )
        
        print("🎨 載入 VAE...")
        _models['vae'] = AutoencoderKLHunyuanVideo.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo", 
            subfolder='vae', 
            torch_dtype=vae_dtype
        ).cpu()
        
        print("🖼️ 載入圖像編碼器...")
        _models['feature_extractor'] = SiglipImageProcessor.from_pretrained(
            "lllyasviel/flux_redux_bfl", 
            subfolder='feature_extractor'
        )
        
        _models['image_encoder'] = SiglipVisionModel.from_pretrained(
            "lllyasviel/flux_redux_bfl", 
            subfolder='image_encoder', 
            torch_dtype=text_dtype
        ).cpu()
        
        print("🤖 載入 FramePack 變換器...")
        _models['transformer'] = HunyuanVideoTransformer3DModelPacked.from_pretrained(
            'lllyasviel/FramePack_F1_I2V_HY_20250503', 
            torch_dtype=transformer_dtype
        ).cpu()
        
        # 設定所有模型
        for model_name in ['vae', 'text_encoder', 'text_encoder_2', 'image_encoder', 'transformer']:
            model = _models[model_name]
            if hasattr(model, 'eval'):
                model.eval()
            if hasattr(model, 'requires_grad_'):
                model.requires_grad_(False)
        
        # 記憶體優化設定（Mac 特化）
        if not _high_vram or gpu.type == 'mps':
            _models['vae'].enable_slicing()
            _models['vae'].enable_tiling()
        
        # 高品質輸出設定
        _models['transformer'].high_quality_fp32_output_for_inference = True
        
        # 設定數據類型
        _models['transformer'].to(dtype=transformer_dtype)
        _models['vae'].to(dtype=vae_dtype)
        _models['image_encoder'].to(dtype=text_dtype)
        _models['text_encoder'].to(dtype=text_dtype)
        _models['text_encoder_2'].to(dtype=text_dtype)
        
        _models_loaded = True
        print("✅ 所有 FramePack 模型載入完成")
        return True
        
    except Exception as e:
        print(f"❌ FramePack 模型載入失敗: {e}")
        traceback.print_exc()
        return False

def generate_dream_video(image_path, dream_text, video_length=5, video_quality='standard', static_dir="./static", progress_callback=None):
    """
    生成夢境影片主函數（Mac 相容版本）
    """
    if not FRAMEPACK_AVAILABLE:
        print("❌ FramePack 不可用")
        if progress_callback:
            progress_callback("❌ FramePack 模組不可用")
        return None
    
    # 載入模型
    if not load_framepack_models():
        print("❌ 模型載入失敗")
        if progress_callback:
            progress_callback("❌ 模型載入失敗")
        return None
    
    try:
        if progress_callback:
            progress_callback("🖼️ 處理輸入圖像...")
        
        # 載入並處理圖像
        input_image = Image.open(image_path)
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        input_image_np = np.array(input_image)
        
        # 創建影片提示詞
        prompt = f"The scene comes alive with gentle movement, {dream_text}, cinematic, smooth motion, detailed"
        n_prompt = ""
        
        # 根據品質和設備調整參數
        if gpu.type == 'mps':
            # Mac MPS 優化參數
            quality_params = {
                'fast': {'steps': 10, 'gs': 6.0, 'teacache': True, 'crf': 22},
                'standard': {'steps': 15, 'gs': 8.0, 'teacache': True, 'crf': 18},
                'high': {'steps': 20, 'gs': 10.0, 'teacache': False, 'crf': 14}
            }
        else:
            # 原始參數
            quality_params = {
                'fast': {'steps': 15, 'gs': 8.0, 'teacache': True, 'crf': 20},
                'standard': {'steps': 25, 'gs': 10.0, 'teacache': True, 'crf': 16},
                'high': {'steps': 35, 'gs': 12.0, 'teacache': False, 'crf': 12}
            }
        
        params = quality_params.get(video_quality, quality_params['standard'])
        
        if progress_callback:
            device_info = "Mac MPS" if gpu.type == 'mps' else str(gpu)
            progress_callback(f"🎬 開始生成 {video_length} 秒影片（{video_quality} 品質，{device_info}）...")
        
        # 調用核心生成邏輯
        video_path = _generate_video_core(
            input_image_np, prompt, n_prompt, video_length, 
            params, static_dir, progress_callback
        )
        
        if progress_callback:
            progress_callback("✅ 影片生成完成！")
        
        return video_path
        
    except Exception as e:
        print(f"❌ 影片生成失敗: {e}")
        traceback.print_exc()
        if progress_callback:
            progress_callback(f"❌ 生成失敗: {str(e)}")
        return None

def _generate_video_core(input_image, prompt, n_prompt, total_second_length, params, static_dir, progress_callback):
    """
    核心影片生成邏輯（Mac 最佳化版本）
    """
    # 固定參數
    seed = 31337
    latent_window_size = 9
    cfg = 1.0
    rs = 0.0
    
    # Mac 記憶體保護設定
    if gpu.type == 'mps':
        gpu_memory_preservation = 4  # Mac MPS 較寬鬆的記憶體管理
    else:
        gpu_memory_preservation = 6
    
    # 從參數中提取設定
    steps = params['steps']
    gs = params['gs']
    use_teacache = params['teacache']
    mp4_crf = params['crf']
    
    # 計算總段落數
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))
    
    job_id = generate_timestamp()
    outputs_folder = os.path.join(static_dir, 'videos')
    os.makedirs(outputs_folder, exist_ok=True)
    
    try:
        # 清理記憶體
        if not _high_vram:
            unload_complete_models(
                _models['text_encoder'], _models['text_encoder_2'], 
                _models['image_encoder'], _models['vae'], _models['transformer']
            )
        
        # 文本編碼
        if progress_callback:
            progress_callback("📝 編碼文本提示...")
        
        if not _high_vram:
            fake_diffusers_current_device(_models['text_encoder'], gpu)
            load_model_as_complete(_models['text_encoder_2'], target_device=gpu)
        
        llama_vec, clip_l_pooler = encode_prompt_conds(
            prompt, _models['text_encoder'], _models['text_encoder_2'], 
            _models['tokenizer'], _models['tokenizer_2']
        )
        
        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
                n_prompt, _models['text_encoder'], _models['text_encoder_2'], 
                _models['tokenizer'], _models['tokenizer_2']
            )
        
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        
        # 處理輸入圖像
        if progress_callback:
            progress_callback("🖼️ 處理圖像尺寸...")
        
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
        
        # VAE 編碼
        if progress_callback:
            progress_callback("🎨 VAE 編碼...")
        
        if not _high_vram:
            load_model_as_complete(_models['vae'], target_device=gpu)
        
        start_latent = vae_encode(input_image_pt, _models['vae'])
        
        # CLIP 視覺編碼
        if progress_callback:
            progress_callback("👁️ CLIP 視覺編碼...")
        
        if not _high_vram:
            load_model_as_complete(_models['image_encoder'], target_device=gpu)
        
        image_encoder_output = hf_clip_vision_encode(
            input_image_np, _models['feature_extractor'], _models['image_encoder']
        )
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        
        # 設定數據類型（Mac MPS 相容）
        transformer = _models['transformer']
        target_dtype = transformer.dtype
        
        llama_vec = llama_vec.to(target_dtype)
        llama_vec_n = llama_vec_n.to(target_dtype)
        clip_l_pooler = clip_l_pooler.to(target_dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(target_dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(target_dtype)
        
        # 開始採樣
        if progress_callback:
            progress_callback("🚀 開始影片生成採樣...")
        
        rnd = torch.Generator("cpu").manual_seed(seed)
        
        history_latents = torch.zeros(
            size=(1, 16, 16 + 2 + 1, height // 8, width // 8), 
            dtype=torch.float32
        ).cpu()
        
        history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
        total_generated_latent_frames = 1
        history_pixels = None
        
        # 生成影片段落
        for section_index in range(total_latent_sections):
            if progress_callback:
                progress = 40 + (section_index / total_latent_sections) * 50
                progress_callback(f"🎬 生成段落 {section_index + 1}/{total_latent_sections}... ({progress:.0f}%)")
            
            if not _high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(
                    transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation
                )
            
            # 初始化 TeaCache
            transformer.initialize_teacache(enable_teacache=use_teacache, num_steps=steps)
            
            # 設定索引
            indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
            clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)
            
            clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
            clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)
            
            # 生成潛在變量
            try:
                generated_latents = sample_hunyuan(
                    transformer=transformer,
                    sampler='unipc',
                    width=width,
                    height=height,
                    frames=latent_window_size * 4 - 3,
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
                    device=gpu,
                    dtype=target_dtype,
                    image_embeddings=image_encoder_last_hidden_state,
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                )
            except Exception as e:
                print(f"⚠️ 採樣過程出現錯誤: {e}")
                if gpu.type == 'mps':
                    print("🍎 MPS 可能需要更多記憶體或降低品質設定")
                raise
            
            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)
            
            # VAE 解碼
            if not _high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
                load_model_as_complete(_models['vae'], target_device=gpu)
            
            real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]
            
            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, _models['vae']).cpu()
            else:
                section_latent_frames = latent_window_size * 2
                overlapped_frames = latent_window_size * 4 - 3
                
                current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], _models['vae']).cpu()
                history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)
            
            if not _high_vram:
                unload_complete_models()
        
        # 保存影片
        if progress_callback:
            progress_callback("💾 保存影片檔案...")
        
        output_filename = f"dream_video_{job_id}.mp4"
        output_path = os.path.join(outputs_folder, output_filename)
       
        save_bcthw_as_mp4(history_pixels, output_path, fps=30, crf=mp4_crf)
       
        print(f"✅ 影片已保存: {output_filename}")
        return os.path.join('videos', output_filename)
       
    except Exception as e:
       print(f"❌ 核心生成錯誤: {e}")
       traceback.print_exc()
       
       if not _high_vram:
           unload_complete_models(
               _models['text_encoder'], _models['text_encoder_2'], 
               _models['image_encoder'], _models['vae'], _models['transformer']
           )
       
       return None

# 向後相容介面
class FramePackVideoGenerator:
   def __init__(self, static_dir):
       self.static_dir = static_dir
   
   def is_available(self):
       return is_framepack_available()
   
   def generate_video_from_image(self, image_path, prompt, video_length=5, progress_callback=None):
       return generate_dream_video(
           image_path, prompt, video_length, 'standard', 
           self.static_dir, progress_callback
       )

def get_video_generator(static_dir="./static"):
   return FramePackVideoGenerator(static_dir)