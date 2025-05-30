"""
FramePack 影片生成模組
整合 FramePack 的 Image-to-Video 功能到夢境分析系統中
"""

import os
import sys
import torch
import time
import uuid
import traceback
import numpy as np
from PIL import Image
from pathlib import Path

# 添加 FramePack 模組路徑
framepack_path = os.path.join(os.path.dirname(__file__), 'FramePack')
if framepack_path not in sys.path:
    sys.path.insert(0, framepack_path)

try:
    from diffusers import AutoencoderKLHunyuanVideo
    from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel
    from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode
    from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, generate_timestamp
    from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
    from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
    from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, unload_complete_models, load_model_as_complete
    from diffusers_helper.clip_vision import hf_clip_vision_encode
    from diffusers_helper.bucket_tools import find_nearest_bucket
    FRAMEPACK_AVAILABLE = True
    print("✅ FramePack 模組載入成功")
except ImportError as e:
    FRAMEPACK_AVAILABLE = False
    print(f"❌ FramePack 模組載入失敗: {e}")
    print("請確保 FramePack 資料夾在正確位置且依賴項目已安裝")


class FramePackVideoGenerator:
    """FramePack 影片生成器"""
    
    def __init__(self, static_dir="./static"):
        self.static_dir = static_dir
        self.models_loaded = False
        self.device = self._get_device()
        self.models = {}
        
        # 創建輸出目錄
        self.videos_dir = os.path.join(static_dir, 'videos')
        os.makedirs(self.videos_dir, exist_ok=True)
        
        print(f"🎬 FramePack 影片生成器初始化，使用設備: {self.device}")
    
    def _get_device(self):
        """獲取最佳可用設備"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def is_available(self):
        """檢查 FramePack 是否可用"""
        return FRAMEPACK_AVAILABLE and self._check_gpu_requirements()
    
    def _check_gpu_requirements(self):
        """檢查 GPU 記憶體需求"""
        if self.device.type == 'cuda':
            try:
                free_mem_gb = get_cuda_free_memory_gb(self.device)
                return free_mem_gb >= 6  # FramePack 最低需求 6GB
            except:
                return False
        elif self.device.type == 'mps':
            # MPS 暫時支援較有限，謹慎返回
            return True
        else:
            return False  # CPU 模式通常效能不足
    
    def load_models(self):
        """載入 FramePack 模型"""
        if not FRAMEPACK_AVAILABLE:
            raise RuntimeError("FramePack 模組不可用")
        
        if self.models_loaded:
            return True
        
        try:
            print("🔄 正在載入 FramePack 模型...")
            
            # 設定 HuggingFace 快取目錄
            os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), 'FramePack', 'hf_download')
            
            # 檢查 GPU 記憶體
            free_mem_gb = get_cuda_free_memory_gb(self.device) if self.device.type == 'cuda' else 8
            high_vram = free_mem_gb > 60
            
            print(f"可用 VRAM: {free_mem_gb:.1f} GB, 高VRAM模式: {high_vram}")
            
            # 載入文本編碼器
            print("📝 載入文本編碼器...")
            self.models['text_encoder'] = LlamaModel.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo", 
                subfolder='text_encoder', 
                torch_dtype=torch.float16
            ).cpu()
            
            self.models['text_encoder_2'] = CLIPTextModel.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo", 
                subfolder='text_encoder_2', 
                torch_dtype=torch.float16
            ).cpu()
            
            # 載入分詞器
            self.models['tokenizer'] = LlamaTokenizerFast.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo", 
                subfolder='tokenizer'
            )
            
            self.models['tokenizer_2'] = CLIPTokenizer.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo", 
                subfolder='tokenizer_2'
            )
            
            # 載入 VAE
            print("🎨 載入 VAE...")
            self.models['vae'] = AutoencoderKLHunyuanVideo.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo", 
                subfolder='vae', 
                torch_dtype=torch.float16
            ).cpu()
            
            # 載入圖像編碼器
            print("🖼️ 載入圖像編碼器...")
            self.models['feature_extractor'] = SiglipImageProcessor.from_pretrained(
                "lllyasviel/flux_redux_bfl", 
                subfolder='feature_extractor'
            )
            
            self.models['image_encoder'] = SiglipVisionModel.from_pretrained(
                "lllyasviel/flux_redux_bfl", 
                subfolder='image_encoder', 
                torch_dtype=torch.float16
            ).cpu()
            
            # 載入主要的變換器模型 (使用 F1 版本)
            print("🤖 載入 FramePack 變換器模型...")
            self.models['transformer'] = HunyuanVideoTransformer3DModelPacked.from_pretrained(
                'lllyasviel/FramePack_F1_I2V_HY_20250503', 
                torch_dtype=torch.bfloat16
            ).cpu()
            
            # 設定所有模型為評估模式
            for model_name, model in self.models.items():
                if hasattr(model, 'eval'):
                    model.eval()
                if hasattr(model, 'requires_grad_'):
                    model.requires_grad_(False)
            
            # 啟用記憶體優化
            if not high_vram:
                self.models['vae'].enable_slicing()
                self.models['vae'].enable_tiling()
            
            # 啟用高質量輸出
            self.models['transformer'].high_quality_fp32_output_for_inference = True
            
            # 設定數據類型
            self.models['transformer'].to(dtype=torch.bfloat16)
            self.models['vae'].to(dtype=torch.float16)
            self.models['image_encoder'].to(dtype=torch.float16)
            self.models['text_encoder'].to(dtype=torch.float16)
            self.models['text_encoder_2'].to(dtype=torch.float16)
            
            # 根據 VRAM 情況決定載入策略
            if high_vram:
                # 高 VRAM 模式：全部載入到 GPU
                print("🚀 高VRAM模式：將所有模型載入到 GPU")
                for model_name, model in self.models.items():
                    if hasattr(model, 'to'):
                        model.to(self.device)
            else:
                # 低 VRAM 模式：使用動態卸載
                print("💾 低VRAM模式：啟用動態模型卸載")
                # 這裡可以根據需要設定動態卸載策略
            
            self.models_loaded = True
            self.high_vram = high_vram
            
            print("✅ FramePack 模型載入完成")
            return True
            
        except Exception as e:
            print(f"❌ FramePack 模型載入失敗: {e}")
            traceback.print_exc()
            return False
    
    def generate_video_from_image(self, image_path, prompt, video_length=5, progress_callback=None):
        """
        從圖像生成影片
        
        Args:
            image_path: 輸入圖像路徑
            prompt: 影片生成提示詞
            video_length: 影片長度（秒）
            progress_callback: 進度回調函數
        
        Returns:
            生成的影片文件路徑，失敗時返回 None
        """
        return self.generate_video_from_image_with_params(
            image_path=image_path,
            prompt=prompt,
            video_length=video_length,
            generation_params={},  # 使用預設參數
            progress_callback=progress_callback
        )

    def generate_video_from_image_with_params(self, image_path, prompt, video_length=5, generation_params=None, progress_callback=None):
        """
        使用指定參數從圖像生成影片
        
        Args:
            image_path: 輸入圖像路徑
            prompt: 影片生成提示詞
            video_length: 影片長度（秒）
            generation_params: 生成參數字典
            progress_callback: 進度回調函數
        
        Returns:
            生成的影片文件路徑，失敗時返回 None
        """
        if not self.models_loaded:
            if not self.load_models():
                return None
        
        # 合併預設參數
        default_params = {
            'use_teacache': True,
            'num_inference_steps': 25,
            'guidance_scale': 10.0,
            'fps': 30,
            'optimization_level': 'balanced'
        }
        
        if generation_params is None:
            generation_params = {}
        
        params = {**default_params, **generation_params}
        
        try:
            if progress_callback:
                progress_callback("🖼️ 載入並處理輸入圖像...")
            
            # 載入並處理輸入圖像
            input_image = Image.open(image_path)
            if input_image.mode != 'RGB':
                input_image = input_image.convert('RGB')
            
            input_image_np = np.array(input_image)
            
            if progress_callback:
                progress_callback("📝 編碼文本提示...")
            
            # 編碼文本提示
            llama_vec, clip_l_pooler = self._encode_text_prompt(prompt)
            
            if progress_callback:
                quality_info = params.get('optimization_level', 'balanced')
                progress_callback(f"🎬 開始生成影片（{quality_info}模式）...")
            
            # 使用參數生成影片
            video_path = self._generate_video_internal_with_params(
                input_image_np, 
                llama_vec, 
                clip_l_pooler, 
                video_length,
                params,
                progress_callback
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
    
    def _encode_text_prompt(self, prompt):
        """編碼文本提示"""
        # 載入文本編碼器到設備（如果需要）
        if not self.high_vram:
            load_model_as_complete(self.models['text_encoder'], target_device=self.device)
            load_model_as_complete(self.models['text_encoder_2'], target_device=self.device)
        
        # 編碼正面提示
        llama_vec, clip_l_pooler = encode_prompt_conds(
            prompt, 
            self.models['text_encoder'], 
            self.models['text_encoder_2'], 
            self.models['tokenizer'], 
            self.models['tokenizer_2']
        )
        
        # 裁剪和填充到固定長度
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        
        return llama_vec, clip_l_pooler
    
    def _generate_video_internal(self, input_image_np, llama_vec, clip_l_pooler, video_length, progress_callback):
        """內部影片生成邏輯（保留原始方法以向後兼容）"""
        return self._generate_video_internal_with_params(
            input_image_np, 
            llama_vec, 
            clip_l_pooler, 
            video_length,
            {},  # 使用預設參數
            progress_callback
        )

    def _generate_video_internal_with_params(self, input_image_np, llama_vec, clip_l_pooler, video_length, params, progress_callback):
        """
        使用指定參數的內部影片生成邏輯
        """
        
        # 調整圖像大小到合適的解析度
        H, W, C = input_image_np.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image_np, target_width=width, target_height=height)
        
        # 轉換為 PyTorch 張量
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
        
        if progress_callback:
            progress_callback("🎨 VAE 編碼圖像...")
        
        # VAE 編碼
        if not self.high_vram:
            load_model_as_complete(self.models['vae'], target_device=self.device)
        
        start_latent = vae_encode(input_image_pt, self.models['vae'])
        
        if progress_callback:
            progress_callback("👁️ CLIP 視覺編碼...")
        
        # CLIP 視覺編碼
        if not self.high_vram:
            load_model_as_complete(self.models['image_encoder'], target_device=self.device)
        
        image_encoder_output = hf_clip_vision_encode(
            input_image_np, 
            self.models['feature_extractor'], 
            self.models['image_encoder']
        )
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        
        # 設定數據類型
        transformer = self.models['transformer']
        llama_vec = llama_vec.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)
        
        if progress_callback:
            progress_callback("🚀 開始擴散採樣...")
        
        # 計算影片參數
        latent_window_size = 9  # 預設窗口大小
        total_latent_sections = (video_length * 30) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))
        
        # 載入變換器模型到設備
        if not self.high_vram:
            unload_complete_models()
            move_model_to_device_with_memory_preservation(
                transformer, 
                target_device=self.device, 
                preserved_memory_gb=6
            )
        
        # 根據參數設定初始化 TeaCache
        use_teacache = params.get('use_teacache', True)
        num_steps = params.get('num_inference_steps', 25)
        transformer.initialize_teacache(enable_teacache=use_teacache, num_steps=num_steps)
        
        if progress_callback and use_teacache:
            progress_callback(f"⚡ 已啟用 TeaCache 加速（{num_steps} 步驟）")
        elif progress_callback:
            progress_callback(f"🔧 使用標準模式（{num_steps} 步驟）")
        
        # 生成參數
        rnd = torch.Generator("cpu").manual_seed(42)  # 固定種子以確保一致性
        
        # 創建歷史潛在空間
        history_latents = torch.zeros(
            size=(1, 16, 16 + 2 + 1, height // 8, width // 8), 
            dtype=torch.float32
        ).cpu()
        
        history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
        total_generated_latent_frames = 1
        history_pixels = None
        
        # 從參數獲取指導尺度
        guidance_scale = params.get('guidance_scale', 10.0)
        
        # 生成影片段落
        for section_index in range(total_latent_sections):
            if progress_callback:
                progress = 50 + (section_index / total_latent_sections) * 40
                progress_callback(f"🎬 生成影片段落 {section_index + 1}/{total_latent_sections}... ({progress:.0f}%)")
            
            # 設定索引
            indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
            clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)
            
            # 準備清潔潛在變數
            clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
            clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)
            
            # 採樣生成（使用參數化設定）
            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=latent_window_size * 4 - 3,
                real_guidance_scale=1.0,
                distilled_guidance_scale=guidance_scale,  # 使用參數化指導尺度
                guidance_rescale=0.0,
                num_inference_steps=num_steps,  # 使用參數化步數
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=torch.ones((1, llama_vec.shape[1]), dtype=torch.bool, device=llama_vec.device),
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=torch.zeros_like(llama_vec),
                negative_prompt_embeds_mask=torch.ones((1, llama_vec.shape[1]), dtype=torch.bool, device=llama_vec.device),
                negative_prompt_poolers=torch.zeros_like(clip_l_pooler),
                device=self.device,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
            )
            
            # 更新歷史
            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)
            
            # VAE 解碼
            if not self.high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=self.device, preserved_memory_gb=8)
                load_model_as_complete(self.models['vae'], target_device=self.device)
            
            real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]
            
            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, self.models['vae']).cpu()
            else:
                section_latent_frames = latent_window_size * 2
                overlapped_frames = latent_window_size * 4 - 3
                
                current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], self.models['vae']).cpu()
                history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)
            
            if not self.high_vram:
                unload_complete_models()
        
        if progress_callback:
            progress_callback("💾 保存影片文件...")
        
        # 保存影片（使用參數化的 FPS 和品質）
        timestamp = int(time.time())
        random_id = str(uuid.uuid4())[:8]
        output_filename = f"dream_video_{timestamp}_{random_id}.mp4"
        output_path = os.path.join(self.videos_dir, output_filename)
        
        fps = params.get('fps', 30)
        # 根據優化等級調整 CRF（壓縮率）
        optimization_level = params.get('optimization_level', 'balanced')
        if optimization_level == 'high' or optimization_level == 'quality':
            crf = 12  # 更高品質，更大文件
        elif optimization_level == 'fast' or optimization_level == 'aggressive':
            crf = 20  # 較低品質，較小文件
        else:
            crf = 16  # 平衡
        
        save_bcthw_as_mp4(history_pixels, output_path, fps=fps, crf=crf)
        
        # 返回相對路徑
        return os.path.join('videos', output_filename)
    
    def cleanup_models(self):
        """清理模型以釋放記憶體"""
        if self.models_loaded:
            print("🧹 清理 FramePack 模型...")
            if not self.high_vram:
                unload_complete_models()
            
            # 清理 GPU 記憶體
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except:
                    pass
    
    def __del__(self):
        """析構函數，確保清理資源"""
        try:
            self.cleanup_models()
        except:
            pass


# 全局實例
_video_generator = None

def get_video_generator(static_dir="./static"):
    """獲取影片生成器實例（單例模式）"""
    global _video_generator
    if _video_generator is None:
        _video_generator = FramePackVideoGenerator(static_dir)
    return _video_generator

def generate_dream_video(image_path, dream_text, video_length=5, video_quality='standard', static_dir="./static", progress_callback=None):
    """
    生成夢境影片的便捷函數
    
    Args:
        image_path: 輸入圖像路徑
        dream_text: 夢境描述文本
        video_length: 影片長度（秒）
        video_quality: 影片品質 ('fast', 'standard', 'high')
        static_dir: 靜態文件目錄
        progress_callback: 進度回調函數
    
    Returns:
        影片文件的相對路徑，失敗時返回 None
    """
    generator = get_video_generator(static_dir)
    
    if not generator.is_available():
        print("❌ FramePack 不可用或 GPU 記憶體不足")
        if progress_callback:
            progress_callback("❌ 影片生成不可用")
        return None
    
    # 創建適合影片生成的提示詞
    video_prompt = f"The scene comes alive with gentle movement, {dream_text}, cinematic, smooth motion, detailed"
    
    # 根據品質設定生成參數
    generation_params = {
        'fast': {
            'use_teacache': True,
            'num_inference_steps': 15,
            'guidance_scale': 8.0,
            'optimization_level': 'aggressive'
        },
        'standard': {
            'use_teacache': True,
            'num_inference_steps': 25,
            'guidance_scale': 10.0,
            'optimization_level': 'balanced'
        },
        'high': {
            'use_teacache': False,
            'num_inference_steps': 35,
            'guidance_scale': 12.0,
            'optimization_level': 'quality'
        }
    }.get(video_quality, {})
    
    return generator.generate_video_from_image_with_params(
        image_path=image_path,
        prompt=video_prompt,
        video_length=video_length,
        generation_params=generation_params,
        progress_callback=progress_callback
    )


if __name__ == "__main__":
    # 測試腳本
    print("🧪 測試 FramePack 影片生成器...")
    
    generator = get_video_generator()
    
    if generator.is_available():
        print("✅ FramePack 可用")
        
        # 測試模型載入
        if generator.load_models():
            print("✅ 模型載入成功")
        else:
            print("❌ 模型載入失敗")
    else:
        print("❌ FramePack 不可用")
    
    print("測試完成")