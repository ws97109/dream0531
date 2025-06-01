"""
影片生成整合模組 - 使用 HuggingFace API
"""
from api_video_generator import generate_dream_video as hf_generate, is_api_video_available

def is_framepack_available():
    """檢查 HuggingFace API 影片生成是否可用"""
    return is_api_video_available()

def generate_dream_video(image_path, dream_text, video_length=3, video_quality='standard', 
                        static_dir="./static", progress_callback=None):
    """
    影片生成主函數 - 使用 HuggingFace API
    """
    return hf_generate(
        image_path=image_path,
        dream_text=dream_text,
        video_length=video_length,
        video_quality=video_quality,
        static_dir=static_dir,
        progress_callback=progress_callback
    )

class FramePackVideoGenerator:
    def __init__(self, static_dir):
        self.static_dir = static_dir
    
    def is_available(self):
        return is_api_video_available()
    
    def generate_video_from_image(self, image_path, prompt, video_length=5, progress_callback=None):
        return generate_dream_video(
            image_path, prompt, video_length, 'standard', 
            self.static_dir, progress_callback
        )

def get_video_generator(static_dir="./static"):
    return FramePackVideoGenerator(static_dir)
