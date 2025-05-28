from diffusers import StableDiffusionPipeline,UniPCMultistepScheduler
import torch
import gc
import matplotlib.pyplot as plt
import gradio as gr
import random
import os
from openai import OpenAI

# 定義可用的模型清單
models = {
    "MajicMIX Realistic v6": "digiplay/majicMIX_realistic_v6",
    "Realistic Vision V5.1": "SG161222/Realistic_Vision_V5.1_noVAE",
    "stable-diffusion-v1-5": "runwayml/stable-diffusion-v1-5",
}
model_name = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# 加載不同模型的函數
def load_model(model_key):
    global pipe, model_name

    # 獲取選擇的模型路徑
    model_name = models[model_key]


    # 加載新模型
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    return f"已成功載入模型: {model_key}"
# Groq翻譯機器人
api_key = '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'
os.environ['OPENAI_API_KEY'] = api_key

client = OpenAI(
    base_url = "https://api.groq.com/openai/v1"
)

def translate_to_english(text):
    """將任何語言翻譯成英文"""
    # 如果文本為空，直接返回
    if not text or text.strip() == "":
        return text

    # 檢查是否是英文
    if all('\u4e00' > char or char > '\u9fff' for char in text):
        return text

    try:
        # 創建翻譯請求
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "你是一位翻譯專家。請將用戶輸入的任何語言翻譯成英文，只返回翻譯結果。"},
                {"role": "user", "content": f"將以下文本翻譯成英文: {text}"}
            ],
            temperature=0.1
        )

        # 獲取翻譯結果
        translation = response.choices[0].message.content
        print(f"已翻譯: {text} -> {translation}")
        return translation

    except Exception as e:
        print(f"翻譯錯誤: {str(e)}")
        return text
# 整合翻譯功能的圖像生成函數
def generate_images(prompt, use_enhance, enhance_text, use_negative, negative_text,
                   use_custom_seed, custom_seed, height, width, steps, num_images):

    # 翻譯主要提示詞
    translated_prompt = translate_to_english(prompt)

    # 翻譯增強提示詞
    translated_enhance_text = enhance_text
    if use_enhance and enhance_text:
        translated_enhance_text = translate_to_english(enhance_text)

    # 翻譯負面提示詞
    translated_negative_text = negative_text
    if use_negative and negative_text:
        translated_negative_text = translate_to_english(negative_text)


    height = int(height)
    width = int(width)

    if height % 8 != 0 or width % 8 != 0:
        raise ValueError("高度和寬度必須是8的倍數！")

    if use_custom_seed:
        base_seed = int(custom_seed)
    else:
        base_seed = random.randint(0, 2**32 - 1)

    seeds = [base_seed + i for i in range(num_images)]

    prompts = []
    negative_prompts = []
    generators = []

    # 使用翻譯後的提示詞
    final_prompt = translated_prompt
    if use_enhance and translated_enhance_text:
        final_prompt = translated_prompt + ", " + translated_enhance_text

    # 使用翻譯後的負面提示詞
    final_negative = translated_negative_text if use_negative else None

    for seed in seeds:
        g = torch.Generator("cuda").manual_seed(seed)
        generators.append(g)
        prompts.append(final_prompt)
        negative_prompts.append(final_negative)

    gc.collect()
    torch.cuda.empty_cache()

    images = []
    for i in range(num_images):
        with torch.no_grad():
            image = pipe(
                prompt=prompts[i],
                negative_prompt=negative_prompts[i] if final_negative else None,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=7.5,
                generator=generators[i]
            ).images[0]
            images.append(image)

    return images, f"使用的 random seeds: {seeds}"

default_enhance = "detailed fur texture, bright eyes, whiskers, cinematic lighting"
default_negative = "human, person, ugly, blurry, distorted, deformed, low quality, bad anatomy"

with gr.Blocks(css=".gradio-container {background-color: #FAFAFA; padding: 20px;} .gr-button {font-size: 18px; background: linear-gradient(to right, #667eea, #764ba2); color: white;}") as demo:
    gr.Markdown("""
    # 🎨 多風格 AI 圖像工作室
    歡迎使用智能圖像生成器！

    選擇你喜愛的模型，輸入創意提示詞，調整細節參數，一鍵生成令人驚豔的藝術作品。無論是寫實風格的精緻肖像、動漫風格的角色設計、奇幻場景還是自然風光，都能輕鬆實現。調整參數探索無限可能，讓你的創意立即成真！

    模型提供：

    MajicMIX Realistic v6 提供優質寫實風格與細膩人物細節。

    Realistic Vision V5.1 專注於極致寫實與出色的光影材質表現。

    Stable Diffusion v1.5 是最穩定且多功能的基礎模型，適合各種風格創作。

    """)

    # 添加模型選擇下拉選單
    model_dropdown = gr.Dropdown(
        choices=list(models.keys()),
        value="stable-diffusion-v1-5",
        label="選擇模型",
    )

    with gr.Row():
        with gr.Column(scale=6):
            prompt = gr.Textbox(label="Prompt", placeholder="請輸入你的提示詞 (prompt)", lines=3)
            with gr.Row():
                use_enhance = gr.Checkbox(label="加強 Prompt", value=True)
                enhance_text = gr.Textbox(label="加強內容", value=default_enhance)
            with gr.Row():
                use_negative = gr.Checkbox(label="使用 Negative Prompt", value=True)
                negative_text = gr.Textbox(label="Negative Prompt 內容", value=default_negative)
            with gr.Row():
                use_custom_seed = gr.Checkbox(label="自訂 Random Seed", value=False)
                custom_seed = gr.Number(label="指定 seed (選填)", value=42)
            with gr.Row():
                height = gr.Dropdown(["512", "768", "1024"], label="高度 Height", value="512")
                width = gr.Dropdown(["512", "768", "1024"], label="寬度 Width", value="512")
            with gr.Row():
                steps = gr.Slider(10, 50, value=20, step=5, label="生成步數 (Steps)")
                num_images = gr.Slider(1, 4, step=1, value=1, label="生成張數")
            generate_btn = gr.Button("🚀 開始生成！")

        with gr.Column(scale=6):
            gallery = gr.Gallery(label="生成結果", columns=2, object_fit="contain", height="auto")
            seed_info = gr.Label(label="使用的 Random Seeds")
            model_info = gr.Textbox(label="模型資訊", value="已載入模型: stable-diffusion-v1-5")

    # 模型切換事件
    model_dropdown.change(
        fn=load_model,
        inputs=[model_dropdown],
        outputs=[model_info]
    )

    generate_btn.click(
        fn=generate_images,
        inputs=[prompt, use_enhance, enhance_text, use_negative, negative_text,
                use_custom_seed, custom_seed, height, width, steps, num_images],
        outputs=[gallery, seed_info]
    )
demo.launch(share=True, debug=True)