import torch
import os
from diffusers import DiffusionPipeline
from diffusers import KolorsPipeline
import gradio as gr


#执行上面脚本时，请设置环境变量：
os.environ['HF_ENDPOINT'] = 'https://hf-api.gitee.com'
os.environ['HF_HOME'] = '~/.cache/gitee-ai'

# 目前启动脚本工作目录未指定为 app.py 所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to(device)


# prompt = '一张瓢虫的照片，微距，变焦，高质量，电影，拿着一个牌子，写着"可图"'

# examples = [f"./example{i}.jpg" for i in range(1,7)]
def text2img(prompt):
    image = pipe(
        prompt=prompt,
        negative_prompt="",
        guidance_scale=5.0,
        num_inference_steps=50,
        generator=torch.Generator(pipe.device).manual_seed(66),
    ).images[0]
    return image


title = "kolors测试"
description = ""
interface = gr.Interface(
        fn=text2img,
        description=description,
        inputs = gr.Text(label = "输入文字"),
        theme=gr.themes.Glass(),
        outputs=gr.Image(),
        # examples = examples,
        title=title,
        allow_flagging="never"
    )
interface.launch(debug=True)