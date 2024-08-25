#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @author: wisdom-pan <wisdompan1@outlook.com>
# @date: 2024/08/22
#
import os
os.environ['PT_SDPA_ENABLE_HEAD_DIM_PADDING'] = '1'

import tqdm
import hashlib
import fitz
import gradio as gr
import spaces
from transformers import AutoModel
from transformers import AutoTokenizer
from PIL import Image
import torch
import numpy as np
import json
import cv2
os.environ['HF_ENDPOINT'] = 'https://hf-api.gitee.com'
#执行上面脚本时，请设置环境变量：
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '~/.cache/gitee-ai'


cache_dir = '/data/kb_cache'
os.makedirs(cache_dir, exist_ok=True)


def get_image_md5(img: Image.Image):
    img_byte_array = img.tobytes()
    hash_md5 = hashlib.md5()
    hash_md5.update(img_byte_array)
    hex_digest = hash_md5.hexdigest()
    return hex_digest


def calculate_md5_from_binary(binary_data):
    hash_md5 = hashlib.md5()
    hash_md5.update(binary_data)
    return hash_md5.hexdigest()


@spaces.GPU(duration=100)
def add_pdf_gradio(pdf_file_binary, progress=gr.Progress()):
    global model, tokenizer
    model.eval()

    knowledge_base_name = calculate_md5_from_binary(pdf_file_binary)

    this_cache_dir = os.path.join(cache_dir, knowledge_base_name)
    os.makedirs(this_cache_dir, exist_ok=True)

    with open(os.path.join(this_cache_dir, f"src.pdf"), 'wb') as file:
        file.write(pdf_file_binary)

    dpi = 200
    doc = fitz.open("pdf", pdf_file_binary)

    reps_list = []
    images = []
    image_md5s = []

    for page in progress.tqdm(doc):
        # with self.lock: # because we hope one 16G gpu only process one image at the same time
        pix = page.get_pixmap(dpi=dpi)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image_md5 = get_image_md5(image)
        image_md5s.append(image_md5)
        with torch.no_grad():
            reps = model(text=[''], image=[image], tokenizer=tokenizer).reps
        reps_list.append(reps.squeeze(0).cpu().numpy())
        images.append(image)

    for idx in range(len(images)):
        image = images[idx]
        image_md5 = image_md5s[idx]
        cache_image_path = os.path.join(this_cache_dir, f"{image_md5}.png")
        image.save(cache_image_path)

    np.save(os.path.join(this_cache_dir, f"reps.npy"), reps_list)

    with open(os.path.join(this_cache_dir, f"md5s.txt"), 'w') as f:
        for item in image_md5s:
            f.write(item + '\n')

    return knowledge_base_name

@spaces.GPU(duration=100)
def add_video_gradio(video_file_binary, progress=gr.Progress()):
    global model, tokenizer
    model.eval()

    knowledge_base_name = calculate_md5_from_binary(video_file_binary)

    this_cache_dir = os.path.join(cache_dir, knowledge_base_name)
    os.makedirs(this_cache_dir, exist_ok=True)

    video_path = os.path.join(this_cache_dir, f"src.mp4")
    with open(video_path, 'wb') as file:
        file.write(video_file_binary)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    reps_list = []
    images = []
    image_md5s = []

    for i in progress.tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_md5 = get_image_md5(image)
        image_md5s.append(image_md5)
        with torch.no_grad():
            reps = model(text=[''], image=[image], tokenizer=tokenizer).reps
        reps_list.append(reps.squeeze(0).cpu().numpy())
        images.append(image)

    cap.release()

    for idx in range(len(images)):
        image = images[idx]
        image_md5 = image_md5s[idx]
        cache_image_path = os.path.join(this_cache_dir, f"{image_md5}.png")
        image.save(cache_image_path)

    np.save(os.path.join(this_cache_dir, f"reps.npy"), reps_list)

    with open(os.path.join(this_cache_dir, f"md5s.txt"), 'w') as f:
        for item in image_md5s:
            f.write(item + '\n')

    return knowledge_base_name


@spaces.GPU
def retrieve_gradio(knowledge_base: str, query: str, topk: int):
    global model, tokenizer

    model.eval()

    target_cache_dir = os.path.join(cache_dir, knowledge_base)

    if not os.path.exists(target_cache_dir):
        return None

    md5s = []
    with open(os.path.join(target_cache_dir, f"md5s.txt"), 'r') as f:
        for line in f:
            md5s.append(line.rstrip('\n'))

    doc_reps = np.load(os.path.join(target_cache_dir, f"reps.npy"))

    query_with_instruction = "Represent this query for retrieving relavant document: " + query
    with torch.no_grad():
        query_rep = model(text=[query_with_instruction], image=[None], tokenizer=tokenizer).reps.squeeze(0).cpu()

    query_md5 = hashlib.md5(query.encode()).hexdigest()

    doc_reps_cat = torch.stack([torch.Tensor(i) for i in doc_reps], dim=0)

    similarities = torch.matmul(query_rep, doc_reps_cat.T)

    topk_values, topk_doc_ids = torch.topk(similarities, k=topk)

    topk_values_np = topk_values.cpu().numpy()

    topk_doc_ids_np = topk_doc_ids.cpu().numpy()

    similarities_np = similarities.cpu().numpy()

    images_topk = [Image.open(os.path.join(target_cache_dir, f"{md5s[idx]}.png")) for idx in topk_doc_ids_np]

    with open(os.path.join(target_cache_dir, f"q-{query_md5}.json"), 'w') as f:
        f.write(json.dumps(
            {
                "knowledge_base": knowledge_base,
                "query": query,
                "retrived_docs": [os.path.join(target_cache_dir, f"{md5s[idx]}.png") for idx in topk_doc_ids_np]
            }, indent=4, ensure_ascii=False
        ))

    return images_topk


def upvote(knowledge_base, query):
    """
    更新知识库中指定查询的用户偏好为“upvote”，并将结果保存到新的JSON文件中。

    参数:
    knowledge_base (str): 知识库的名称或路径。
    query (str): 用户查询的字符串。
    """
    global model, tokenizer

    target_cache_dir = os.path.join(cache_dir, knowledge_base)

    query_md5 = hashlib.md5(query.encode()).hexdigest()

    with open(os.path.join(target_cache_dir, f"q-{query_md5}.json"), 'r') as f:
        data = json.loads(f.read())

    data["user_preference"] = "upvote"

    with open(os.path.join(target_cache_dir, f"q-{query_md5}-withpref.json"), 'w') as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))

    print("up", os.path.join(target_cache_dir, f"q-{query_md5}-withpref.json"))

    gr.Info('Received, babe! Thank you!')

    return


def downvote(knowledge_base, query):
    global model, tokenizer

    target_cache_dir = os.path.join(cache_dir, knowledge_base)

    query_md5 = hashlib.md5(query.encode()).hexdigest()

    with open(os.path.join(target_cache_dir, f"q-{query_md5}.json"), 'r') as f:
        data = json.loads(f.read())

    data["user_preference"] = "downvote"

    with open(os.path.join(target_cache_dir, f"q-{query_md5}-withpref.json"), 'w') as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))

    print("down", os.path.join(target_cache_dir, f"q-{query_md5}-withpref.json"))

    gr.Info('Received, babe! Thank you!')

    return


device = 'cuda'

print("emb model load begin...")
model_path = 'RhapsodyAI/minicpm-visual-embedding-v0'  # replace with your local model path
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
model.eval()
model.to(device)
print("emb model load success!")

print("gen model load begin...")
gen_model_path = 'openbmb/MiniCPM-V-2_6'
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_path, trust_remote_code=True)
# gen_model = AutoModel.from_pretrained(gen_model_path, trust_remote_code=True, attn_implementation='sdpa',
#                                       torch_dtype=torch.bfloat16)
gen_model = AutoModel.from_pretrained(gen_model_path, trust_remote_code=True,torch_dtype=torch.bfloat16)
gen_model.eval()
gen_model.to(device)
print("gen model load success!")


@spaces.GPU(duration=50)
def answer_question(images, question):
    global gen_model, gen_tokenizer
    # here each element of images is a tuple of (image_path, None).
    images_ = [Image.open(image[0]).convert('RGB') for image in images]
    msgs = [{'role': 'user', 'content': [question, *images_]}]
    answer = gen_model.chat(
        image=None,
        msgs=msgs,
        tokenizer=gen_tokenizer
    )
    print(answer)
    return answer


with gr.Blocks() as app:
    gr.Markdown("#你的个人图书馆管家FancyLibrary")

    gr.Markdown("""
- **提出问题**，它会检索最相关的页面，将根据返回top k的页面回答您的问题。

    -可以帮助您阅读较长的图像、文本类的PDF以及视频中的与问题相关的关键帧并找到回答您问题的页面。

    - 可以帮助您建立个人图书馆并从多模态数据中最相关的页面用于RAG问答系统。

    - 用完整的视觉阅读、存储、检索和回答.
""")

    # gr.Markdown(
    #     "- Currently online demo support PDF document with less than 50 pages due to GPU time limit. Deploy on your own machine for longer PDFs and books.")

    with gr.Row():
        file_input = gr.File(type="binary", label="上传你的文件")
        file_result = gr.Text(label="知识库ID ")
        process_button = gr.Button("处理 PDF（PDF 上传成功后才可点击）")

    process_button.click(add_pdf_gradio, inputs=[file_input], outputs=file_result)

    with gr.Row():
        video_input = gr.File(type="binary", label="上传你的视频")
        video_result = gr.Text(label="知识库ID ")
        process_video_button = gr.Button("处理视频（视频上传成功后才可点击）")

    process_video_button.click(add_video_gradio, inputs=[video_input], outputs=video_result)

    with gr.Row():
        kb_id_input = gr.Text(label="您的知识库 ID（在此处粘贴您的知识库 ID，可重复使用：）")
        query_input = gr.Text(label="输入你的问题")
        topk_input = inputs = gr.Number(value=5, minimum=1, maximum=10, step=1, label="要检索的页数，1-10页")
        retrieve_button = gr.Button("步骤2：检索页面")

    with gr.Row():
        images_output = gr.Gallery(label="检索页面")
        retrieve_button.click(retrieve_gradio, inputs=[kb_id_input, query_input, topk_input], outputs=images_output)

    with gr.Row():
        button = gr.Button("步骤 3：使用检索到的页面回答问题")

        gen_model_response = gr.Textbox(label="检索响应")

        button.click(fn=answer_question, inputs=[images_output, query_input], outputs=gen_model_response)

    with gr.Row():
        downvote_button = gr.Button("👍赞一下")
        upvote_button = gr.Button("👎踩一下")

    upvote_button.click(upvote, inputs=[kb_id_input, query_input], outputs=None)
    downvote_button.click(downvote, inputs=[kb_id_input, query_input], outputs=None)



app.launch()

