#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：sd-webui-animed-video-controlnet 
@File    ：web_ui.py
@Author  ：AkiiLucky
@Date    ：2024/2/19 20:30 
'''

import os
import gradio as gr
from gradio import components
from animed_video_controlnet import generate_anime_video
import numpy as np
from datetime import datetime
import cv2


def process_inputs(text, image_path, video_path):
    video_name = video_path.split("/")[-1]
    video_name_with_extension = f"{''.join(video_name.split('.')[0:-1])}_{video_name.split('.')[-1]}"
    image_name = image_path.split("/")[-1]
    image_name_with_extension = f"{''.join(image_name.split('.')[0:-1])}_{image_name.split('.')[-1]}"

    prompt = "<lora:lcm_lora_v15_weights:1>,masterpiece,best quality,highres,(simple white background),anime screencap,"
    prompt += text
    output_folder = os.path.join("outputs", image_name_with_extension, video_name_with_extension)
    generate_anime_video(video_path, image_path, prompt, output_folder, test_mode=False)
    output_video = os.path.join(output_folder, "transfer_reference_keyframes_video.mp4")
    return output_video  # 返回视频文件路径


# 定义 Gradio 界面，使用新的组件导入方式
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        components.Textbox(lines=2, label="正向提示词", placeholder="输入提示词..."),  # 文本输入
        components.Image(type="filepath", width=728, height=256, label="上传动漫风格图像"),  # 图像输入
        components.Video(width=728, height=256, label="上传真人实拍的动作视频")  # 视频输入
    ],
    outputs=gr.PlayableVideo(width=728, height=256, label="输出动漫风格视频")  # 视频输出
)

# 启动应用
iface.launch(server_port=7861)
