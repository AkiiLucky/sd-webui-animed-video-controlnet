#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：sd-webui-animed-video-controlnet 
@File    ：test_all_video_image_pairs.py
@Author  ：AkiiLucky
@Date    ：2024/2/17 11:46 
'''

import os
from animed_video_controlnet import generate_anime_video
from utils import sort_file_name

if __name__ == '__main__':
    images_path = os.path.join("test_dataset", "images")
    videos_path = os.path.join("test_dataset", "videos")
    image_files = [f for f in os.listdir(images_path) if f.endswith(".png")]
    image_files = sort_file_name(image_files)  # 按照文件名排序确保顺序正确
    video_files = [f for f in os.listdir(videos_path) if f.endswith(".mp4")]

    base_prompt = "<lora:lcm_lora_v15_weights:1>,masterpiece,best quality,highres,(simple white background),anime screencap"
    image_prompts = [
        "1boy,male focus,blue hair,(solo),white shorts,blue eyes,white sportswear,full body,male child,red shoes,white socks",
        ""
    ]

    for i, image_name in enumerate(image_files):
        image_name_with_extension = f"{''.join(image_name.split('.')[0:-1])}_{image_name.split('.')[-1]}"
        for video_name in video_files:
            video_name_with_extension = f"{''.join(video_name.split('.')[0:-1])}_{video_name.split('.')[-1]}"

            prompt = base_prompt + image_prompts[i]
            output_folder = os.path.join("test_dataset_results", image_name_with_extension, video_name_with_extension)

            video_path = os.path.join(videos_path, video_name)
            image_path = os.path.join(images_path, image_name)
            generate_anime_video(video_path, image_path, prompt, output_folder)
