#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：sd-webui-animed-video-controlnet 
@File    ：test_all_video_image_pairs.py
@Author  ：AkiiLucky
@Date    ：2024/2/17 11:46 
'''

import os
import cv2
import json
import numpy as np
from skimage.metrics import structural_similarity
from sklearn.metrics.pairwise import cosine_similarity
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgba2rgb
from animed_video_controlnet import generate_anime_video
from utils import sort_file_name


def compute_ssim(imageA_path, imageB_path):
    # 加载两幅图像
    imageA = cv2.imread(imageA_path)
    imageB = cv2.imread(imageB_path)

    # 转换为灰度图像
    imageA_gray = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    imageB_gray = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # 计算SSIM
    ssim_value = structural_similarity(imageA_gray, imageB_gray)
    return ssim_value


def compute_ssim_mean(image_path, frames_transfer_folder):
    frames_transfer = [f for f in os.listdir(frames_transfer_folder) if f.endswith(".png")]
    ssim_1 = 0  # 全局参考图片与每个风格帧的SSIM
    ssim_2 = 0  # 每个相邻风格帧的SSIM
    for frame_id, frame_name in enumerate(frames_transfer):
        frame_path = os.path.join(frames_transfer_folder, frame_name)
        ssim_1 += compute_ssim(image_path, frame_path)
        if frame_id < len(frames_transfer) - 1:
            next_frame_path = os.path.join(frames_transfer_folder, frames_transfer[frame_id + 1])
            ssim_2 += compute_ssim(frame_path, next_frame_path)
    ssim_1 /= len(frames_transfer)
    ssim_2 /= len(frames_transfer) - 1
    return ssim_1, ssim_2


def compute_cos_sim(imageA_path, imageB_path):
    # 读取两张图像
    imageA = imread(imageA_path)
    imageB = imread(imageB_path)

    # 如果imageA是4通道的，将其转换为3通道RGB
    if imageA.shape[2] == 4:
        imageA = rgba2rgb(imageA)

    # 确保两张图像大小一致
    imageA = resize(imageA, (imageB.shape[0], imageB.shape[1]), anti_aliasing=True)

    # 将图像展平成向量
    vector1 = imageA.flatten().reshape(1, -1)
    vector2 = imageB.flatten().reshape(1, -1)

    # 计算余弦相似度
    cos_sim = cosine_similarity(vector1, vector2)
    return cos_sim[0][0]


def compute_cos_sim_mean(image_path, frames_transfer_folder):
    frames_transfer = [f for f in os.listdir(frames_transfer_folder) if f.endswith(".png")]
    cos_sim_1 = 0  # 全局参考图片与每个风格帧的余弦相似度
    cos_sim_2 = 0  # 每个相邻风格帧的余弦相似度
    for frame_id, frame_name in enumerate(frames_transfer):
        frame_path = os.path.join(frames_transfer_folder, frame_name)
        cos_sim_1 += compute_cos_sim(image_path, frame_path)
        if frame_id < len(frames_transfer) - 1:
            next_frame_path = os.path.join(frames_transfer_folder, frames_transfer[frame_id + 1])
            cos_sim_2 += compute_cos_sim(frame_path, next_frame_path)
    cos_sim_1 /= len(frames_transfer)
    cos_sim_2 /= len(frames_transfer) - 1
    return cos_sim_1, cos_sim_2


if __name__ == '__main__':
    images_folder = os.path.join("test_dataset", "images")
    videos_folder = os.path.join("test_dataset", "videos")
    image_files = [f for f in os.listdir(images_folder) if f.endswith(".png")]
    image_files = sort_file_name(image_files)  # 按照文件名排序确保顺序正确
    video_files = [f for f in os.listdir(videos_folder) if f.endswith(".mp4")]
    video_files = sort_file_name(video_files)  # 按照文件名排序确保顺序正确

    base_prompt = "<lora:lcm_lora_v15_weights:1>,masterpiece,best quality,highres,(simple white background),anime screencap,"
    image_prompts = [
        "1boy,male focus,blue hair,(solo),white shorts,blue eyes,white sportswear,full body,male child,red shoes,white socks",
        "1girl, solo, puffy long sleeves, blonde hair, blue eyes, long hair, white background, closed mouth, long sleeves, virtual youtuber, simple background, white socks, bangs, puffy sleeves, socks, shoes, standing, full body, red footwear, frilled socks, blush, bobby socks, skirt, shirt, smile, twintails, looking at viewer, open clothes, ribbon, center frills, hair ribbon, frills, pink footwear, white shirt, jacket, bow, red ribbon, open jacket, brown skirt, very long hair, sailor collar, pink jacket, sleeves past wrists",
        "1girl,solo,socks,dress,school uniform,open mouth,smile,white background,simple background,kneehighs,full body,shoes,white socks,mary janes,twintails,ahoge,pinafore dress,waving,short sleeves,:d,double bun,hair bun,short hair,ribbon,red hair,red eyes,",
        "solo,1boy,male focus,white hair,red eyes,shirt,grey pants,full body,black t-shirt,",
        "a certain high school uniform,1boy,kamijou touma,solo,male focus,spiked hair,school uniform,black hair,shirt,full body,pants,shoes,white shirt,simple background,short sleeves,black eyes,sneakers,black pants,closed mouth,black eyes,white shoes,",
        "solo,1boy,school uniform,male focus,glasses,gakuran,black hair,full body,",
        "sawamura spencer eriri,1girl,solo,thighhighs,skirt,long hair,black thighhighs,blonde hair,blue eyes,twintails,school uniform,full body,pleated skirt,white background,ribbon,floating hair,shoes,hair ribbon,miniskirt,black ribbon,black skirt,loafers,long sleeves,bangs,blush,blue shirt,",
        "1girl,solo,thighhighs,skirt,long hair,school uniform,bag,zettai ryouiki,ponytail,white background,looking at viewer,white shirt,full body,shoes,shirt,open mouth,blonde hair,black thighhighs,short sleeves,loafers,blue skirt,midriff,neckerchief,smile,navel,serafuku,brown footwear,miniskirt,sailor collar,orange eyes,bow,pleated skirt,ribbon,bracelet,standing,crop top,",
        "katou megumi,1girl,solo,skirt,bangs,short hair,school uniform,socks,brown hair,full body,white background,shoes,brown eyes,simple background,black socks,pleated skirt,loafers,kneehighs,looking back,long sleeves,",
        "1girl,solo,red eyes,virtual youtuber,thighhighs,dress,long hair,flower,single thighhigh,smile,ahoge,full body,white background,hair ornament,very long hair,hair flower,bangs,simple background,pink footwear,white hair,shoes,blush,pink dress,pink flower,looking at viewer,white dress,open mouth,white thighhighs,ribbon,white flower,animal ears,:d,frilled dress,asymmetrical legwear,",
    ]

    # 生成全部动漫风格视频
    for i, image_name in enumerate(image_files):
        image_name_with_extension = f"{''.join(image_name.split('.')[0:-1])}_{image_name.split('.')[-1]}"
        for video_name in video_files:
            video_name_with_extension = f"{''.join(video_name.split('.')[0:-1])}_{video_name.split('.')[-1]}"
            prompt = base_prompt + image_prompts[i]
            output_folder = os.path.join("test_dataset_results", image_name_with_extension, video_name_with_extension)
            video_path = os.path.join(videos_folder, video_name)
            image_path = os.path.join(images_folder, image_name)
            generate_anime_video(video_path, image_path, prompt, output_folder)

    # 定量评估全部动漫风格视频（三组对照组）
    # 有全局参考 无帧间参考 生成全部帧 frames_transfer
    # 有全局参考 有帧间参考 生成全部帧 frames_transfer_reference
    # 有全局参考 有帧间参考 生成关键帧 keyframes_transfer_reference
    result_dict = {
        "frames_transfer": {
            "ssim_1": [],
            "ssim_2": [],
            "ssim": [],
            "cos_sim_1": [],
            "cos_sim_2": [],
            "cos_sim": [],
        },
        "frames_transfer_reference": {
            "ssim_1": [],
            "ssim_2": [],
            "ssim": [],
            "cos_sim_1": [],
            "cos_sim_2": [],
            "cos_sim": [],
        },
        "keyframes_transfer_reference": {
            "ssim_1": [],
            "ssim_2": [],
            "ssim": [],
            "cos_sim_1": [],
            "cos_sim_2": [],
            "cos_sim": [],
        }
    }

    for i, image_name in enumerate(image_files):
        image_name_with_extension = f"{''.join(image_name.split('.')[0:-1])}_{image_name.split('.')[-1]}"
        for video_name in video_files:
            video_name_with_extension = f"{''.join(video_name.split('.')[0:-1])}_{video_name.split('.')[-1]}"
            output_folder = os.path.join("test_dataset_results", image_name_with_extension, video_name_with_extension)
            image_path = os.path.join(images_folder, image_name)
            for frames_transfer_folder_name in result_dict.keys():
                frames_transfer_folder = os.path.join(output_folder, frames_transfer_folder_name)
                # 计算ssim
                ssim_1, ssim_2 = compute_ssim_mean(image_path, frames_transfer_folder)
                ssim = 0.5 * ssim_1 + 0.5 * ssim_2
                result_dict[frames_transfer_folder_name]["ssim_1"].append(ssim_1)
                result_dict[frames_transfer_folder_name]["ssim_2"].append(ssim_2)
                result_dict[frames_transfer_folder_name]["ssim"].append(ssim)
                # 计算cos_sim
                cos_sim_1, cos_sim_2 = compute_cos_sim_mean(image_path, frames_transfer_folder)
                cos_sim = 0.5 * cos_sim_1 + 0.5 * cos_sim_2
                result_dict[frames_transfer_folder_name]["cos_sim_1"].append(cos_sim_1)
                result_dict[frames_transfer_folder_name]["cos_sim_2"].append(cos_sim_2)
                result_dict[frames_transfer_folder_name]["cos_sim"].append(cos_sim)
                print(image_name, video_name, frames_transfer_folder_name, "已完成评估")
                print(f"ssim: {ssim}, cos_sim: {cos_sim}")

    for frames_transfer_folder_name in result_dict.keys():
        result_dict[frames_transfer_folder_name]["ssim_1_mean"] = np.mean(result_dict[frames_transfer_folder_name]["ssim_1"])
        result_dict[frames_transfer_folder_name]["ssim_2_mean"] = np.mean(result_dict[frames_transfer_folder_name]["ssim_2"])
        result_dict[frames_transfer_folder_name]["ssim_mean"] = np.mean(result_dict[frames_transfer_folder_name]["ssim"])
        result_dict[frames_transfer_folder_name]["cos_sim_1_mean"] = np.mean(result_dict[frames_transfer_folder_name]["cos_sim_1"])
        result_dict[frames_transfer_folder_name]["cos_sim_2_mean"] = np.mean(result_dict[frames_transfer_folder_name]["cos_sim_2"])
        result_dict[frames_transfer_folder_name]["cos_sim_mean"] = np.mean(result_dict[frames_transfer_folder_name]["cos_sim"])

    print(result_dict)

    with open("test_dataset_results/test_results.json", 'w') as file:
        # 使用json.dump()函数将字典保存为JSON格式的文件
        # ensure_ascii=False 参数支持写入非ASCII字符，例如中文
        # indent 参数用于美化输出，使JSON文件易于阅读
        json.dump(result_dict, file, ensure_ascii=False, indent=4)


