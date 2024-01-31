# coding:utf-8
import os
import cv2
import json
import numpy as np
# from googletrans import Translator
import glob
import re


def natural_sort_key(s):
    """
    按文件名的结构排序，即依次比较文件名的非数字和数字部分
    """
    # 将字符串按照数字和非数字部分分割，返回分割后的子串列表
    sub_strings = re.split(r'(\d+)', s)
    # 如果当前子串由数字组成，则将它转换为整数；否则返回原始子串
    sub_strings = [int(c) if c.isdigit() else c for c in sub_strings]
    # 根据分割后的子串列表以及上述函数的返回值，创建一个新的列表
    # 按照数字部分从小到大排序，然后按照非数字部分的字典序排序
    return sub_strings


def sort_file_name(file_list):
    sorted_file_list = sorted(file_list, key=natural_sort_key)
    return sorted_file_list

# def translate_text(text, dest_language="en"):
#     """
#     将文本从一种语言翻译成另一种语言。
#
#     :param text: 要翻译的文本。
#     :param dest_language: 目标语言的代码（例如，'en'代表英语）。
#     :return: 翻译后的文本。
#     """
#     translator = Translator()
#     translation = translator.translate(text, dest=dest_language)
#     return translation.text

def rename_files_in_directory(directory, prefix="File"):
    """
    重命名指定目录下的所有文件，添加编号前缀。

    :param directory: 包含需要重命名的文件的目录。
    :param prefix: 重命名文件时使用的前缀。
    """
    # 获取目录中的所有文件
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # 对每个文件进行重命名
    for i, file in enumerate(files, start=1):
        # 构造新的文件名
        new_name = f"{prefix}{i}{os.path.splitext(file)[1]}"
        # 构造完整的旧文件和新文件路径
        old_file = os.path.join(directory, file)
        new_file = os.path.join(directory, new_name)
        # 重命名文件
        os.rename(old_file, new_file)
        print(f"Renamed '{file}' to '{new_name}'")


def process_image(img):
    min_side = 512
    size = img.shape
    h, w = size[0], size[1]
    # 长边缩放为min_side
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w / scale), int(h / scale)
    resize_img = cv2.resize(img, (new_w, new_h))
    # 填充至min_side * min_side
    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = (min_side - new_h) / 2, (min_side - new_h) / 2, (min_side - new_w) / 2 + 1, (
                    min_side - new_w) / 2
    elif new_h % 2 != 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side - new_h) / 2 + 1, (min_side - new_h) / 2, (min_side - new_w) / 2, (
                    min_side - new_w) / 2
    elif new_h % 2 == 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side - new_h) / 2, (min_side - new_h) / 2, (min_side - new_w) / 2, (
                    min_side - new_w) / 2
    else:
        top, bottom, left, right = (min_side - new_h) / 2 + 1, (min_side - new_h) / 2, (min_side - new_w) / 2 + 1, (
                    min_side - new_w) / 2
    pad_img = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])  # 从图像边界向上,下,左,右扩的像素数目
    # print pad_img.shape
    # cv2.imwrite("after-" + os.path.basename(filename), pad_img)
    return pad_img


# 切割视频帧序列
def extract_frames(video_path, output_folder):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # 读取视频的帧
    frame_count = 0
    while True:
        ret, frame = cap.read()
        # 如果没有更多的帧，则退出循环
        if not ret:
            break
        # 保存每一帧为 PNG 文件
        frame_path = os.path.join(output_folder, f'{frame_count}.png')
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    # 关闭视频文件
    cap.release()
    print(f"{frame_count} frames extracted and saved to '{output_folder}'.")



def find_video_files(directory):
    """
    获取指定目录中所有视频文件的列表。

    :param directory: 要搜索的目录。
    :return: 视频文件列表。
    """
    # 定义常见的视频文件扩展名
    video_extensions = (
        ".mp4",  # MPEG-4 Part 14
        ".avi",  # Audio Video Interleave
        ".mov",  # QuickTime File Format
        ".wmv",  # Windows Media Video
        ".flv",  # Flash Video
        ".mkv",  # Matroska Video
        ".webm",  # WebM
        ".mpeg",  # Moving Picture Experts Group
        ".mpg",  # Moving Picture Experts Group
        ".rmvb",  # RealMedia Variable Bitrate
        ".m4v",  # MPEG-4 Video
        ".3gp",  # 3GPP
        ".ts",  # MPEG Transport Stream
        ".vob",  # DVD-Video Object
        ".ogv",  # Ogg Video
        ".asf",  # Advanced Systems Format
        ".divx",  # DivX
        ".f4v",  # Flash MP4 Video
        ".m2ts",  # MPEG-2 Transport Stream
        ".m2v",  # MPEG-2 Video
        ".mxf",  # Material Exchange Format
        ".nsv",  # Nullsoft Streaming Video
        ".rm",  # RealMedia
        ".svi",  # Samsung Video File
        ".vob",  # Video Object
        ".webm",  # WebM
        ".wmv",  # Windows Media Video
        ".yuv"  # YUV Encoded Video File
    )
    # 遍历目录，获取所有视频文件
    video_files = [f for f in os.listdir(directory) 
                   if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(video_extensions)]
    return video_files


# # 切割视频帧序列
# def extract_frames(video_path, output_folder):
#     # 打开视频文件
#     cap = cv2.VideoCapture(video_path)
#
#     # 检查视频是否成功打开
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         exit()
#
#     # 读取视频的帧
#     frame_count = 0
#     while True:
#         ret, frame = cap.read()
#         # 如果没有更多的帧，则退出循环
#         if not ret:
#             break
#         # 保存每一帧为 PNG 文件
#         frame_path = os.path.join(output_folder, f'{frame_count}.png')
#         cv2.imwrite(frame_path, frame)
#         frame_count += 1
#     # 关闭视频文件
#     cap.release()
#     print(f"{frame_count} frames extracted and saved to '{output_folder}'.")
#     return frame_count


# 切割视频帧序列 并resize为512*512
def extract_frames(video_path, output_folder):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    target_size = 512  # 目标尺寸

    # 读取视频的帧
    frame_count = 0
    while True:
        ret, frame = cap.read()
        # 如果没有更多的帧，则退出循环
        if not ret:
            break

        # 获取原始帧的尺寸
        h, w = frame.shape[:2]

        # 计算缩放比例
        scale = min(target_size / h, target_size / w)
        nh, nw = int(h * scale), int(w * scale)

        # 缩放帧
        frame_resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

        # 创建一个512x512的黑色背景
        new_frame = np.zeros((target_size, target_size, 3), dtype=np.uint8)

        # 计算放置缩放后图像的位置
        top = (target_size - nh) // 2
        left = (target_size - nw) // 2

        # 将缩放后的图像放置在背景上
        new_frame[top:top+nh, left:left+nw] = frame_resized

        # 保存每一帧为 PNG 文件
        frame_path = os.path.join(output_folder, f'{frame_count}.png')
        cv2.imwrite(frame_path, new_frame)
        frame_count += 1

    # 关闭视频文件
    cap.release()
    print(f"{frame_count} frames extracted and saved to '{output_folder}'.")
    return frame_count


def remove_file_extension(file_name):
    """
    去除文件名的后缀。

    :param file_name: 完整的文件名。
    :return: 去除后缀的文件名。
    """
    return os.path.splitext(file_name)[0]


def preprocess_videos(original_videos_folder, dataset_folder):
    video_files = find_video_files(original_videos_folder)
    video_files = sort_file_name(video_files)
    os.makedirs(dataset_folder, exist_ok=True)
    prompts = []

    total_videos = 0
    total_frames = 0
    for i, video_name in enumerate(video_files):
        original_video_path = os.path.join(original_videos_folder, video_name)
        # video_name_en = translate_text(remove_file_extension(video_name), dest_language="en")
        frames_folder = os.path.join(dataset_folder, f"{remove_file_extension(video_name)}")
        os.makedirs(frames_folder, exist_ok=True)
        frame_count = extract_frames(video_path=original_video_path, output_folder=frames_folder)

        frames_folder_name = os.path.basename(frames_folder)
        for frame_id in range(frame_count-1):
            prompt = {
                "source": f"{frames_folder_name}/{frame_id}.png",
                "target": f"{frames_folder_name}/{frame_id+1}.png",
                "prompt": f"masterpiece,best quality,simple background,anime,interframe reference"
            }
            prompts.append(prompt)

        total_videos = total_videos + 1
        total_frames = total_frames + frame_count

    prompt_file = f"{dataset_folder}/prompt.json"
    with open(prompt_file, "w") as file:
        for record in prompts:
            json.dump(record, file)
            file.write("\n")  # 每条记录后添加换行符

    return total_videos, total_frames


if __name__ == "__main__":
    # original_videos_folder = "original_videos_val"
    # rename_files_in_directory(original_videos_folder, prefix='video_')

    # 训练集
    original_videos_folder = "original_videos_train"
    dataset_name = "my_dataset_train"
    dataset_folder = f"{dataset_name}"
    total_videos, total_frames = preprocess_videos(original_videos_folder, dataset_folder)
    print("train dataset:")
    print(f"total_videos = {total_videos}, total_frames = {total_frames}")

    # 验证集
    original_videos_folder = "original_videos_val"
    dataset_name = "my_dataset_val"
    dataset_folder = f"{dataset_name}"
    total_videos, total_frames = preprocess_videos(original_videos_folder, dataset_folder)
    print("val dataset:")
    print(f"total_videos = {total_videos}, total_frames = {total_frames}")

