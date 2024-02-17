import cv2
import numpy as np
import os
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


# 切割视频帧序列
def extract_frames(video_path, output_folder):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
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


# 统计文件夹中的文件数量
def count_files_in_folder(folder_path):
    try:
        # 使用 os.listdir() 获取文件夹中的所有文件和子文件夹
        file_list = os.listdir(folder_path)
        # 使用列表推导式过滤出文件，而不是子文件夹
        files = [file for file in file_list if os.path.isfile(os.path.join(folder_path, file))]
        # 返回文件数量
        return len(files)
    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")
        return None


# 光流法提取视频关键帧
def extract_keyframes(video_path, output_folder, frame_interval=3, dynamic_threshold_decay=0.9):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    keyframe_count = 0
    prev_frame = None
    dynamic_threshold = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 将当前帧转换为灰度图
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 在第一帧之后才开始计算光流
        if prev_frame is not None and frame_count % frame_interval == 0:
            # 计算光流
            flow = cv2.calcOpticalFlowFarneback(prev_frame, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # 计算光流矢量的幅值
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

            # 动态调整阈值
            if dynamic_threshold is None:
                dynamic_threshold = np.mean(magnitude)
            else:
                dynamic_threshold = dynamic_threshold * dynamic_threshold_decay + np.mean(magnitude) * (
                            1 - dynamic_threshold_decay)

            # 使用动态阈值判断光流幅值是否足够大，将该帧标记为关键帧
            if np.sum(magnitude > dynamic_threshold) > 0:
                frame_filename = f"{output_folder}/{frame_count}.png"
                cv2.imwrite(frame_filename, frame)
                #print(f"Saved keyframe: {frame_filename}, Dynamic Threshold: {dynamic_threshold}")
                keyframe_count += 1

        prev_frame = frame_gray
        frame_count += 1

    cap.release()
    print(f"{keyframe_count} keyframes extracted and saved to '{output_folder}'.")


# 组帧成视频
def concatenate_keyframes_to_video(keyframes_folder, output_video_path, fps=24):
    # 获取关键帧文件夹中的所有图像文件
    image_files = [f for f in os.listdir(keyframes_folder) if f.endswith(".png")]
    image_files = sort_file_name(image_files)  # 按照文件名排序确保顺序正确

    # 获取第一帧的图像大小
    first_frame = cv2.imread(os.path.join(keyframes_folder, image_files[0]))
    height, width, _ = first_frame.shape

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用MP4编解码器
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 将关键帧逐一写入视频
    for image_file in image_files:
        frame = cv2.imread(os.path.join(keyframes_folder, image_file))
        video_writer.write(frame)

    # 释放视频写入器
    video_writer.release()


# 视频补帧
def generate_interpolated_frames(keyframes_folder, output_video_path, fps=24):
    # 获取关键帧文件夹中的所有图像文件
    image_files = [f for f in os.listdir(keyframes_folder) if f.endswith(".png")]
    image_files = sort_file_name(image_files)   # 按照文件名排序确保顺序正确

    # 获取第一帧的图像大小
    first_frame = cv2.imread(os.path.join(keyframes_folder, image_files[0]))
    height, width, _ = first_frame.shape

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用MP4编解码器
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 插值生成过渡帧并写入视频
    for i in range(len(image_files) - 1):
        current_frame = cv2.imread(os.path.join(keyframes_folder, image_files[i]))
        next_frame = cv2.imread(os.path.join(keyframes_folder, image_files[i + 1]))

        # 将当前帧转换为灰度图
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        # 使用光流法计算光流矢量
        flow = cv2.calcOpticalFlowFarneback(current_frame_gray, next_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # 插值生成过渡帧
        for alpha in np.linspace(0, 1, fps):
            # 使用线性插值生成过渡帧
            interpolated_frame = cv2.addWeighted(current_frame, 1 - alpha, next_frame, alpha, 0)

            # 将光流应用于插值帧的每个通道
            for c in range(3):
                interpolated_frame[:, :, c] += (flow[:, :, 0] * alpha).astype(np.uint8)

            video_writer.write(interpolated_frame.astype(np.uint8))

    # 写入最后一个关键帧
    last_frame = cv2.imread(os.path.join(keyframes_folder, image_files[-1]))
    for _ in range(fps):
        video_writer.write(last_frame)

    # 释放视频写入器
    video_writer.release()


if __name__ == "__main__":
    # 使用示例
    video_path = "./videos/video2.mp4"  # 替换为实际视频文件路径
    output_folder = "key_frame_extract_result"
    extract_keyframes(video_path, output_folder)

    keyframes_folder = output_folder
    output_video_path = "output_video.mp4"  # 替换为输出视频文件路径
    concatenate_keyframes_to_video(keyframes_folder, output_video_path)

    keyframes_folder = output_folder
    output_video_path = "output_interpolated_video_optical_flow.mp4"  # 替换为输出视频文件路径
    generate_interpolated_frames(keyframes_folder, output_video_path)
