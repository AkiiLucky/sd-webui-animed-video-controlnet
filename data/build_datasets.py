import os
import cv2

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
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mpg', '.mpeg', '.m4v')
    
    # 遍历目录，获取所有视频文件
    video_files = [f for f in os.listdir(directory) 
                   if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(video_extensions)]
    
    return video_files


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


def preprocess_videos(original_videos_folder, dataset_folder):
    video_files = find_video_files(original_videos_folder)
    temp_folder = "temp"
    os.makedirs(temp_folder, exist_ok=True)

    for i, video_name in enumerate(video_files):
        video_path = os.path.join(original_videos_folder, video_name)
        frames_folder = os.path.join(temp_folder, f"{video_name.replace('.','_')}")
        os.makedirs(frames_folder, exist_ok=True)
        extract_frames(video_path=video_path, output_folder=frames_folder)
        


if __name__ == "__main__":

    original_videos_folder = "original_videos"
    rename_files_in_directory(directory=original_videos_folder, prefix='video')

    dataset_name = "minimiku"

    dataset_folder = f"./{dataset_name}"
    dataset_inputs_folder = f"./{dataset_name}/inputs"
    dataset_outputs_folder = f"./{dataset_name}/outputs"

    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(dataset_inputs_folder, exist_ok=True)
    os.makedirs(dataset_outputs_folder, exist_ok=True)

    preprocess_videos(original_videos_folder, dataset_folder)
