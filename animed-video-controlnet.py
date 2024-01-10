import io
import cv2
import base64
import random
import requests
from PIL import Image
import numpy as np
import os

"""
    To use this example make sure you've done the following steps before executing:
    1. Ensure automatic1111 is running in api mode with the controlnet extension. 
       Use the following command in your terminal to activate:
            ./webui.sh --no-half --api
    2. Validate python environment meet package dependencies.
       If running in a local repo you'll likely need to pip install cv2, requests and PIL 
"""

LOWVRAM = False
PIXEL_PERFECT = False

class ControlnetRequest:
    def __init__(self, prompt, img_path):
        self.url = "http://localhost:7860/sdapi/v1/txt2img"
        self.prompt = prompt
        self.img_path = img_path
        self.body = None

    def build_body(self):
        self.body = {
            "prompt": self.prompt,
            "negative_prompt": "(worst quality, low quality:1.2),bad body,long body,missing limb,disconnected limbs,extra legs,bad feet,extra arms,floating limbs,poorly drawn face,mutated hands,extra limb,poorly drawn hands,too many fingers,fused fingers,missing fingers,bad hands,mutated hands and fingers,malformed hands,deformed,mutated,disfigured,malformed limbs,cross-eyed,ugly,NSFW,",
            "batch_size": 1,
            "steps": 8,
            "cfg_scale": 1.5,
            # "width": 512,
            # "height": 512,
            "sampler_index": "LCM",
            "seed": 4281519988,
            # "seed_resize_from_h": 4281519988,
            # "seed_resize_from_w": 4281519988,
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                            "enabled": True,
                            "module": "openpose_full",
                            "model": "control_v11p_sd15_openpose_fp16 [73c2b67d]",
                            "weight": 1,
                            "image": self.read_image(),
                            "resize_mode": 1,
                            "lowvram": LOWVRAM,
                            "processor_res": 512,
                            "threshold_a": 512,
                            "threshold_b": 512,
                            "guidance_start": 0.0,
                            "guidance_end": 1.0,
                            "control_mode": 0,  # 0均衡 1更偏向提示词 2更偏向 ControlNet
                            "pixel_perfect": PIXEL_PERFECT
                        },
                        {
                            "enabled": True,
                            "module": "lineart_anime",
                            "model": "control_v11p_sd15s2_lineart_anime_fp16 [c58f338b]",
                            "weight": 1,
                            "image": self.read_image(),
                            "resize_mode": 1,
                            "lowvram": LOWVRAM,
                            "processor_res": 512,
                            "threshold_a": 512,
                            "threshold_b": 512,
                            "guidance_start": 0.0,
                            "guidance_end": 1.0,
                            "control_mode": 1,  # 0均衡 1更偏向提示词 2更偏向 ControlNet
                            "pixel_perfect": PIXEL_PERFECT
                        }
                    ]
                }
            }
        }
        self.init_seed()

    def init_seed(self):
        self.body["seed"] = random.randint(1, 2147483647)

    def reset_seed(self, seed):
        self.body["seed"] = seed

    def get_seed(self):
        return self.body["seed"]

    def add_reference(self, reference_img):
        self.reference_img_path = reference_img
        self.body["alwayson_scripts"]["controlnet"]["args"].append(
            {
                "enabled": True,
                "module": "reference_only",
                "model": "none",
                "weight": 1,
                "image": self.read_reference_image(),
                "resize_mode": 1,
                "lowvram": LOWVRAM,
                "processor_res": 512,
                "threshold_a": 512,
                "threshold_b": 512,
                "guidance_start": 0.0,
                "guidance_end": 1.0,
                "control_mode": 0,
                "pixel_perfect": PIXEL_PERFECT
                # r'Fidelity': 1.0 #改了precessor.py中的源码
            }
        )

    def send_request(self):
        response = requests.post(url=self.url, json=self.body)
        return response.json()

    def read_image(self):
        img = cv2.imread(self.img_path)
        retval, bytes = cv2.imencode('.png', img)
        encoded_image = base64.b64encode(bytes).decode('utf-8')
        return encoded_image

    def read_reference_image(self):
        img = cv2.imread(self.reference_img_path)
        retval, bytes = cv2.imencode('.png', img)
        encoded_image = base64.b64encode(bytes).decode('utf-8')
        return encoded_image



# 单帧风格转换
def single_frame_transfer(img_path, prompt):
    control_net = ControlnetRequest(prompt, img_path)
    control_net.build_body()
    output = control_net.send_request()
    result = output['images'][0]
    return result


# 切割视频帧序列
def split_frames(video_path):
    file_name = video_path.split("/")[-1]
    file_name = f"{''.join(file_name.split('.')[0:-1])}_{file_name.split('.')[-1]}"
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # 创建保存帧的文件夹
    frames_folder = f"outputs/video/{file_name}/frames"
    os.makedirs(frames_folder, exist_ok=True)

    # 读取视频的帧
    frame_count = 0
    while True:
        ret, frame = cap.read()
        # 如果没有更多的帧，则退出循环
        if not ret:
            break
        # 保存每一帧为 PNG 文件
        frame_path = os.path.join(frames_folder, f'{frame_count}.png')
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    # 关闭视频文件
    cap.release()
    print(f"{frame_count} frames extracted and saved to '{frames_folder}'.")
    return frames_folder


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


# 逐帧风格转换 无参考
def multi_frames_transfer(frames_folder, prompt, frames_num):
    count = count_files_in_folder(frames_folder)
    assert frames_num <= count
    frames_transfer_folder = f"{frames_folder}_transfer"
    os.makedirs(frames_transfer_folder, exist_ok=True)

    for i in range(frames_num):
        img_path = f"{frames_folder}/{i}.png"
        control_net = ControlnetRequest(prompt, img_path)
        control_net.build_body()
        output = control_net.send_request()
        result = output['images'][0]
        image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))
        image.save(f"{frames_transfer_folder}/{i}.png")


def get_first_reference_img():
    return "images/first_reference_img_7.png"

# 逐帧风格转换 有参考
def multi_frames_transfer_reference(frames_folder, prompt, frames_num=-1):
    count = count_files_in_folder(frames_folder)
    assert frames_num <= count
    if frames_num == -1:  # 渲染所有帧
        frames_num = count
    frames_transfer_folder = f"{frames_folder}_transfer_reference"
    os.makedirs(frames_transfer_folder, exist_ok=True)

    first_reference_img = get_first_reference_img()

    for i in range(frames_num):
        img_path = f"{frames_folder}/{i}.png"
        control_net = ControlnetRequest(prompt, img_path)
        control_net.build_body()

        # if i != 0:
        #     control_net.add_reference(f"{frames_transfer_folder}/{i-1}.png")  # 参考第i-1张风格帧
        # else:
        #     control_net.add_reference(first_reference_img)  # 参考第1张风格图片

        control_net.add_reference(first_reference_img)  # 参考同一张风格图片

        if i != 0:   # 参考前一张风格图片
            control_net.add_reference(f"{frames_transfer_folder}/{i-1}.png")  # 参考第i-1张风格帧

        output = control_net.send_request()
        result = output['images'][0]
        image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))
        image.save(f"{frames_transfer_folder}/{i}.png")
        print(f"{i}.png saved to {frames_transfer_folder}")


if __name__ == '__main__':
    # # test single_frame_transfer()
    # path = 'images/img2.png'
    # prompt = '(anime screencap), white background, masterpiece, best quality' \
    #          ' actual 8K portrait photo of gareth person,' \
    #          ' <lora:lcm_lora_v15_weights:1>'
    # result = single_frame_transfer(path, prompt)
    # image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))
    # image.show()
    # file_name = path.split("/")[-1]
    # file_name = f"{''.join(file_name.split('.')[0:-1])}_{file_name.split('.')[-1]}"
    # image.save(f"outputs/images/file_name/{file_name}.png")


    video_path = "videos/video2.mp4"
    prompt = "<lora:lcm_lora_v15_weights:1>,masterpiece,best quality,highres,(simple white background),1girl,solo,blonde hair,blue eyes,a basketball,basketball uniform,anime screencap,"
    # prompt = "<lora:lcm_lora_v15_weights:1>,masterpiece,best quality,highres,(simple white background),1girl,solo,white short hair,yellow eyes,a basketball,basketball uniform,anime screencap,"

    # 切割视频帧
    frames_folder = split_frames(video_path)
    # 逐帧风格转换 无参考
    # multi_frames_transfer(frames_folder, prompt, frames_num=8)
    # 逐帧风格转换 有参考
    multi_frames_transfer_reference(frames_folder, prompt, frames_num=-1)


