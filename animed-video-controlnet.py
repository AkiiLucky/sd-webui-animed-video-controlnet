import io
import cv2
import base64
import random
import requests
from PIL import Image
import numpy as np
import os
from utils import sort_file_name, extract_frames, count_files_in_folder, extract_keyframes, concatenate_keyframes_to_video, generate_interpolated_frames


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

    def init_seed(self):
        self.body["seed"] = random.randint(1, 2147483647)

    def reset_seed(self, seed):
        self.body["seed"] = seed

    def get_seed(self):
        return self.body["seed"]

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
    return "images/first_reference_img_9.png"


# 逐帧风格转换 有参考
def multi_frames_transfer_reference(frames_folder, prompt, frames_num=-1):  # 仅生成前frames_num张图片，-1表示生成所有帧
    # 获取文件夹中的所有图像文件
    image_files = [f for f in os.listdir(keyframes_folder) if f.endswith(".png")]
    image_files = sort_file_name(image_files)  # 按照文件名排序确保顺序正确

    frames_count = len(image_files)
    assert frames_num <= frames_count
    if frames_num == -1:  # 生成所有帧
        frames_num = frames_count

    frames_transfer_folder = f"{frames_folder}_transfer_reference"
    os.makedirs(frames_transfer_folder, exist_ok=True)

    first_reference_img = get_first_reference_img()

    for i in range(frames_num):

        img_path = os.path.join(frames_folder, image_files[i])
        control_net = ControlnetRequest(prompt, img_path)
        control_net.build_body()
        control_net.add_reference(first_reference_img)  # 参考同一张风格图片

        if i > 0:   # 参考前一张风格图片
            control_net.add_reference(os.path.join(frames_transfer_folder, image_files[i-1]))  # 参考前一张风格帧

        output = control_net.send_request()
        result = output['images'][0]
        image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))
        image.save(os.path.join(frames_transfer_folder, image_files[i]))
        print(f"{image_files[i]} saved to {frames_transfer_folder}")


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


    video_path = "videos/video6.mp4"
    file_name = video_path.split("/")[-1]
    file_name = f"{''.join(file_name.split('.')[0:-1])}_{file_name.split('.')[-1]}"
    frames_folder = f"outputs/video/{file_name}/frames"
    keyframes_folder = f"outputs/video/{file_name}/keyframes"
    transfer_video_path = f"outputs/video/{file_name}/transfer_video.mp4"
    transfer_video_path_interpolated = f"outputs/video/{file_name}/transfer_video_interpolated.mp4"

    # prompt = "<lora:lcm_lora_v15_weights:1>,masterpiece,best quality,highres,(simple white background),1girl,solo,blonde hair,blue eyes,a basketball,basketball uniform,anime screencap,"
    # prompt = "<lora:lcm_lora_v15_weights:1>,masterpiece,best quality,highres,(simple white background),1girl,solo,white short hair,yellow eyes,a basketball,basketball uniform,anime screencap,"
    # prompt = "<lora:lcm_lora_v15_weights:1>,masterpiece,best quality,highres,(simple white background),anime screencap,a basketball, 1boy,blue basketball uniform, male focus, blue short hair, solo, shorts, blue eyes,blue sportswear, full body, male child, shoes, sneakers, looking at viewer,white socks"
    prompt = "<lora:lcm_lora_v15_weights:1>,masterpiece,best quality,highres,(((simple white background))),anime screencap,1boy,male focus,blue hair,(solo),white shorts,blue eyes,white sportswear,full body,male child,red shoes,white socks,walking,"

    # 提取视频帧
    # extract_frames(video_path, output_folder=frames_folder)    # 提取所有帧
    extract_keyframes(video_path, output_folder=keyframes_folder, frame_interval=3)   # 提取关键帧

    # 逐帧风格转换 无参考
    # multi_frames_transfer(frames_folder, prompt, frames_num=8)

    # 逐帧风格转换 有参考 生成全部帧
    # multi_frames_transfer_reference(frames_folder, prompt, frames_num=-1)

    # 逐帧风格转换 有参考 仅生成关键帧
    multi_frames_transfer_reference(keyframes_folder, prompt, frames_num=-1)

    # 组帧
    concatenate_keyframes_to_video(keyframes_folder=keyframes_folder+"_transfer_reference", output_video_path=transfer_video_path, fps=8)

    # 补帧
    generate_interpolated_frames(keyframes_folder=keyframes_folder+"_transfer_reference",  output_video_path=transfer_video_path_interpolated, fps=24)
