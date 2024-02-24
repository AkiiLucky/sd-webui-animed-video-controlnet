# sd-webui-animed-video-controlnet
sd-webui 扩展插件， 用于实现可控动漫视频生成（开发中）

## 底层原理
stable diffusion + controlnet（多个）

- stable diffusion model: anything v5
- controlnet_1: openpose
- controlnet_2: lineart
- controlnet_3: interframes_reference_controlnet （我自己训练的帧间一致性参考控制网络）

![multicontrolnet.drawio.png](images%2Fmulticontrolnet.drawio.png)

## 功能介绍
- 输入：一张参考动漫人物图像 + 一个真人实拍的动作视频
- 输出：一个动漫人物的动作视频

![web_ui.png](images%2Fweb_ui.png)

## 使用方法
- 安装sd-webui
- 安装sd-webui-controlnet插件
- 安装本插件

## 演示视频
[demo.mp4](demo.mp4)