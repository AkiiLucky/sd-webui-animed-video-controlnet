# from PIL import Image
# import os

# def merge_images(folder_path, output_path):
#     # 读取文件夹内所有图片文件
#     images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
#     images.sort()  # 如果需要可以按文件名排序

#     # 加载所有图片
#     image_objects = [Image.open(os.path.join(folder_path, img)) for img in images]

#     # 确定单个图片的尺寸
#     if not image_objects:
#         raise ValueError("没有找到图片文件")
    
#     # 所有图片宽度之和与高度的最大值
#     total_width = sum(img.width for img in image_objects)
#     max_height = max(img.height for img in image_objects)

#     # 创建新的图片，宽度是所有图片宽度的总和，高度是最高的一张图片的高度
#     new_im = Image.new('RGB', (total_width, max_height))

#     # 将图片拼接到新创建的图片对象上
#     x_offset = 0
#     for img in image_objects:
#         new_im.paste(img, (x_offset, 0))
#         x_offset += img.width

#     # 保存合并后的图片
#     new_im.save(output_path)


# folder_path = r'F:\diffusion\sd-webui-aki\sd-webui-aki-v4.5\extensions\sd-webui-animed-video-controlnet\draw_images\4_PNG_jump_mp4\frames'  # 这里填写你的帧文件夹的路径
# output_path = f'{folder_path}\combined_image.jpg'  # 输出图片的路径

# # 调用函数
# merge_images(folder_path, output_path)


from PIL import Image
import os

from utils import sort_file_name

def merge_images(folder_path, output_path, images_per_row=10):
    # 读取文件夹中的所有图片文件
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    files = sort_file_name(files)  # 如果需要可以按文件名排序
    print(files)
    images = [Image.open(file) for file in files]
    
    if not images:
        raise ValueError("没有找到图片文件")

    # 确定单个图片的宽度和高度
    width, height = images[0].size

    # 计算合并后的总宽度和高度
    total_width = width * images_per_row
    total_height = height * (len(images) // images_per_row + (1 if len(images) % images_per_row > 0 else 0))

    # 创建一个新的空白图片用于合并
    merged_image = Image.new('RGB', (total_width, total_height))

    # 将图片按照指定的行列数合并
    x_offset = 0
    y_offset = 0
    for i, img in enumerate(images):
        merged_image.paste(img, (x_offset, y_offset))
        x_offset += width
        if (i + 1) % images_per_row == 0:
            x_offset = 0
            y_offset += height

    # 保存合并后的图片
    merged_image.save(output_path)
    print(f"图片已合并，保存路径：{output_path}")

# 使用示例
folder_path = r'F:\diffusion\sd-webui-aki\sd-webui-aki-v4.5\extensions\sd-webui-animed-video-controlnet\draw_images\walking_mp4\frames_transfer_reference'  # 这里填写你的帧文件夹的路径
output_path = f'{folder_path}\combined_image.jpg'  # 输出图片的路径

merge_images(folder_path, output_path)
