import cv2
import os
import random

# 输入和输出文件夹路径
input_dir = r'./raw_data/1'
output_dir = r'./predict_data/1'
target_size = (512, 512)

# 确保输出文件夹存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取所有图片文件
image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

# 随机打乱文件列表
random.shuffle(image_files)

# 遍历随机排序后的图片文件
for i, filename in enumerate(image_files):
    # 构建完整的文件路径
    input_path = os.path.join(input_dir, filename)

    try:
        # 读取图片
        img = cv2.imread(input_path)
        if img is not None:
            # 调整图片大小
            resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            
            # 生成新的文件名，格式为 IMG_001.jpg, IMG_002.jpg, ...
            new_filename = f'IMG_{i+1:03d}.jpg'
            
            # 构建输出文件路径
            output_path = os.path.join(output_dir, new_filename)
            
            # 保存调整大小后的图片
            cv2.imwrite(output_path, resized_img)
            print(f'成功处理并保存: {output_path} (原文件: {filename})')
        else:
            print(f'无法读取图片: {input_path}')
    except Exception as e:
        print(f'处理图片 {input_path} 时出错: {e}')

print('所有图片处理完成。')
