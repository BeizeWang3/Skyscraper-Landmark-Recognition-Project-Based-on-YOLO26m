import cv2
import os
import numpy as np

def enhance_images():
    """
    对指定文件夹中的图像应用三种不同的数据增强方法，
    并将结果以新的命名规则保存在各自的文件夹中。
    """
    # --- 1. 配置 ---
    # 输入文件夹
    input_dir = r'E:\DIP\MyLab\project\resized_data'

    # 三种增强方法对应的输出文件夹
    output_grayscale_dir = r'E:\DIP\MyLab\project\augmented_grayscale'
    output_hist_eq_dir = r'E:\DIP\MyLab\project\augmented_hist_equalized'
    output_sharpen_dir = r'E:\DIP\MyLab\project\augmented_sharpened'

    # --- 2. 初始化 ---
    # 确保所有输出文件夹都存在
    os.makedirs(output_grayscale_dir, exist_ok=True)
    os.makedirs(output_hist_eq_dir, exist_ok=True)
    os.makedirs(output_sharpen_dir, exist_ok=True)
    print("输出文件夹已准备就绪。")

    # 获取所有图片文件
    try:
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"错误：在 '{input_dir}' 文件夹中没有找到任何图片。")
            return
    except FileNotFoundError:
        print(f"错误：输入文件夹 '{input_dir}' 不存在。")
        return

    print(f"找到 {len(image_files)} 张图片，开始进行数据增强...")

    # --- 3. 图像处理与保存 ---
    for i, filename in enumerate(image_files):
        # 构建完整输入路径
        input_path = os.path.join(input_dir, filename)
        
        # 读取原始图片
        original_img = cv2.imread(input_path)
        if original_img is None:
            print(f"警告：无法读取图片 {input_path}，已跳过。")
            continue

        # 分离文件名和扩展名以创建新名称
        basename, ext = os.path.splitext(filename)

        # --- 方法 1: 灰度变换 ---
        grayscale_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        grayscale_img_3ch = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2BGR)
        new_filename_01 = f"{basename}_01{ext}"
        grayscale_output_path = os.path.join(output_grayscale_dir, new_filename_01)
        cv2.imwrite(grayscale_output_path, grayscale_img_3ch)

        # --- 方法 2: 直方图均衡化 (在V通道上) ---
        hsv_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)
        v_equalized = cv2.equalizeHist(v)
        hsv_equalized_img = cv2.merge([h, s, v_equalized])
        hist_eq_img = cv2.cvtColor(hsv_equalized_img, cv2.COLOR_HSV2BGR)
        new_filename_02 = f"{basename}_02{ext}"
        hist_eq_output_path = os.path.join(output_hist_eq_dir, new_filename_02)
        cv2.imwrite(hist_eq_output_path, hist_eq_img)

        # --- 方法 3: 空间域滤波 - 图像锐化 (非锐化掩模) ---
        blurred_img = cv2.GaussianBlur(original_img, (0, 0), 5)
        sharpened_img = cv2.addWeighted(original_img, 1.5, blurred_img, -0.5, 0)
        new_filename_03 = f"{basename}_03{ext}"
        sharp_output_path = os.path.join(output_sharpen_dir, new_filename_03)
        cv2.imwrite(sharp_output_path, sharpened_img)
        
        # 打印进度
        print(f"  ({i+1}/{len(image_files)}) 已处理: {filename} -> {new_filename_01}, {new_filename_02}, {new_filename_03}")

    print("\n所有图片数据增强完成！")
    print(f"灰度变换结果保存在: {output_grayscale_dir}")
    print(f"直方图均衡化结果保存在: {output_hist_eq_dir}")
    print(f"图像锐化结果保存在: {output_sharpen_dir}")


if __name__ == '__main__':
    enhance_images()