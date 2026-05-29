import os
import shutil
from ultralytics import YOLO
from multiprocessing import freeze_support

def save_predictions_as_yolo_dataset(results, output_dir, conf_threshold=0.5):
    """
    将模型预测结果保存为新的YOLO格式训练数据集。

    :param results: YOLO模型的预测结果。
    :param output_dir: 新数据集的根目录。
    :param conf_threshold: 置信度阈值，只保存高于此值的预测。
    """
    # --- 创建输出目录结构 ---
    images_output_dir = os.path.join(output_dir, 'images')
    labels_output_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)
    print(f"新的数据集将保存在: '{output_dir}'")

    # --- 遍历每张图片的预测结果 ---
    processed_count = 0
    for result in results:
        # 获取原始图片路径和文件名
        original_image_path = result.path
        image_filename = os.path.basename(original_image_path)
        label_filename = os.path.splitext(image_filename)[0] + '.txt'

        # 定义新文件的输出路径
        output_image_path = os.path.join(images_output_dir, image_filename)
        output_label_path = os.path.join(labels_output_dir, label_filename)

        # --- 复制原始图片 ---
        shutil.copy(original_image_path, output_image_path)

        # --- 生成并保存YOLO格式的标签文件 ---
        with open(output_label_path, 'w') as f:
            # 获取归一化的边界框坐标 (x_center, y_center, width, height)
            boxes = result.boxes.xywhn
            # 获取类别ID
            classes = result.boxes.cls
            # 获取置信度
            confs = result.boxes.conf

            # 写入每个符合条件的边界框
            for box, cls, conf in zip(boxes, classes, confs):
                if conf >= conf_threshold:
                    # box已经是归一化格式的tensor
                    x_center, y_center, width, height = box
                    # cls是类别ID的tensor
                    class_id = int(cls)
                    
                    # 写入文件
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        processed_count += 1
        print(f"  ({processed_count}) 已处理: {image_filename}, 标签已保存至: {label_filename}")

    print(f"\n共处理了 {processed_count} 张图片。")
    print("所有预测结果已成功转换为YOLO数据集！")


def main():
    # --- 用户配置 ---
    # 1. 加载您训练好的模型
    model = YOLO("runs/detect/train-20/weights/best.pt") 

    # 2. 指定要进行预测并生成标签的图像文件夹
    source_images = r"augmented_sharpened"

    # 3. 指定新生成的数据集的输出文件夹
    new_dataset_output_dir = "new_pseudo_labeled_dataset"

    # 4. 设置置信度阈值 (只会保存置信度高于此值的预测框)
    confidence_threshold = 0.0

    # --- 模型预测 ---
    # 设置 stream=True 可以更有效地处理大量图片
    print("开始对源图片进行预测以生成伪标签...")
    results = model(source_images, stream=True, verbose=False) # 使用 verbose=False 减少不必要的输出

    # --- 将预测结果保存为新的训练数据 ---
    save_predictions_as_yolo_dataset(results, new_dataset_output_dir, confidence_threshold)


if __name__ == '__main__':
    freeze_support()
    main()
