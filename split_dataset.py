import os
import random
import shutil

# --- 配置 ---
# 源文件夹
source_dir = r'.\resized_data'

# 目标基础文件夹
dest_base_dir = r'.\pre_train_data\images'

# 数据集划分比例
# 80% 训练集, 10% 验证集, 10% 测试集
split_ratio = {'train': 0.8, 'val': 0.1, 'test': 0.1}

# --- 初始化 ---
# 确保目标文件夹存在
train_dir = os.path.join(dest_base_dir, 'train')
val_dir = os.path.join(dest_base_dir, 'val')
test_dir = os.path.join(dest_base_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# --- 执行划分 ---
print(f"开始从 '{source_dir}' 划分数据集...")

# 获取所有图片文件
try:
    image_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
except FileNotFoundError:
    print(f"错误：源文件夹 '{source_dir}' 不存在。请确保脚本在正确的目录下运行。")
    exit()

if not image_files:
    print(f"错误：在 '{source_dir}' 中没有找到任何图片。")
    exit()

# 随机打乱文件列表
random.shuffle(image_files)

# 计算每个集合的大小
total_images = len(image_files)
train_split_index = int(total_images * split_ratio['train'])
val_split_index = int(total_images * (split_ratio['train'] + split_ratio['val']))

# 划分文件列表
train_files = image_files[:train_split_index]
val_files = image_files[train_split_index:val_split_index]
test_files = image_files[val_split_index:]

# 定义一个函数来复制文件并打印进度
def copy_files(files, dest_dir, set_name):
    print(f"\n正在复制 {len(files)} 个文件到 {set_name} 文件夹 ('{dest_dir}')...")
    for i, filename in enumerate(files):
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        shutil.copy2(source_path, dest_path) # copy2 会同时复制元数据
        # 打印简化的进度
        if (i + 1) % 50 == 0 or (i + 1) == len(files):
            print(f"  已复制 {i + 1}/{len(files)}", end='\r')
    print(f"\n{set_name} 集合复制完成。")

# 执行复制操作
copy_files(train_files, train_dir, '训练集')
copy_files(val_files, val_dir, '验证集')
copy_files(test_files, test_dir, '测试集')

# --- 总结 ---
print("\n======================================")
print("数据集划分成功！")
print(f"总图片数: {total_images}")
print(f"训练集: {len(train_files)} 张图片 -> {train_dir}")
print(f"验证集: {len(val_files)} 张图片 -> {val_dir}")
print(f"测试集: {len(test_files)} 张图片 -> {test_dir}")
print("======================================")
