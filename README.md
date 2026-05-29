# 基于YOLO26m的摩天地标识别项目 (Skyscraper Landmark Recognition Project Based on YOLO26m)

本项目是一个中山大学2025-2026学年《数字图像处理》课程的大作业，主要内容为基于 YOLO26m 模型的摩天地标识别任务，项目涵盖了从数据采集、预处理、增强、标注到模型训练、评估和部署预测的全过程。


## 目录

- [项目架构](#项目架构)
- [技术栈](#技术栈)
- [环境搭建](#环境搭建)
  - [1. 创建虚拟环境](#1-创建虚拟环境)
  - [2. 安装依赖](#2-安装依赖)
- [数据处理流程](#数据处理流程)
  - [1. 原始数据](#1-原始数据)
  - [2. 图像尺寸统一 (resize_images.py)](#2-图像尺寸统一-resize_images.py)
  - [3. 图像增强 (enhance_images.py)](#3-图像增强-enhance_images.py)
  - [4. 图像标注 (labelimg)](#4-图像标注-labelimg)
  - [5. 数据集划分 (split_dataset.py)](#5-数据集划分-split_dataset.py)
- [模型训练](#模型训练)
- [模型预测](#模型预测)
- [文件结构](#文件结构)

---

## 项目架构

项目遵循一个标准的计算机视觉任务流程：

1.  **数据准备**：收集原始图像，通过一系列脚本进行尺寸统一、数据增强和格式化。
2.  **数据标注**：使用 labelimg 对图像进行标注，存储为 YOLO 格式。
3.  **模型训练**：使用处理好的数据集对 YOLO26m 模型进行微调。
4.  **模型推理**：使用训练好的模型对新的图像或视频进行地标识别。

## 技术栈

- **labelimg：Python 3.8**: 用于图像标注，生成 YOLO 格式的标签文件
- **PyTorch 2.12.0：Python 3.12** : 用于模型训练和推理
- **CUDA 12.6**: 用于在 GPU 上加速训练和推理
- **Ultralytics YOLO26m/YOLO26n**: 核心检测框架
- **OpenCV**: 用于图像处理和数据增强

## 环境搭建

### 1. 创建虚拟环境

由于labelimg和PyTorch对Python版本有不同的要求，建议为每个工具创建独立的虚拟环境。
labelimg使用Python 3.8，PyTorch使用Python 3.12

```bash
conda create -n yolo_dip python=3.8
conda create -n pytorch python=3.12
```

### 2. 安装依赖

在激活的虚拟环境中，安装项目所需的库。

```bash
pip install labelimg
pip install ultralytics opencv-python numpy 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## 数据处理流程

本项目的数据处理流程如下：

### 1. 原始数据

收集到的原始地标图片存放于 `./raw_data` 目录下的不同子文件夹中。

### 2. 图像尺寸统一 (resize_images.py)

由于原始图像尺寸不一，首先使用此脚本将所有图像调整为模型输入所需的 `512x512` 尺寸。

- **输入**: `raw_data/<folder>`
- **输出**: `resized_data/<folder>` (重命名并调整尺寸) 
- **执行**:
  ```bash
  python resize_images.py
  ```

### 3. 图像增强 (enhance_images.py)

为了提升模型的泛化能力和鲁棒性，我们对调整尺寸后的图像应用了三种不同的数据增强技术。这有助于模型学习到在不同光照、颜色和清晰度条件下的特征。

-   **输入**: `resized_data/` 文件夹中的 `512x512` 图像。
-   **输出**: 三个独立的增强结果文件夹。
-   **执行**:
    ```bash
    python enhance_images.py
    ```

#### 增强方法详解:

1.  **灰度变换 (`augmented_grayscale/`)**
    -   **方法**: 将彩色图像转换为灰度图像。
    -   **参数**: 使用 `cv2.cvtColor` 与 `cv2.COLOR_BGR2GRAY`。
    -   **作用**: 模拟在缺乏色彩信息或光线较暗场景下的情况，强制模型更多地关注物体的形状和纹理，而不是颜色。

2.  **直方图均衡化 (`augmented_hist_equalized/`)**
    -   **方法**: 在 HSV 颜色空间中，仅对亮度（Value）通道应用直方图均衡化。
    -   **参数**: 使用 `cv2.equalizeHist` 对 V 通道进行处理。
    -   **作用**: 显著增强图像的对比度，尤其对于那些光照过暗或过曝的图像效果明显。这有助于模型在光照条件不佳时也能识别出目标。

3.  **图像锐化 (`augmented_sharpened/`)**
    -   **方法**: 使用非锐化掩模（Unsharp Masking）技术来增强图像的边缘和细节。
    -   **参数**:
        -   `cv2.GaussianBlur` 使用 `(0, 0)` 的核大小和 `sigma=5` 来创建模糊版本的图像。
        -   `cv2.addWeighted` 将原图（权重 `1.5`）和模糊图像（权重 `-0.5`）进行加权叠加。
    -   **作用**: 使图像的轮廓和细节更加突出，有助于模型更好地捕捉目标的边缘特征。

### 4. 图像标注 (半监督学习策略)

面对上千张图片的数据集，完全手动标注（约900张训练/验证图片）工作量巨大。为了高效地完成标注，我们采用了一种半监督的学习策略：先用少量手动标注的数据训练一个初步模型，再利用这个模型为海量数据自动生成“伪标签”，最后人工校验和修正。

具体流程如下：

1.  **创建初始小规模数据集**
    -   **动作**: 从总数据集中取未经变换的250张原图作为初始数据集 `pre_train_data`。
    -   **工具**: `split_dataset.py`
    -   **细节**: 使用脚本将这250张图片按 8:1:1 的比例划分为训练集、验证集和测试集。

2.  **手动标注**
    -   **动作**: 仅对上一步中划分出的训练集和验证集（250 * 90% = 225张图片）进行手动标注。
    -   **工具**: `labelimg`
    -   **结果**: 获得了一个小而精的高质量标注数据集。

3.  **训练初步模型**
    -   **动作**: 使用这个小数据集训练一个轻量的 `yolo26n`模型。
    -   **工具**: `train.py`
    -   **结果**: 得到一个初步但有效的检测模型，`runs/detect/train-20/weights/best.pt`。我们通过在测试集上验证，确认了该模型已具备良好的识别能力。

4.  **生成伪标签**
    -   **动作**: 利用上一步训练好的初步模型，对 **全部** 未经标注的图像数据进行预测，从而自动生成标签。
    -   **工具**: `create_pseudo_labels.py`
    -   **细节**: 脚本会读取模型预测结果，并将边界框转换为 YOLO 格式的 `.txt` 标签文件，形成一个带有伪标签的大型数据集。

### 5. 数据集划分 (split_dataset.py)

最后将完整数据集再次按 8:1:1 的比例划分为最终的训练、验证和测试集。

- **输入**: `resized_data/` 
- **输出**: `data/images/` 下的 `train/`, `val/`, `test/` 文件夹
- **执行**:
  ```bash
  python split_dataset.py
  ```

最后，手动创建一个 `data.yaml` 文件来告诉 YOLO26m 数据集的位置和类别信息。

## 模型训练

模型训练由 `train.py` 脚本执行。它会加载一个预训练的 YOLO26m 模型，并使用我们准备好的数据集进行微调。

- **配置**:
  - **模型**: `yolo26m.pt`
  - **数据**: `data/data.yaml`
  - **超参数**: 100个周期 (epochs), 图像尺寸512, 批大小32
- **执行**:
  ```bash
  python train.py
  ```
- **输出**: 训练日志、权重文件和结果会保存在 `runs/detect/train-21` 目录下，最好的权重是 `runs/detect/train-21/weights/best.pt`。

## 模型预测

使用 `predict.py` 脚本和训练好的最佳权重来对新的图像或视频进行预测。

- **配置**:
  - **模型路径**: `runs/detect/train-21/weights/best.pt` 
  - **预测目标**: `predict_data/<folder>`
- **执行**:
  ```bash
  python predict.py
  ```
- **输出**: 预测结果（带边界框的图像或视频）将默认保存在 `runs/detect/predict<N>/` 目录下。

## 文件结构

```
project/
│
├── pre_train_data/        # 初始小规模数据集（250张原图）
│   ├── images/            # 图像
│   │   ├── train/         # 训练集图像
│   │   ├── val/           # 验证集图像
│   │   └── test/          # 测试集图像
│   ├── labels/            # 标签
│   │   ├── train/         # 训练集标签
│   │   ├── val/           # 验证集标签
│   │   └── test/          # 测试集标签
│   └── pre_train_data.yaml# 数据集配置文件
│
├── data/                  # 存放符合YOLO格式的最终数据集
│   ├── images/            # 图像
│   │   ├── train/         # 训练集图像
│   │   ├── val/           # 验证集图像
│   │   └── test/          # 测试集图像
│   ├── labels/            # 标签
│   │   ├── train/         # 训练集标签
│   │   ├── val/           # 验证集标签
│   │   └── test/          # 测试集标签
│   └── data.yaml          # 数据集配置文件
│
├── predict_data/          # 存放待预测的图像数据
│
├── runs/                  # YOLO26 训练和预测的输出
│   ├── detect/            # 训练和预测结果
│   │   ├── train-XX/      # 训练结果目录
│   │   └── predict-XX/    # 预测结果目录
│
├── train.py               # 模型训练脚本
├── predict.py             # 模型预测脚本
│
├── resize_images.py       # 预处理：调整尺寸
├── enhance_images.py      # 预处理：数据增强
├── create_pseudo_labels.py# 预处理：生成伪标签
├── split_dataset.py       # 预处理：划分数据集
│
├── yolo26n.pt             # 预训练的YOLO26n模型权重 
├── yolo26m.pt             # 预训练的YOLO26m模型权重 
│
└── README.md              # 本文档
```


---

## 项目所有者

- **BeizeWang**
- **Olivia Xu**

**联系方式:** <wangbz3@mail2.sysu.edu.cn>