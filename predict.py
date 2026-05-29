from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    # 模型预测
    model = YOLO("runs/detect/train-21/weights/best.pt") 
    model("./raw_data/Videos", save=True)

if __name__ == '__main__':
    freeze_support()
    main()
