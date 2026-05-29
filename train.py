from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    # 模型训练
    model = YOLO("yolo26m.pt")
    results = model.train(data="./data/data.yaml", epochs=100, imgsz=512, batch=32, device=0)

if __name__ == '__main__':
    freeze_support()
    main()
