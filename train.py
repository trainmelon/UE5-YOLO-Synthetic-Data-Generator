import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ultralytics import YOLO

model = YOLO('yolov8s.pt')

if __name__ == '__main__':
    results = model.train(
        data='D:/YOLO_Dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,          # 【修改 1】将多线程从 8 降到 4，减少内存并行复制压力
        project='UE5_Drone_Detection',
        name='v1_baseline',
        cache=False         # 【修改 2】关闭内存缓存！强制从你的硬盘直接读取图片
    )