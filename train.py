import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ultralytics import YOLO

model = YOLO('yolov8s.pt')

if __name__ == '__main__':
    results = model.train(
        data='D:/YOLO_Dataset_v2/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,          # 多线程设置为 4，减少内存并行复制压力
        project='UE5_Drone_Detection',
        name='v2_baseline',
        cache=False         # 关闭内存缓存，强制从硬盘直接读取图片，降低性能压力
    )