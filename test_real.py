from ultralytics import YOLO

# 1. 加载训练出来的最高分权重 (注意替换为实际的准确路径)
model = YOLO('C:/Users/24281/PycharmProjects/Binance/runs/detect/UE5_Drone_Detection/v1_baseline3/weights/best.pt')

if __name__ == '__main__':
    # 2. 让模型去预测真实世界的图片或视频
    # 实物照片路径配置
    source_path = 'D:/Real_World_Test'

    # 3. 执行预测并保存结果
    results = model.predict(
        source=source_path,
        conf=0.1,       # 置信度阈值
        save=True,      # 强制保存画好框的图片
        project='UE5_Drone_Detection',
        name='Real_World_Results'
    )
    print("推理完成！去 UE5_Drone_Detection/Real_World_Results 文件夹查看结果！")