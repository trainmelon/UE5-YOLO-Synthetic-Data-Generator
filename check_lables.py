import cv2
import os

# 1. 数据集路径配置
data_dir = "D:/Dataset_Output_Full/data_v2"
img_dir = os.path.join(data_dir, "images")
lbl_dir = os.path.join(data_dir, "labels")
vis_dir = os.path.join(data_dir, "visualized")  # 存放画好框的验证图片

# 2. 类别映射 (与 UE5 生成脚本中完全一致)
class_map = {
    0: "Turbine",
    1: "Tower",
    2: "Car",
    3: "Ship"
}

# 3. 为不同类别分配醒目的 BGR 颜色 (OpenCV 默认是 BGR 而非 RGB)
colors = {
    0: (0, 255, 255),  # Turbine - 黄色
    1: (255, 144, 30),  # Tower - 蓝色
    2: (0, 255, 0),  # Car - 绿色
    3: (0, 0, 255)  # Ship - 红色
}


def visualize_yolo_dataset():
    # 如果可视化输出文件夹不存在，则自动创建
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    # 扫描目录下所有的 png 图片
    try:
        image_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    except FileNotFoundError:
        print(f"错误：找不到图片目录 {img_dir}，请检查路径。")
        return

    if not image_files:
        print(f"在 {img_dir} 中没有找到任何 PNG 图片。")
        return

    print(f"开始可视化验证，共找到 {len(image_files)} 张图片...")

    for img_file in image_files:
        img_path = os.path.join(img_dir, img_file)
        # 根据图片名推导 TXT 标签名 (例如 frame_0000.png -> frame_0000.txt)
        lbl_file = os.path.splitext(img_file)[0] + ".txt"
        lbl_path = os.path.join(lbl_dir, lbl_file)

        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}，跳过。")
            continue

        h, w, _ = img.shape

        # 如果存在对应的 YOLO 标签文件，则开始解析并画框
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # 核心还原算法：YOLO 归一化坐标 -> 绝对像素坐标
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)

                    # 获取类别名称和颜色
                    class_name = class_map.get(class_id, f"Unknown_{class_id}")
                    color = colors.get(class_id, (255, 255, 255))

                    # 1. 绘制矩形边界框 (线宽设为 2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                    # 2. 绘制类别名称底色板，防止文字看不清
                    label_text = class_name
                    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)

                    # 3. 绘制类别文字 (黑色)
                    cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # 保存画好框的图片
        output_path = os.path.join(vis_dir, img_file)
        cv2.imwrite(output_path, img)
        print(f"已验证并保存: {img_file}")

    print(f"\n全部验证完成！\n请前往目录查看肉眼验证结果: {vis_dir}")


if __name__ == "__main__":
    visualize_yolo_dataset()