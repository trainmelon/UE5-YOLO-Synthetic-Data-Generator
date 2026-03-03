import os
import cv2
import glob
import random

# 数据集路径配置
base_dir = "D:/Dataset_Output_Full/data"
img_dir = os.path.join(base_dir, "images")
lbl_dir = os.path.join(base_dir, "labels")
# 抽样画框输出路径
sample_dir = os.path.join(base_dir, "samples_check")

# 类别映射表 (和虚幻引擎里的 class_map 保持一致)
class_map = {0: "Turbine", 1: "Tower", 2: "Car", 3: "Ship"}


def clean_dataset():
    print("=== 第一步：开始清理孤立的TXT文件和空数据 ===")

    # 获取所有的 txt 和 png 文件名
    all_txts = set([os.path.basename(p) for p in glob.glob(os.path.join(lbl_dir, "*.txt"))])
    all_imgs = set([os.path.basename(p) for p in glob.glob(os.path.join(img_dir, "*.png"))])

    deleted_orphans = 0
    deleted_empties = 0

    # 1. 遍历 txt，如果同名 png 不存在（被你手动删了），把 txt 也删掉
    for txt_name in list(all_txts):
        basename = os.path.splitext(txt_name)[0]
        img_name = basename + ".png"
        txt_path = os.path.join(lbl_dir, txt_name)
        img_path = os.path.join(img_dir, img_name)

        if img_name not in all_imgs:
            os.remove(txt_path)
            all_txts.remove(txt_name)
            deleted_orphans += 1
            continue

        # 2. 检查空文件 (如果生成时目标出画，可能会产生空白txt)
        with open(txt_path, "r") as f:
            lines = f.readlines()

        valid_lines = [l.strip() for l in lines if l.strip()]
        if not valid_lines:
            os.remove(txt_path)
            if os.path.exists(img_path):
                os.remove(img_path)
                all_imgs.remove(img_name)
            all_txts.remove(txt_name)
            deleted_empties += 1

    print(f"-> 清理完毕！帮你删除了 {deleted_orphans} 个孤立的 TXT 标签。")
    print(f"-> 另外清理了 {deleted_empties} 组没有包含任何目标的空数据。")
    print(f"-> 当前剩余有效数据对：{len(all_imgs)} 组。")

    return list(all_imgs)


def draw_samples(valid_imgs, num_samples=100):
    print(f"\n=== 第二步：开始随机抽取 {num_samples} 张样本并画框验证 ===")
    os.makedirs(sample_dir, exist_ok=True)

    # 清空之前的抽样文件夹，防止旧图干扰
    for f in glob.glob(os.path.join(sample_dir, "*.png")):
        os.remove(f)

    # 如果剩余图片不足 100 张，则全部画框
    if len(valid_imgs) < num_samples:
        num_samples = len(valid_imgs)

    sampled_imgs = random.sample(valid_imgs, num_samples)

    for img_name in sampled_imgs:
        basename = os.path.splitext(img_name)[0]
        txt_name = basename + ".txt"

        img_path = os.path.join(img_dir, img_name)
        txt_path = os.path.join(lbl_dir, txt_name)
        out_path = os.path.join(sample_dir, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        with open(txt_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])

                # YOLO 的相对坐标 (0~1) 转换为图像的绝对像素坐标
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                # 绘制绿色边界框，线宽 2
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 绘制类别名称底色和文字，防止背景太亮看不清字
                class_name = class_map.get(class_id, "Unknown")
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2

                (text_w, text_h), _ = cv2.getTextSize(class_name, font, font_scale, thickness)
                cv2.rectangle(img, (x1, y1 - text_h - 5), (x1 + text_w, y1), (0, 255, 0), -1)
                cv2.putText(img, class_name, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)

        # 保存画好框的图片
        cv2.imwrite(out_path, img)

    print(f"-> 抽样画框完成！")
    print(f"-> 请打开文件夹验收成果：{sample_dir}")


if __name__ == "__main__":
    remaining_imgs = clean_dataset()
    if remaining_imgs:
        draw_samples(remaining_imgs, 100)
    else:
        print("错误：没有找到有效的图片数据！")