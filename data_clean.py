import os
import cv2
import glob
import random

# 数据集路径配置（请确保与你实际路径一致）
base_dir = "D:/Dataset_Output_Full/data_v2"
img_dir = os.path.join(base_dir, "images")
lbl_dir = os.path.join(base_dir, "labels")
sample_dir = os.path.join(base_dir, "samples_check")

class_map = {0: "Turbine", 1: "Tower", 2: "Car", 3: "Ship"}

# ==========================================
# 工业级清洗阈值配置
# ==========================================
IMG_W = 1920.0
IMG_H = 1080.0
MIN_PIXEL = 15  # 最小像素阈值：宽或高小于 15 像素直接删除
EDGE_MARGIN = 0.005  # 边缘判定厚度：距离屏幕边缘 0.5% 以内视为“触碰边缘”
LARGE_AREA_THRESH = 0.02  # 面积豁免权：如果物体面积超过全屏的 2% (约4万像素)，即使在边缘也保留


def clean_dataset():
    print("=== 第一步：开始执行高级脏数据清洗 (边缘过滤 & 尺寸过滤 & 空图清理) ===")

    all_txts = set([os.path.basename(p) for p in glob.glob(os.path.join(lbl_dir, "*.txt"))])
    all_imgs = set([os.path.basename(p) for p in glob.glob(os.path.join(img_dir, "*.png"))])

    deleted_orphans = 0
    deleted_empties = 0
    deleted_edge_boxes = 0
    deleted_tiny_boxes = 0

    # 1. 反向清理：如果图片不存在，删掉对应的 txt
    for txt_name in list(all_txts):
        basename = os.path.splitext(txt_name)[0]
        img_name = basename + ".png"
        txt_path = os.path.join(lbl_dir, txt_name)

        if img_name not in all_imgs:
            os.remove(txt_path)
            all_txts.remove(txt_name)
            deleted_orphans += 1

    # 2. 正向清理：如果 txt 不存在，删掉孤立的图片
    for img_name in list(all_imgs):
        basename = os.path.splitext(img_name)[0]
        txt_name = basename + ".txt"
        img_path = os.path.join(img_dir, img_name)

        if txt_name not in all_txts:
            os.remove(img_path)
            all_imgs.remove(img_name)
            deleted_orphans += 1

    # 3. 核心清洗逻辑：遍历所有成对的数据，清洗标签内部的杂质
    for txt_name in list(all_txts):
        basename = os.path.splitext(txt_name)[0]
        img_name = basename + ".png"
        txt_path = os.path.join(lbl_dir, txt_name)
        img_path = os.path.join(img_dir, img_name)

        with open(txt_path, "r") as f:
            lines = f.readlines()

        valid_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5: continue

            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])

            # 【判定 A：极小像素过滤】
            pixel_w = w * IMG_W
            pixel_h = h * IMG_H
            if pixel_w < MIN_PIXEL or pixel_h < MIN_PIXEL:
                deleted_tiny_boxes += 1
                continue  # 丢弃该行

            # 【判定 B：边缘 + 面积豁免权】
            xmin = cx - w / 2.0
            xmax = cx + w / 2.0
            ymin = cy - h / 2.0
            ymax = cy + h / 2.0

            # 是否触碰到画面的上下左右任意边缘？
            is_on_edge = (xmin <= EDGE_MARGIN) or (xmax >= 1.0 - EDGE_MARGIN) or \
                         (ymin <= EDGE_MARGIN) or (ymax >= 1.0 - EDGE_MARGIN)

            area = w * h  # 物体占据全屏的比例 (0.0 ~ 1.0)

            if is_on_edge and area < LARGE_AREA_THRESH:
                # 在边缘，且不够巨大（比如只是边缘露出的半个车），无情删除！
                deleted_edge_boxes += 1
                continue  # 丢弃该行

            # 通过所有考核，加入保留名单
            valid_lines.append(line)

        # 4. 根据清洗结果，决定文件的生死
        if len(valid_lines) == 0:
            # 如果这图里所有的框都被判定为“垃圾”并删光了，它变成了空图
            os.remove(txt_path)
            if os.path.exists(img_path):
                os.remove(img_path)
                all_imgs.remove(img_name)
            all_txts.remove(txt_name)
            deleted_empties += 1
        else:
            # 如果还有活下来的框，把干净的框重新写回原文件（覆盖旧数据）
            with open(txt_path, "w") as f:
                f.writelines(valid_lines)

    print("\n=== 清洗结果报告 ===")
    print(f"-> 删除了 {deleted_orphans} 个孤立/无法配对的文件。")
    print(f"-> 剔除了 {deleted_tiny_boxes} 个小到肉眼看不清的微粒目标。")
    print(f"-> 剔除了 {deleted_edge_boxes} 个位于屏幕边缘的残缺碎片。")
    print(f"-> 连带销毁了 {deleted_empties} 组全军覆没的空图片。")
    print(f"-> 最终剩余黄金数据对：{len(all_imgs)} 组。")

    return list(all_imgs)


def draw_samples(valid_imgs, num_samples=100):
    print(f"\n=== 第二步：开始随机抽取 {num_samples} 张样本并画框验证 ===")
    os.makedirs(sample_dir, exist_ok=True)

    for f in glob.glob(os.path.join(sample_dir, "*.png")):
        os.remove(f)

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
        if img is None: continue
        h, w = img.shape[:2]

        with open(txt_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])

                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                class_name = class_map.get(class_id, "Unknown")
                font = cv2.FONT_HERSHEY_SIMPLEX

                (text_w, text_h), _ = cv2.getTextSize(class_name, font, 0.7, 2)
                cv2.rectangle(img, (x1, y1 - text_h - 5), (x1 + text_w, y1), (0, 255, 0), -1)
                cv2.putText(img, class_name, (x1, y1 - 5), font, 0.7, (0, 0, 0), 2)

        cv2.imwrite(out_path, img)

    print(f"-> 抽样画框完成！请打开文件夹验收：{sample_dir}")


if __name__ == "__main__":
    remaining_imgs = clean_dataset()
    if remaining_imgs:
        draw_samples(remaining_imgs, 100)
    else:
        print("错误：没有找到有效的图片数据！")