import os
import random
import shutil
from pathlib import Path

# 配置路径
source_dir = "D:/Dataset_Output_Full/data_v2"  # 数据源路径配置
img_src_dir = os.path.join(source_dir, "images")
lbl_src_dir = os.path.join(source_dir, "labels")

# YOLO 标准数据集输出路径
yolo_base_dir = "D:/YOLO_Dataset_v2"  # 数据输出路径配置

# 划分比例
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1


def create_yolo_structure():
    # 创建 YOLO 所需的目录树
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(yolo_base_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(yolo_base_dir, 'labels', split), exist_ok=True)
    print("-> YOLO 标准目录树创建完毕。")


def split_and_copy_data():
    # 获取所有有对应 txt 的 png 图片
    all_imgs = [f for f in os.listdir(img_src_dir) if f.endswith('.png')]
    valid_pairs = []

    for img in all_imgs:
        lbl = img.replace('.png', '.txt')
        if os.path.exists(os.path.join(lbl_src_dir, lbl)):
            valid_pairs.append((img, lbl))

    print(f"-> 共发现 {len(valid_pairs)} 组有效数据。")

    # 核心：随机打乱数据集（极其重要，防止模型按时间顺序学习导致过拟合）
    random.seed(42)  # 固定随机种子，保证每次划分结果一致
    random.shuffle(valid_pairs)

    total_len = len(valid_pairs)
    train_end = int(total_len * train_ratio)
    val_end = train_end + int(total_len * val_ratio)

    splits = {
        'train': valid_pairs[:train_end],
        'val': valid_pairs[train_end:val_end],
        'test': valid_pairs[val_end:]
    }

    # 开始拷贝文件
    for split_name, pairs in splits.items():
        print(f"-> 正在拷贝 {split_name} 集: {len(pairs)} 张...")
        for img_name, lbl_name in pairs:
            # 拷贝图片
            shutil.copy(
                os.path.join(img_src_dir, img_name),
                os.path.join(yolo_base_dir, 'images', split_name, img_name)
            )
            # 拷贝标签
            shutil.copy(
                os.path.join(lbl_src_dir, lbl_name),
                os.path.join(yolo_base_dir, 'labels', split_name, lbl_name)
            )

    print(f"\n=== 数据集划分完成！ ===")
    print(f"训练集 (Train): {len(splits['train'])}")
    print(f"验证集 (Val): {len(splits['val'])}")
    print(f"测试集 (Test): {len(splits['test'])}")
    print(f"请前往 {yolo_base_dir} 查看最终数据。")


if __name__ == "__main__":
    create_yolo_structure()
    split_and_copy_data()
