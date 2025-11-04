import os
import random
from pathlib import Path

def check_data_integrity(img_dir, ann_dir):
    """
    校验图像与标签匹配性，返回有效数据总量及文件名列表
    :param img_dir: 图像目录（JPEGImages）
    :param ann_dir: 标签目录（SegmentationClass）
    :return: Total_Valid（有效数据总量）、valid_files（有效文件名列表，无后缀）
    """
    # 提取图像和标签文件（仅保留后缀匹配的文件，取无后缀名）
    img_files = {Path(f).stem for f in os.listdir(img_dir) if f.lower().endswith('.jpg')}
    ann_files = {Path(f).stem for f in os.listdir(ann_dir) if f.lower().endswith('.png')}
    
    # 有效数据：同时存在图像和标签的文件
    valid_files = sorted(list(img_files.intersection(ann_files)))
    Total_Valid = len(valid_files)
    
    # 打印校验结果
    print(f"=== 数据校验结果 ===")
    print(f"图像文件总数：{len(img_files)} 张")
    print(f"标签文件总数：{len(ann_files)} 个")
    print(f"有效匹配数据：{Total_Valid} 个（图像+标签均存在）")
    if Total_Valid == 0:
        raise ValueError("无有效匹配数据，请检查图像/标签目录路径及文件格式！")
    return Total_Valid, valid_files

def calc_label_num_by_ratio(Total_Valid, target_ratios, dataset_type):
    """
    按论文比例计算带标签数据量，过滤无效比例
    :param Total_Valid: 有效数据总量
    :param target_ratios: 论文指定比例列表
    :param dataset_type: 数据集类型（"original"或"augmented"）
    :return: 有效比例-带标签数量字典（如{0.063:63, 0.125:125}）
    """
    ratio_label_dict = {}
    for ratio in target_ratios:
        label_num = int(Total_Valid * ratio)
        # 过滤条件：带标签数量≥1，且100%比例时需等于有效总量
        if (label_num >= 1) or (ratio == 1.0 and Total_Valid >= 1):
            if ratio == 1.0:
                label_num = Total_Valid  # 100%比例强制取全量有效数据
            ratio_label_dict[ratio] = label_num
        else:
            print(f"[跳过] {dataset_type}数据集：比例{ratio:.3f}计算得{label_num}张（<1张，无效）")
    return ratio_label_dict

def split_by_ratio(valid_files, ratio_label_dict, base_output_dir, dataset_type):
    """
    按比例划分数据，生成label.txt和unlabeled.txt
    :param valid_files: 有效文件名列表
    :param ratio_label_dict: 比例-带标签数量字典
    :param base_output_dir: 输出根目录
    :param dataset_type: 数据集类型（"original"或"augmented"）
    """
    # 固定随机种子（复现论文划分逻辑，避免每次运行结果不同）
    random.seed(2022)
    shuffled_files = random.sample(valid_files, len(valid_files))
    
    # 遍历每个有效比例，生成划分文件
    for ratio, label_num in ratio_label_dict.items():
        # 划分带标签和无标签数据
        label_files = shuffled_files[:label_num]
        unlabeled_files = shuffled_files[label_num:] if label_num < len(shuffled_files) else []
        
        # 生成子目录名（如original_0.063、augmented_0.125）
        ratio_str = f"{ratio:.3f}".rstrip('0').rstrip('.')  # 比例格式化为字符串（如0.063→"0.063"，1.0→"1"）
        split_dir = os.path.join(base_output_dir, f"{dataset_type}_{ratio_str}")
        os.makedirs(split_dir, exist_ok=True)
        
        # 写入文件
        label_path = os.path.join(split_dir, "label.txt")
        unlabeled_path = os.path.join(split_dir, "unlabeled.txt")
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(label_files))
        with open(unlabeled_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(unlabeled_files))
        
        # 打印划分结果
        print(f"[{dataset_type}数据集] 比例{ratio:.3f}：")
        print(f"  - 带标签数据：{len(label_files)} 张（存于{label_path}）")
        print(f"  - 无标签数据：{len(unlabeled_files)} 张（存于{unlabeled_path}）")

if __name__ == "__main__":
    # -------------------------- 请修改为你的实际路径 --------------------------
    VOC2012_ROOT = "/DeepLearning_linux/Projects/GTA-Seg/data/VOC2012/"  # 你的VOC2012根目录
    IMG_DIR = os.path.join(VOC2012_ROOT, "JPEGImages")  # 图像目录（.jpg文件）
    ANN_DIR = os.path.join(VOC2012_ROOT, "SegmentationClass")  # 标签目录（.png文件）
    BASE_OUTPUT_DIR = "\\wsl.localhost\\Ubuntu-22.04\\DeepLearning_linux\\Projects\\GTA-Seg\\data\\splits\\pascal"  # 目标输出路径
    # --------------------------------------------------------------------------
    
    # 1. 初始化输出目录
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    print(f"输出根目录已创建：{BASE_OUTPUT_DIR}\n")
    
    # 2. 校验有效数据，获取总量和文件名列表
    Total_Valid, valid_files = check_data_integrity(IMG_DIR, ANN_DIR)
    print(f"\n开始按论文比例划分数据（有效数据总量：{Total_Valid} 张）\n")
    
    # 3. 定义论文指定的目标比例（原始训练集+增强训练集）
    original_ratios = [0.063, 0.125, 0.25, 0.5, 1.0]  # 原始训练集：~6.3%、~12.5%、25%、50%、100%
    augmented_ratios = [0.063, 0.125, 0.25, 0.5]      # 增强训练集：~6.3%、~12.5%、25%、50%（无100%单独划分）
    
    # 4. 处理原始训练集划分（按比例计算）
    print("=== 原始训练集划分（论文比例）===")
    original_ratio_label = calc_label_num_by_ratio(Total_Valid, original_ratios, "original")
    if original_ratio_label:
        split_by_ratio(valid_files, original_ratio_label, BASE_OUTPUT_DIR, "original")
    else:
        print("原始训练集无有效比例可划分\n")
    
    # 5. 处理增强训练集划分（需提前融合SBD数据，否则跳过）
    print("\n=== 增强训练集划分（需融合SBD数据）===")
    # 增强训练集判断：论文总标注量10582张，若有效数据≥5000张（近似阈值），则视为已融合SBD
    if Total_Valid >= 5000:
        augmented_ratio_label = calc_label_num_by_ratio(Total_Valid, augmented_ratios, "augmented")
        if augmented_ratio_label:
            split_by_ratio(valid_files, augmented_ratio_label, BASE_OUTPUT_DIR, "augmented")
        else:
            print("增强训练集无有效比例可划分")
    else:
        print("未检测到融合SBD的数据（有效数据<5000张），跳过增强训练集划分")
    
    # 6. 输出最终文件路径汇总
    print(f"\n=== 划分完成！所有文件路径如下 ===")
    for root, dirs, files in os.walk(BASE_OUTPUT_DIR):
        for file in files:
            if file in ["label.txt", "unlabeled.txt"]:
                full_path = os.path.join(root, file)
                print(f"- {full_path}")