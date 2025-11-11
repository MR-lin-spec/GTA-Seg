#!/usr/bin/env python3
# splits.py  ——  从指定txt文件读取训练池，其余作为验证集
# /root/ST-PlusPlus/splits.py
import os, random
from pathlib import Path

# -------------------------------------------------
# 1. 从txt文件读取basename列表（无后缀）
# -------------------------------------------------
def read_txt_basename(txt_path: Path):
    """读取txt文件，每行一个basename，返回排序后的列表"""
    if not txt_path.exists():
        raise FileNotFoundError(f"指定的txt文件不存在: {txt_path}")
    
    basenames = []
    with txt_path.open('r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if name:  # 跳过空行
                basenames.append(name)
    return sorted(basenames)

# -------------------------------------------------
# 2. 扫描真实jpg-png对，并按txt文件拆分训练池与验证池
# -------------------------------------------------
def scan_and_split_basenames(img_root: Path, img_sub: str, ann_sub: str, txt_path: Path):
    """扫描磁盘上所有真实对，拆分为训练池（txt中且存在）和验证池（剩余）"""
    img_dir = img_root / img_sub
    ann_dir = img_root / ann_sub
    
    # 磁盘上所有真实存在的对
    all_basenames = []
    for jpg in img_dir.glob("*.jpg"):
        png = ann_dir / f"{jpg.stem}.png"
        if png.exists():
            all_basenames.append(jpg.stem)
    all_basenames = sorted(all_basenames)
    
    # txt文件中指定的对（需确保真实存在）
    txt_basenames = read_txt_basename(txt_path)
    txt_set = set(txt_basenames)
    all_set = set(all_basenames)
    
    # 训练池：txt文件中指定且在磁盘上真实存在的
    training_pool = sorted(list(txt_set & all_set))
    
    # 验证池：磁盘上存在但不在txt文件中的
    validation_pool = sorted(list(all_set - txt_set))
    
    print(f"磁盘上真实存在的图片-标签对：{len(all_basenames)} 组")
    print(f"txt文件中指定的参与训练的对：{len(txt_basenames)} 组")
    print(f"  └─ 其中真实存在（训练池）：{len(training_pool)} 组")
    print(f"剩余作为验证集（验证池）：{len(validation_pool)} 组")
    
    return training_pool, validation_pool

# -------------------------------------------------
# 3. 划分训练子集 + 独立测试集
# -------------------------------------------------
def split_and_write_basenames(training_pool, validation_pool, ratios, out_base: Path, tag: str):
    random.seed(2022)
    
    # 验证集：直接使用所有剩余的图片（验证池）
    val_list = validation_pool.copy()
    random.shuffle(val_list)  # 打乱顺序
    
    # 写 val.txt（与比例子目录同级）
    (out_base / "val.txt").write_text("\n".join(val_list) + "\n")
    print(f"[val] 测试集数量：{len(val_list)}（从验证池选取）")
    
    # 对 training_pool 做比例划分（不再从中拆分验证集）
    train_list = training_pool.copy()
    random.shuffle(train_list)
    
    print(f"[train] 训练池总数：{len(train_list)}，开始划分 labeled/unlabeled...")
    
    for ratio in ratios:
        n_label = max(1, int(len(train_list) * ratio)) if ratio <= 1.0 else len(train_list)
        labeled = train_list[:n_label]
        unlabeled = train_list[n_label:]

        ratio_str = f"{ratio}".replace("/", "_")
        split_dir = out_base / f"{tag}_{ratio_str}"
        split_dir.mkdir(parents=True, exist_ok=True)

        (split_dir / "labeled.txt").write_text("\n".join(labeled) + "\n")
        (split_dir / "unlabeled.txt").write_text("\n".join(unlabeled) + "\n")

        print(f"  ratio={ratio:>6} -> labeled={len(labeled):>4}  unlabeled={len(unlabeled):>4}")

# -------------------------------------------------
# 4. 主入口
# -------------------------------------------------
if __name__ == "__main__":
    # --------------- 用户路径（按需修改） ---------------
    VOC2012_ROOT    = Path("/root/autodl-tmp/VOC2012")
    BASE_OUTPUT_DIR = Path("/root/GTA-Seg/data/splits_pascal")
    TXT_FILE_PATH   = Path("/root/autodl-tmp/VOC2012/usage.txt")  # 新增：指定训练图片列表
    # ----------------------------------------------------
    
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 获取训练池和验证池
    training_pool, validation_pool = scan_and_split_basenames(
        VOC2012_ROOT, "JPEGImages", "SegmentationClassAug", TXT_FILE_PATH
    )

    ratios = [1, 1/2, 1/4, 1/8, 1/16]

    print("\n=== 开始划分（训练池来自txt，验证池使用剩余）===")
    split_and_write_basenames(training_pool, validation_pool, ratios, BASE_OUTPUT_DIR, "original")

    print("\n=== 划分完成！文件清单 ===")
    print(BASE_OUTPUT_DIR / "val.txt")  # 验证集
    for root, _, files in os.walk(BASE_OUTPUT_DIR):
        for f in files:
            if f in {"labeled.txt", "unlabeled.txt"}:
                print(Path(root) / f)