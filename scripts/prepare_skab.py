"""
prepare_skab.py
将 SKAB 原始 CSV 转换为 HAISegLoader 兼容的 npy 格式。

目录结构假设（与 SKAB repo 一致）：
  skab_root/
    anomaly-free/anomaly-free.csv   ← 训练集（无异常）
    valve1/1.csv ... 16.csv
    valve2/1.csv ... 4.csv
    other/1.csv  ... 14.csv

运行方式：
  python prepare_skab.py --skab_root ./data --out_dir ./dataset/SKAB

输出：
  dataset/SKAB/SKAB_train.npy        shape: (N_train, C)
  dataset/SKAB/SKAB_test.npy         shape: (N_test,  C)
  dataset/SKAB/SKAB_test_label.npy   shape: (N_test,)
"""

import os
import argparse
import numpy as np
import pandas as pd


SENSOR_COLS = [
    "Accelerometer1RMS", "Accelerometer2RMS",
    "Current", "Pressure",
    "Temperature", "Thermocouple",
    "Voltage", "Volume Flow RateRMS",
]


def load_csv(path: str):
    """读单个 SKAB csv，返回 (features: ndarray, labels: ndarray)"""
    df = pd.read_csv(path, sep=";", index_col="datetime", parse_dates=True)
    # 有些文件列名带空格，统一 strip
    df.columns = df.columns.str.strip()

    # 只取传感器列（anomaly / changepoint 列不算特征）
    feat_cols = [c for c in SENSOR_COLS if c in df.columns]
    missing = set(SENSOR_COLS) - set(feat_cols)
    if missing:
        print(f"  [警告] {path} 缺少列: {missing}，用 0 填充")
        for c in missing:
            df[c] = 0.0

    features = df[SENSOR_COLS].values.astype(np.float32)

    if "anomaly" in df.columns:
        labels = df["anomaly"].values.astype(np.float32)
    else:
        labels = np.zeros(len(df), dtype=np.float32)

    return features, labels


def collect_files(root: str, subdir: str):
    folder = os.path.join(root, subdir)
    if not os.path.isdir(folder):
        return []
    files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]
    )
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skab_root", type=str, default="./data",
                        help="SKAB repo 里 data/ 文件夹的路径")
    parser.add_argument("--out_dir", type=str, default="./dataset/SKAB",
                        help="输出 npy 文件夹")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 训练集：anomaly-free ──────────────────────────────────────────────
    train_path = os.path.join(args.skab_root, "anomaly-free", "anomaly-free.csv")
    print(f"读取训练集: {train_path}")
    train_feat, _ = load_csv(train_path)
    print(f"  训练集 shape: {train_feat.shape}")

    # ── 测试集：valve1 + valve2 + other ──────────────────────────────────
    test_feats, test_labels = [], []
    for subdir in ["valve1", "valve2", "other"]:
        files = collect_files(args.skab_root, subdir)
        for fpath in files:
            print(f"读取测试文件: {fpath}")
            feat, label = load_csv(fpath)
            test_feats.append(feat)
            test_labels.append(label)
            print(f"  shape: {feat.shape}, 异常率: {label.mean():.3f}")

    test_feat_all = np.concatenate(test_feats, axis=0)
    test_label_all = np.concatenate(test_labels, axis=0)
    print(f"\n测试集合并后 shape: {test_feat_all.shape}")
    print(f"测试集总异常率: {test_label_all.mean():.4f}")

    # ── 保存 ──────────────────────────────────────────────────────────────
    np.save(os.path.join(args.out_dir, "SKAB_train.npy"), train_feat)
    np.save(os.path.join(args.out_dir, "SKAB_test.npy"), test_feat_all)
    np.save(os.path.join(args.out_dir, "SKAB_test_label.npy"), test_label_all)
    print(f"\n已保存到 {args.out_dir}/")
    print("  SKAB_train.npy      :", train_feat.shape)
    print("  SKAB_test.npy       :", test_feat_all.shape)
    print("  SKAB_test_label.npy :", test_label_all.shape)


if __name__ == "__main__":
    main()
