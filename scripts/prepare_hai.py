"""
HAI (HIL-based Augmented ICS) data Preprocessor
====================================================
将原始 HAI CSV 文件转换为 Anomaly-Transformer 所需的 .npy 格式。

HAI 数据集下载地址:
  https://github.com/icsdataset/hai/releases
  选择某个版本解压后放入 data/HAI/<version>/ 目录。

支持的版本与目录结构:
  data/HAI/hai-20.07/  →  train*.csv.gz, test*.csv.gz  (Attack 列内嵌)
  data/HAI/hai-21.03/  →  train*.csv.gz, test*.csv.gz  (Attack 列内嵌)
  data/HAI/hai-22.04/  →  train*.csv,    test*.csv      (Attack 列内嵌)
  data/HAI/hai-23.05/  →  hai-train*.csv, hai-test*.csv + label-test*.csv (标签独立)
  data/HAI/haiend-23.05/ → end-train*.csv, end-test*.csv + label-test*.csv (标签独立)

运行方式:
  # 使用 hai-22.04（推荐，格式最简单）
  python scripts/prepare_hai.py --data_dir data/HAI/hai-22.04 --output_dir data/HAI

  # 使用 hai-23.05
  python scripts/prepare_hai.py --data_dir data/HAI/hai-23.05 --output_dir data/HAI

  # 使用 hai-20.07 / hai-21.03（gzip 压缩，自动处理）
  python scripts/prepare_hai.py --data_dir data/HAI/hai-20.07 --output_dir data/HAI
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# 所有版本中需要排除的非特征列（时间戳 + 标签相关）
EXCLUDE_COLS = {
    "timestamp", "time", "Timestamp", "Time",
    "Attack", "attack", "Label", "label",
    # hai-20.07 / hai-21.03 子攻击标签列
    "attack_P1", "attack_P2", "attack_P3",
}


def find_files(data_dir, *patterns):
    """按优先级依次尝试 glob 模式，返回第一个非空结果。"""
    for pattern in patterns:
        files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if files:
            return files
    return []


def _check_lfs(fp):
    """若文件是 Git LFS 指针则抛出清晰的错误。"""
    try:
        if fp.endswith(".gz"):
            import gzip
            with gzip.open(fp, "rt", errors="replace") as f:
                first = f.read(50)
        else:
            with open(fp, "r", errors="replace") as f:
                first = f.read(50)
        if first.startswith("version https://git-lfs"):
            raise RuntimeError(
                f"\n[Git LFS] '{os.path.basename(fp)}' is an LFS pointer, not real data.\n"
                "Run the following command in the HAI dataset repository root to download:\n"
                "  git lfs pull\n"
                "or download the release zip directly from:\n"
                "  https://github.com/icsdataset/hai/releases"
            )
    except (OSError, UnicodeDecodeError):
        pass  # 二进制 gzip 内容，不是 LFS 指针，继续正常读取


def _detect_sep(fp):
    """读取文件前两行，判断分隔符是逗号还是分号。"""
    try:
        if fp.endswith(".gz"):
            import gzip
            with gzip.open(fp, "rt", errors="replace") as f:
                header = f.readline()
        else:
            with open(fp, "r", errors="replace") as f:
                header = f.readline()
        return ";" if header.count(";") > header.count(",") else ","
    except Exception:
        return ","


def read_csvs(file_paths, label_col="Attack"):
    """
    读取一组 CSV / CSV.GZ 文件并拼接。
    返回 (data_array, label_array, feature_col_names)。
    若文件中不含 label_col，则 label_array 全为 0。
    """
    dfs = []
    for fp in file_paths:
        print(f"  Reading {os.path.basename(fp)} ...")
        _check_lfs(fp)
        compression = "gzip" if fp.endswith(".gz") else "infer"
        sep = _detect_sep(fp)
        if sep == ";":
            print(f"    (detected semicolon delimiter)")
        df = pd.read_csv(fp, compression=compression, sep=sep)
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)

    # 标签：大小写不敏感匹配
    col_lower_map = {c.lower(): c for c in df_all.columns}
    matched_col = col_lower_map.get(label_col.lower())
    if matched_col:
        labels = df_all[matched_col].values.astype(np.float32)
        # 部分版本标签为布尔或多值，统一转为 0/1
        labels = (labels > 0).astype(np.float32)
        if matched_col != label_col:
            print(f"  [Info] Matched label column '{matched_col}' (requested '{label_col}')")
    else:
        labels = np.zeros(len(df_all), dtype=np.float32)
        print(f"  [Info] Column '{label_col}' not in file, using zero labels.")

    # 排除标签相关列及任何以 "attack" 开头的子标签列（大小写均排除）
    feature_cols = [
        c for c in df_all.columns
        if c not in EXCLUDE_COLS and not c.lower().startswith("attack")
    ]
    data = df_all[feature_cols].values.astype(np.float32)
    data = np.nan_to_num(data)
    return data, labels, feature_cols


def read_label_files(label_paths, label_col="Attack"):
    """
    读取 hai-23.05 / haiend-23.05 格式的独立标签文件。
    通常只含 timestamp + Attack 列。
    """
    dfs = []
    for fp in label_paths:
        print(f"  Reading label file {os.path.basename(fp)} ...")
        _check_lfs(fp)
        sep = _detect_sep(fp)
        df = pd.read_csv(fp, sep=sep)
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)

    col_lower_map = {c.lower(): c for c in df_all.columns}
    matched_col = col_lower_map.get(label_col.lower())
    if matched_col:
        labels = (df_all[matched_col].values > 0).astype(np.float32)
        if matched_col != label_col:
            print(f"  [Info] Matched label column '{matched_col}' (requested '{label_col}')")
    else:
        # 取第一个非时间戳列
        candidates = [c for c in df_all.columns
                      if c.lower() not in {"timestamp", "time"}]
        if candidates:
            print(f"  [Warning] '{label_col}' not found, using column '{candidates[0]}'")
            labels = (df_all[candidates[0]].values > 0).astype(np.float32)
        else:
            raise ValueError(f"Cannot locate label column in {label_paths}")
    return labels


def detect_version(data_dir):
    """
    根据目录内文件命名自动判断版本，返回
    (train_files, test_files, separate_labels)
    separate_labels: 独立标签文件列表（可为空列表）
    """
    # --- hai-23.05: hai-train*.csv + label-test*.csv ---
    train = find_files(data_dir, "hai-train*.csv", "hai-train*.csv.gz")
    test  = find_files(data_dir, "hai-test*.csv",  "hai-test*.csv.gz")
    labels = find_files(data_dir, "label-test*.csv")
    if train and test:
        return train, test, labels

    # --- haiend-23.05: end-train*.csv + label-test*.csv ---
    train = find_files(data_dir, "end-train*.csv", "end-train*.csv.gz")
    test  = find_files(data_dir, "end-test*.csv",  "end-test*.csv.gz")
    labels = find_files(data_dir, "label-test*.csv")
    if train and test:
        return train, test, labels

    # --- hai-20.07 / hai-21.03 / hai-22.04: train*.csv[.gz] ---
    train = find_files(data_dir, "train*.csv.gz", "train*.csv")
    test  = find_files(data_dir, "test*.csv.gz",  "test*.csv")
    if train and test:
        return train, test, []

    return [], [], []


def prepare(data_dir, output_dir, label_col="Attack"):
    os.makedirs(output_dir, exist_ok=True)

    train_files, test_files, label_files = detect_version(data_dir)

    if not train_files:
        raise FileNotFoundError(
            f"No training CSV files found in '{data_dir}'.\n"
            "Supported patterns: train*.csv[.gz] / hai-train*.csv / end-train*.csv"
        )
    if not test_files:
        raise FileNotFoundError(
            f"No test CSV files found in '{data_dir}'.\n"
            "Supported patterns: test*.csv[.gz] / hai-test*.csv / end-test*.csv"
        )

    print(f"\nTrain files ({len(train_files)}): {[os.path.basename(f) for f in train_files]}")
    print(f"Test  files ({len(test_files)}):  {[os.path.basename(f) for f in test_files]}")
    if label_files:
        print(f"Label files ({len(label_files)}): {[os.path.basename(f) for f in label_files]}")

    # ---- 训练数据（Attack 列全为 0，正常运行）----
    print("\nLoading training data ...")
    train_data, _, feature_cols = read_csvs(train_files, label_col)
    print(f"  shape={train_data.shape}")

    # ---- 测试数据 ----
    print("\nLoading test data ...")
    test_data, inline_labels, _ = read_csvs(test_files, label_col)
    print(f"  shape={test_data.shape}")

    # ---- 标签：优先使用独立标签文件 ----
    if label_files:
        print("\nLoading labels from separate label files ...")
        test_labels = read_label_files(label_files, label_col)
        # 若标签行数与测试数据不一致则截断/填充
        n_test = len(test_data)
        n_lbl  = len(test_labels)
        if n_lbl != n_test:
            print(f"  [Warning] Label rows ({n_lbl}) != test rows ({n_test}), adjusting ...")
            if n_lbl > n_test:
                test_labels = test_labels[:n_test]
            else:
                test_labels = np.concatenate([test_labels, np.zeros(n_test - n_lbl, dtype=np.float32)])
    else:
        test_labels = inline_labels

    print(f"\nAttack ratio in test: {test_labels.mean():.4f}  "
          f"({int(test_labels.sum())} / {len(test_labels)} samples)")

    # ---- 标准化（用训练集 fit）----
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data).astype(np.float32)
    test_data  = scaler.transform(test_data).astype(np.float32)

    # ---- 保存 ----
    np.save(os.path.join(output_dir, "HAI_train.npy"),      train_data)
    np.save(os.path.join(output_dir, "HAI_test.npy"),       test_data)
    np.save(os.path.join(output_dir, "HAI_test_label.npy"), test_labels)

    print(f"\nSaved to '{output_dir}':")
    print(f"  HAI_train.npy      shape={train_data.shape}")
    print(f"  HAI_test.npy       shape={test_data.shape}")
    print(f"  HAI_test_label.npy shape={test_labels.shape}")
    print(f"\n>>> Number of features (input_c / output_c): {train_data.shape[1]}")
    print("    Use this value for --input_c and --output_c when running main.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare HAI data for Anomaly-Transformer")
    parser.add_argument("--data_dir",   type=str, default="data/HAI/hai-22.04",
                        help="Directory containing raw HAI CSV files for ONE version "
                             "(e.g. data/HAI/hai-22.04 or data/HAI/hai-23.05)")
    parser.add_argument("--output_dir", type=str, default="data/HAI",
                        help="Directory to save HAI_train/test/label .npy files")
    parser.add_argument("--label_col",  type=str, default="Attack",
                        help="Name of the attack label column (default: Attack)")
    args = parser.parse_args()

    prepare(args.data_dir, args.output_dir, args.label_col)
