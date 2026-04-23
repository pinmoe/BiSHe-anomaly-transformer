"""
prepare_batadal.py
==================
将 BATADAL 原始 CSV 文件转换为 .npy 格式，供 BATADALSegLoader 使用。

数据集说明
----------
- BATADAL_dataset03.csv  : 训练集，全部为正常数据（ATT_FLAG=0）
- BATADAL_dataset04.csv  : 测试集，含攻击标签
    ATT_FLAG =  1   → 攻击（异常）
    ATT_FLAG = -999 → 竞赛未公开标注段，按 BATADAL 论文惯例映射为 0（正常）
- BATADAL_test_dataset.csv : 竞赛盲测集，无标签，不参与 F1 计算

用法
----
python prepare_batadal.py \
    --src_dir  /path/to/raw_csvs \
    --dst_dir  /path/to/dataset/BATADAL

输出文件（dst_dir 下）
--------------------
  BATADAL_train.npy        shape: (8761, 43)  float32，StandardScaler 归一化后
  BATADAL_test.npy         shape: (4177, 43)  float32，同一 scaler 归一化
  BATADAL_test_label.npy   shape: (4177,)     int32，0/1
  batadal_scaler.pkl       scaler 对象（可选，供推理时复用）
"""

import os
import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# 43 个传感器特征列（去掉 DATETIME 和 ATT_FLAG）
FEATURE_COLS = [
    'L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7',
    'F_PU1', 'S_PU1', 'F_PU2', 'S_PU2', 'F_PU3', 'S_PU3',
    'F_PU4', 'S_PU4', 'F_PU5', 'S_PU5', 'F_PU6', 'S_PU6',
    'F_PU7', 'S_PU7', 'F_PU8', 'S_PU8', 'F_PU9', 'S_PU9',
    'F_PU10', 'S_PU10', 'F_PU11', 'S_PU11',
    'F_V2', 'S_V2',
    'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415',
    'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422',
]


def load_csv(path: str) -> pd.DataFrame:
    """读取 CSV 并统一去除列名首尾空格（dataset04 存在前导空格）。"""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def map_labels(att_flag: pd.Series) -> np.ndarray:
    """
    BATADAL 标签映射规则：
      -999 → 0  （未标注段，视为正常，保持时序连续性）
         1 → 1  （攻击）
         0 → 0  （正常，dataset03 全部为此值）
    """
    labels = att_flag.copy()
    labels[labels == -999] = 0
    return labels.values.astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description='Prepare BATADAL dataset for Anomaly Transformer')
    parser.add_argument('--src_dir', type=str, required=True,
                        help='Directory containing the three raw BATADAL CSV files')
    parser.add_argument('--dst_dir', type=str, required=True,
                        help='Output directory for .npy files')
    parser.add_argument('--save_scaler', action='store_true',
                        help='Save fitted StandardScaler to dst_dir/batadal_scaler.pkl')
    args = parser.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. 读取原始 CSV
    # ------------------------------------------------------------------ #
    train_csv = os.path.join(args.src_dir, 'BATADAL_dataset03.csv')
    test_csv  = os.path.join(args.src_dir, 'BATADAL_dataset04.csv')

    print(f'[1/4] Loading CSVs from {args.src_dir} ...')
    df_train = load_csv(train_csv)
    df_test  = load_csv(test_csv)

    # ------------------------------------------------------------------ #
    # 2. 提取特征矩阵
    # ------------------------------------------------------------------ #
    # 验证特征列全部存在
    for col in FEATURE_COLS:
        assert col in df_train.columns, f'Missing column in train: {col}'
        assert col in df_test.columns,  f'Missing column in test:  {col}'

    X_train = df_train[FEATURE_COLS].values.astype(np.float32)
    X_test  = df_test[FEATURE_COLS].values.astype(np.float32)

    # NaN 填充（BATADAL 无 NaN，但防御性处理）
    X_train = np.nan_to_num(X_train)
    X_test  = np.nan_to_num(X_test)

    print(f'    Train shape: {X_train.shape}')   # (8761, 43)
    print(f'    Test  shape: {X_test.shape}')    # (4177, 43)

    # ------------------------------------------------------------------ #
    # 3. 标签处理
    # ------------------------------------------------------------------ #
    y_test = map_labels(df_test['ATT_FLAG'])
    print(f'[2/4] Test labels → 0: {(y_test==0).sum()}, 1: {(y_test==1).sum()}')
    print(f'      Anomaly ratio: {y_test.mean()*100:.2f}%')

    # ------------------------------------------------------------------ #
    # 4. 归一化（仅用训练集 fit）
    # ------------------------------------------------------------------ #
    print('[3/4] Fitting StandardScaler on train set ...')
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_test_scaled  = scaler.transform(X_test).astype(np.float32)

    # ------------------------------------------------------------------ #
    # 5. 保存
    # ------------------------------------------------------------------ #
    print(f'[4/4] Saving .npy files to {args.dst_dir} ...')
    np.save(os.path.join(args.dst_dir, 'BATADAL_train.npy'),      X_train_scaled)
    np.save(os.path.join(args.dst_dir, 'BATADAL_test.npy'),       X_test_scaled)
    np.save(os.path.join(args.dst_dir, 'BATADAL_test_label.npy'), y_test)

    if args.save_scaler:
        scaler_path = os.path.join(args.dst_dir, 'batadal_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f'    Scaler saved to {scaler_path}')

    # 验证文件
    t  = np.load(os.path.join(args.dst_dir, 'BATADAL_train.npy'))
    te = np.load(os.path.join(args.dst_dir, 'BATADAL_test.npy'))
    lb = np.load(os.path.join(args.dst_dir, 'BATADAL_test_label.npy'))
    print('\n=== Verification ===')
    print(f'  BATADAL_train.npy      : {t.shape}  dtype={t.dtype}')
    print(f'  BATADAL_test.npy       : {te.shape}  dtype={te.dtype}')
    print(f'  BATADAL_test_label.npy : {lb.shape}  dtype={lb.dtype}  '
          f'anomaly={lb.sum()} ({lb.mean()*100:.2f}%)')
    print('\nDone. You can now run BATADALSegLoader from data_loader.py.')


if __name__ == '__main__':
    main()
