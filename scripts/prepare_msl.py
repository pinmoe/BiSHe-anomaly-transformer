"""
MSL 数据预处理脚本
把 data/MSL/train/*.npy + test/*.npy 合并为代码需要的三个文件：
  data/MSL/MSL_train.npy
  data/MSL/MSL_test.npy
  data/MSL/MSL_test_label.npy
"""
import os
import ast
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'MSL')
LABEL_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'labeled_anomalies.csv')

# 读取异常标注表，筛选 MSL 部分
label_df = pd.read_csv(LABEL_CSV)
msl_df = label_df[label_df['spacecraft'] == 'MSL'].copy()
msl_df = msl_df.set_index('chan_id')

# 获取所有子文件名（按字母排序保证一致性）
train_dir = os.path.join(DATA_DIR, 'train')
test_dir = os.path.join(DATA_DIR, 'test')
chan_ids = sorted([f.replace('.npy', '') for f in os.listdir(train_dir) if f.endswith('.npy')])

print(f"找到 {len(chan_ids)} 个 MSL 子通道: {chan_ids}")

# 合并训练数据
train_arrays = []
for cid in chan_ids:
    arr = np.load(os.path.join(train_dir, f'{cid}.npy'))
    train_arrays.append(arr)
    print(f"  train/{cid}.npy: shape={arr.shape}")

train_all = np.concatenate(train_arrays, axis=0)
print(f"\n合并后 MSL_train.npy: shape={train_all.shape}")

# 合并测试数据，同时生成逐点标签
test_arrays = []
label_arrays = []
for cid in chan_ids:
    arr = np.load(os.path.join(test_dir, f'{cid}.npy'))
    test_arrays.append(arr)
    
    # 生成该子通道的标签向量
    n = arr.shape[0]
    labels = np.zeros(n, dtype=np.int32)
    
    if cid in msl_df.index:
        anomaly_seqs = ast.literal_eval(msl_df.loc[cid, 'anomaly_sequences'])
        for seq in anomaly_seqs:
            start, end = seq[0], seq[1]
            labels[start:end+1] = 1
        anomaly_ratio = labels.sum() / n * 100
        print(f"  test/{cid}.npy: shape={arr.shape}, 异常比例={anomaly_ratio:.1f}%")
    else:
        print(f"  test/{cid}.npy: shape={arr.shape}, 无异常标注（全部标记为正常）")
    
    label_arrays.append(labels)

test_all = np.concatenate(test_arrays, axis=0)
label_all = np.concatenate(label_arrays, axis=0)
print(f"\n合并后 MSL_test.npy: shape={test_all.shape}")
print(f"合并后 MSL_test_label.npy: shape={label_all.shape}, 异常比例={label_all.sum()/len(label_all)*100:.1f}%")

# 保存
np.save(os.path.join(DATA_DIR, 'MSL_train.npy'), train_all)
np.save(os.path.join(DATA_DIR, 'MSL_test.npy'), test_all)
np.save(os.path.join(DATA_DIR, 'MSL_test_label.npy'), label_all)

print("\n完成！已生成：")
print(f"  {os.path.join(DATA_DIR, 'MSL_train.npy')}")
print(f"  {os.path.join(DATA_DIR, 'MSL_test.npy')}")
print(f"  {os.path.join(DATA_DIR, 'MSL_test_label.npy')}")
