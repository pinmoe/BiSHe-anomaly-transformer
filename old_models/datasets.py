import ast
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os


# BATADAL 常见列名
_BATADAL_TIME_COLS = {'DATETIME', 'TIMESTAMP', 'DATE', 'TIME'}
_BATADAL_LABEL_COLS = ('ATT_FLAG', 'attack', 'label', 'anomaly')


def _normalize_batadal_col(name):
    return str(name).strip()


def _prepare_batadal_dataframe(df, label_col=None):
    """清洗 BATADAL DataFrame，返回 (features_df, labels_or_none)。"""
    df = df.copy()
    df.columns = [_normalize_batadal_col(c) for c in df.columns]

    # 删除时间列
    drop_cols = [c for c in df.columns if str(c).strip().upper() in _BATADAL_TIME_COLS]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')

    effective_label_col = label_col
    if effective_label_col is None:
        lower_map = {str(c).strip().lower(): c for c in df.columns}
        for cand in _BATADAL_LABEL_COLS:
            if cand.lower() in lower_map:
                effective_label_col = lower_map[cand.lower()]
                break

    labels = None
    if effective_label_col is not None and effective_label_col in df.columns:
        raw = pd.to_numeric(df.pop(effective_label_col), errors='coerce').fillna(0.0).to_numpy()
        # BATADAL 中 ATT_FLAG=1 表示攻击，其余（如 -999/0）按正常处理。
        labels = (raw == 1).astype(np.float32)

    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)
    if df.isna().values.any():
        df = df.ffill().bfill().fillna(df.median(axis=0)).fillna(0.0)

    return df, labels



class IndustrialDataset(Dataset):
    def __init__(
        self,
        data_dir,
        window_size=100,
        mode='train',
        train_mean=None,
        train_std=None,
        dataset_type='auto',
        machine_id=None,
        normalize_per_series: bool = False,
        train_per_series_stats=None,
    ):
        """
        工业时序数据集加载器，支持四类目录结构：
        1) PSM:  train.csv / test.csv / test_label.csv
        2) SMD:  train/*.txt / test/*.txt / test_label/*.txt
        3) SMAP/MSL: data/train/*.npy / data/test/*.npy / labeled_anomalies.csv
        4) BATADAL: train/BATADAL_dataset03.csv（全正常）+ train/BATADAL_dataset04.csv（含 ATT_FLAG）

        参数说明:
        - data_dir: 数据集根目录，例如 data/raw/PSM、data/raw/SMAP、data/raw/BATADAL
        - window_size: 滑动窗口大小
        - mode: 'train' 或 'test'
        - train_mean/train_std: 测试集归一化时使用的训练集统计量
        - dataset_type: 'auto' | 'psm' | 'smd' | 'smap' | 'msl' | 'batadal'
        - machine_id: 仅在 SMD/SMAP 下使用（通道 ID，例如 'P-1'）；为 None 时加载全部
        - normalize_per_series: 是否对每个序列独立计算均值/标准差（仅 smap/msl 有效）
        - train_per_series_stats: 测试模式下传入训练集的逐序列统计量列表 [(mean, std), ...]
        """
        if mode not in ('train', 'test'):
            raise ValueError("mode 必须是 'train' 或 'test'。")
        if window_size <= 0:
            raise ValueError('window_size 必须是正整数。')

        self.window_size = int(window_size)
        self.mode = mode
        self.data_dir = data_dir
        self.normalize_per_series = bool(normalize_per_series)

        # 自动识别数据集类型，避免手动配置错误。
        self.dataset_type = self._resolve_dataset_type(data_dir, dataset_type)

        # series_data: 每个元素是一台机器/一段序列，形状 (T, C)
        # series_labels: 仅 test 模式使用，每个元素形状 (T,)
        self.series_data = []
        self.series_labels = []
        self.series_names = []

        if self.dataset_type == 'psm':
            self._load_psm()
        elif self.dataset_type == 'smd':
            self._load_smd(machine_id=machine_id)
        elif self.dataset_type in {'smap', 'msl'}:
            self._load_smap(channel_id=machine_id)
        elif self.dataset_type == 'batadal':
            self._load_batadal()

        # 统一归一化流程：训练集自己算；测试集使用训练集传入统计量。
        self._normalize(train_mean=train_mean, train_std=train_std,
                        per_series=normalize_per_series,
                        train_per_series_stats=train_per_series_stats)

        # 为多序列构建全局窗口索引，确保窗口不会跨序列（SMD/SMAP 下不会跨通道文件）。
        self._build_window_index()

        # 为兼容旧代码，单序列时保留 data / labels 这两个属性。
        if len(self.series_data) == 1:
            self.data = self.series_data[0]
            if self.mode == 'test':
                self.labels = self.series_labels[0]

    def _resolve_dataset_type(self, data_dir, dataset_type):
        valid = ('auto', 'psm', 'smd', 'smap', 'msl', 'batadal')
        if dataset_type not in valid:
            raise ValueError(f"dataset_type 必须是 {valid} 之一。")

        if dataset_type != 'auto':
            return dataset_type

        has_psm_files = (
            os.path.exists(os.path.join(data_dir, 'train.csv'))
            and os.path.exists(os.path.join(data_dir, 'test.csv'))
        )
        has_smd_dirs = (
            os.path.isdir(os.path.join(data_dir, 'train'))
            and os.path.isdir(os.path.join(data_dir, 'test'))
            and not os.path.exists(os.path.join(data_dir, 'labeled_anomalies.csv'))
        )
        has_smap = os.path.exists(os.path.join(data_dir, 'labeled_anomalies.csv'))
        has_batadal = (
            os.path.exists(os.path.join(data_dir, 'BATADAL_dataset04.csv'))
            or os.path.exists(os.path.join(data_dir, 'BATADAL_test_dataset.csv'))
            or os.path.exists(os.path.join(data_dir, 'train', 'BATADAL_dataset03.csv'))
            or os.path.exists(os.path.join(data_dir, 'train', 'BATADAL_dataset04.csv'))
        )

        if has_psm_files:
            return 'psm'
        if has_smap:
            dataset_name = os.path.basename(os.path.normpath(data_dir)).strip().lower()
            if dataset_name == 'msl':
                return 'msl'
            return 'smap'
        if has_batadal:
            return 'batadal'
        if has_smd_dirs:
            return 'smd'

        raise ValueError(
            f"无法自动识别数据集类型，请检查目录: {data_dir}，"
            "或手动指定 dataset_type='psm'/'smd'/'smap'/'msl'/'batadal'。"
        )

    def _load_batadal(self):
        """加载 BATADAL。

        目录结构：
            BATADAL/
              train/
                BATADAL_dataset03.csv  — 全正常样本（ATT_FLAG=0），用于训练
                BATADAL_dataset04.csv  — 含攻击标签（ATT_FLAG: -999=正常, 1=攻击）
              test/
                BATADAL_test_dataset.csv — 无标签（竞赛盲测集）

        设计策略：
        - train: 使用 dataset03（全正常），若不存在则回退 dataset04 的正常样本
        - test : 使用 dataset04（含 ATT_FLAG 可评估），若不存在则回退 test_dataset

        同时兼容旧的扁平目录结构（文件直接放在 data_dir 下）。
        """
        # 新目录结构路径
        train_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'test')

        # 候选文件路径（优先新结构，回退旧结构）
        ds03_path = os.path.join(train_dir, 'BATADAL_dataset03.csv')
        ds04_path_new = os.path.join(train_dir, 'BATADAL_dataset04.csv')
        ds04_path_old = os.path.join(self.data_dir, 'BATADAL_dataset04.csv')
        test_path_new = os.path.join(test_dir, 'BATADAL_test_dataset.csv')
        test_path_old = os.path.join(self.data_dir, 'BATADAL_test_dataset.csv')

        # 实际使用的 dataset04 路径
        ds04_path = ds04_path_new if os.path.exists(ds04_path_new) else ds04_path_old
        test_path = test_path_new if os.path.exists(test_path_new) else test_path_old

        def _read(path):
            df = pd.read_csv(path, skipinitialspace=True)
            return _prepare_batadal_dataframe(df)

        if self.mode == 'train':
            # 优先使用 dataset03（全正常，无需过滤）
            if os.path.exists(ds03_path):
                feat_df, labels = _read(ds03_path)
                arr = feat_df.to_numpy(dtype=np.float32)
                if len(arr) <= self.window_size:
                    raise ValueError('BATADAL dataset03 样本不足，请检查文件或 window_size。')
                # dataset03 全正常，labels 可能全 0 或 None，无需过滤
                self.series_data.append(arr)
                self.series_names.append('batadal_train_normal')
                print(f'  [BATADAL] 训练集使用 dataset03（全正常），样本数={len(arr)}')
                return

            # 回退：使用 dataset04 的正常样本
            if not os.path.exists(ds04_path):
                raise FileNotFoundError(
                    f'在 {self.data_dir} 下未找到 BATADAL 训练文件。'
                    f'期望 train/BATADAL_dataset03.csv 或 BATADAL_dataset04.csv。'
                )
            feat_df, labels = _read(ds04_path)
            if labels is None:
                raise ValueError('BATADAL_dataset04.csv 缺少 ATT_FLAG，无法构造训练正常样本。')
            normal_mask = labels == 0
            arr = feat_df.to_numpy(dtype=np.float32)[normal_mask]
            if len(arr) <= self.window_size:
                raise ValueError('BATADAL 训练正常样本不足，请检查 ATT_FLAG 或 window_size。')
            self.series_data.append(arr)
            self.series_names.append('batadal_train_normal')
            print(f'  [BATADAL] 训练集回退使用 dataset04 正常样本，样本数={len(arr)}')
            return

        # ── test 模式 ──
        # 优先使用 dataset04（含 ATT_FLAG 标签，可计算 F1 等指标）
        if os.path.exists(ds04_path):
            feat_df, labels = _read(ds04_path)
            if labels is not None:
                arr = feat_df.to_numpy(dtype=np.float32)
                self.series_data.append(arr)
                self.series_names.append('batadal_test')
                self.series_labels.append(labels.astype(np.float32))
                print(f'  [BATADAL] 测试集使用 dataset04（含标签），样本数={len(arr)}，'
                      f'攻击样本={int(np.sum(labels == 1))}')
                return

        # 回退：使用无标签的 test_dataset
        if os.path.exists(test_path):
            feat_df, labels = _read(test_path)
            arr = feat_df.to_numpy(dtype=np.float32)
            self.series_data.append(arr)
            self.series_names.append('batadal_test')
            self.series_labels.append(np.zeros(len(arr), dtype=np.float32))
            print('  [BATADAL] 测试集使用 test_dataset（无标签），标签回退为全零。评估结果仅供误报参考。')
            return

        raise FileNotFoundError(
            f'在 {self.data_dir} 下未找到 BATADAL 测试文件。'
            f'期望 train/BATADAL_dataset04.csv 或 test/BATADAL_test_dataset.csv。'
        )

    def _load_psm(self):
        # PSM 是单文件时序，train/test 都是 CSV。
        csv_name = 'train.csv' if self.mode == 'train' else 'test.csv'
        csv_path = os.path.join(self.data_dir, csv_name)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'找不到文件: {csv_path}')

        df = pd.read_csv(csv_path)

        # PSM 第一列通常是时间戳（如 timestamp_(min)），不作为模型特征。
        first_col = str(df.columns[0]).lower() if len(df.columns) > 0 else ''
        if any(token in first_col for token in ('timestamp', 'time', 'date')):
            df = df.iloc[:, 1:]

        # PSM 训练集中常见少量 NaN，统一做数值化与缺失值清洗，避免统计量被污染。
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.replace([np.inf, -np.inf], np.nan)
        if df.isna().values.any():
            df = df.ffill().bfill().fillna(df.median(axis=0)).fillna(0.0)

        features = df.values.astype(np.float32)
        self.series_data = [features]
        self.series_names = ['psm']

        if self.mode == 'test':
            label_path = os.path.join(self.data_dir, 'test_label.csv')
            if not os.path.exists(label_path):
                raise FileNotFoundError(f'找不到标签文件: {label_path}')

            label_df = pd.read_csv(label_path)
            # 常见格式是两列: timestamp + label。这里统一取最后一列作为标签。
            labels = label_df.iloc[:, -1].to_numpy(dtype=np.float32).reshape(-1)

            if len(labels) != len(features):
                raise ValueError(
                    f'PSM 测试标签长度({len(labels)})与测试数据长度({len(features)})不一致。'
                )
            self.series_labels = [labels]

    def _load_smd(self, machine_id=None):
        # SMD 是按机器拆分的多文件时序。
        data_subdir = 'train' if self.mode == 'train' else 'test'
        data_path = os.path.join(self.data_dir, data_subdir)
        if not os.path.isdir(data_path):
            raise FileNotFoundError(f'找不到目录: {data_path}')

        all_files = sorted([f for f in os.listdir(data_path) if f.endswith('.txt')])
        if machine_id is not None:
            target_file = machine_id if machine_id.endswith('.txt') else f'{machine_id}.txt'
            all_files = [f for f in all_files if f == target_file]

        if not all_files:
            raise ValueError('没有找到可用的 SMD 数据文件，请检查 machine_id 或目录内容。')

        label_dir = os.path.join(self.data_dir, 'test_label')

        for file_name in all_files:
            file_path = os.path.join(data_path, file_name)
            arr = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)

            self.series_data.append(arr)
            self.series_names.append(file_name.replace('.txt', ''))

            if self.mode == 'test':
                label_path = os.path.join(label_dir, file_name)
                if not os.path.exists(label_path):
                    raise FileNotFoundError(f'找不到 SMD 标签文件: {label_path}')

                labels = np.loadtxt(label_path, dtype=np.float32).reshape(-1)
                if len(labels) != len(arr):
                    raise ValueError(
                        f'SMD 文件 {file_name} 的标签长度({len(labels)})与数据长度({len(arr)})不一致。'
                    )
                self.series_labels.append(labels)

    def _load_smap(self, channel_id=None):
        # SMAP 数据以 .npy 格式按通道分别存储在 data/train/ 和 data/test/ 下。
        # 标签通过 labeled_anomalies.csv 中的 anomaly_sequences 字段（异常区间列表）还原。
        npy_subdir = 'train' if self.mode == 'train' else 'test'
        # 兼容两种目录布局：
        # 1) <root>/data/train/*.npy（原始 NASA）
        # 2) <root>/train/*.npy（分离后的 SMAP/MSL）
        data_path_candidates = [
            os.path.join(self.data_dir, 'data', npy_subdir),
            os.path.join(self.data_dir, npy_subdir),
        ]
        data_path = next((p for p in data_path_candidates if os.path.isdir(p)), None)
        if data_path is None:
            raise FileNotFoundError(
                '找不到 SMAP/MSL 数据目录，已尝试: ' + ', '.join(data_path_candidates)
            )

        all_files = sorted([f for f in os.listdir(data_path) if f.endswith('.npy')])

        # 兼容 NASA 数据目录：某些发布版本会在同一目录同时包含 SMAP 与 MSL 通道。
        # 若 data_dir 名称明确为 SMAP / MSL，则按 labeled_anomalies.csv 的 spacecraft 列做过滤。
        # 标签文件允许放在数据集根目录、其父目录或 data 子目录。
        label_csv_candidates = [
            os.path.join(self.data_dir, 'labeled_anomalies.csv'),
            os.path.join(os.path.dirname(self.data_dir), 'labeled_anomalies.csv'),
            os.path.join(self.data_dir, 'data', 'labeled_anomalies.csv'),
        ]
        label_csv = next((p for p in label_csv_candidates if os.path.exists(p)), None)
        if label_csv is not None:
            label_df_full = pd.read_csv(label_csv)
            if 'spacecraft' in label_df_full.columns and 'chan_id' in label_df_full.columns:
                dataset_name = os.path.basename(os.path.normpath(self.data_dir)).upper()
                if dataset_name in ('SMAP', 'MSL'):
                    allowed_chan = set(
                        label_df_full[label_df_full['spacecraft'].astype(str).str.upper() == dataset_name]['chan_id']
                        .astype(str)
                        .str.strip()
                        .tolist()
                    )
                    if allowed_chan:
                        all_files = [f for f in all_files if f.replace('.npy', '') in allowed_chan]

        if channel_id is not None:
            target = channel_id if channel_id.endswith('.npy') else f'{channel_id}.npy'
            all_files = [f for f in all_files if f == target]

        if not all_files:
            raise ValueError('没有找到可用的 SMAP .npy 文件，请检查 channel_id 或目录内容。')

        # 解析标签文件，建立 channel_id -> 异常区间列表 的映射。
        anomaly_map = {}
        if self.mode == 'test':
            if label_csv is None:
                raise FileNotFoundError(
                    '找不到 SMAP/MSL 标签文件 labeled_anomalies.csv，已尝试: '
                    + ', '.join(label_csv_candidates)
                )
            label_df = pd.read_csv(label_csv)
            for _, row in label_df.iterrows():
                cid = str(row['chan_id']).strip()
                seqs = ast.literal_eval(str(row['anomaly_sequences']))
                num_values = int(row['num_values'])
                anomaly_map[cid] = (seqs, num_values)

        for file_name in all_files:
            cid = file_name.replace('.npy', '')
            arr = np.load(os.path.join(data_path, file_name)).astype(np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)

            # 过滤低维通道，避免 DGR 在极低维输入下退化。
            if arr.shape[1] < 5:
                continue

            self.series_data.append(arr)
            self.series_names.append(cid)

            if self.mode == 'test':
                t = len(arr)
                labels = np.zeros(t, dtype=np.float32)
                if cid in anomaly_map:
                    seqs, _num = anomaly_map[cid]
                    for start, end in seqs:
                        # labeled_anomalies.csv 里的区间是基于测试集长度的索引，闭区间。
                        labels[int(start): int(end) + 1] = 1.0
                self.series_labels.append(labels)

        if not self.series_data:
            raise ValueError('SMAP/MSL 过滤后无可用通道（要求特征维度 >= 5），请检查数据与过滤阈值。')

    def _normalize(self, train_mean=None, train_std=None,
                   per_series=False, train_per_series_stats=None):
        # 先做有限值清洗，避免 NaN/Inf 传播到均值方差统计。
        self.series_data = [np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32) for x in self.series_data]

        # 分通道归一化（仅 smap/msl 有效，减少跨通道量纲主导问题）
        if per_series and self.dataset_type in {'smap', 'msl'}:
            if self.mode == 'train':
                self.per_series_stats: list[tuple[np.ndarray, np.ndarray]] = []
                for i, x in enumerate(self.series_data):
                    m = x.mean(axis=0).astype(np.float32)
                    s = x.std(axis=0).astype(np.float32)
                    safe_s = np.where(s < 1e-8, 1.0, s).astype(np.float32)
                    self.series_data[i] = ((x - m) / safe_s).astype(np.float32)
                    self.per_series_stats.append((m, safe_s))
                # 同时保留整体统计量，供外部接口兼容
                stacked = np.concatenate(self.series_data, axis=0)
                self.mean = np.mean(stacked, axis=0).astype(np.float32)
                self.std = np.ones_like(self.mean)   # 已按序列归一化，整体 std=1
            else:
                if train_per_series_stats is None:
                    raise ValueError('SMAP/MSL 分通道测试归一化需传入 train_per_series_stats。')
                self.per_series_stats = train_per_series_stats
                for i, x in enumerate(self.series_data):
                    m, s = train_per_series_stats[i]
                    self.series_data[i] = ((x - m) / s).astype(np.float32)
                # 整体统计量置为 0/1（已按序列归一化，无需整体再归一化）
                if len(self.series_data) > 0:
                    c = self.series_data[0].shape[1]
                    self.mean = np.zeros(c, dtype=np.float32)
                    self.std = np.ones(c, dtype=np.float32)
                else:
                    self.mean = np.zeros(1, dtype=np.float32)
                    self.std = np.ones(1, dtype=np.float32)
            return

        if self.mode == 'train':
            # 训练集统计量按“所有训练点整体”计算，避免分机器统计造成分布不一致。
            stacked = np.concatenate(self.series_data, axis=0)
            self.mean = np.mean(stacked, axis=0).astype(np.float32)
            self.std = np.std(stacked, axis=0).astype(np.float32)
        else:
            if train_mean is None or train_std is None:
                raise ValueError('测试模式必须传入 train_mean 和 train_std。')
            self.mean = np.asarray(train_mean, dtype=np.float32)
            self.std = np.asarray(train_std, dtype=np.float32)

        self.mean = np.nan_to_num(self.mean, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self.std = np.nan_to_num(self.std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # 防止某个通道标准差为 0 导致除零。
        safe_std = np.where(self.std < 1e-8, 1.0, self.std).astype(np.float32)
        self.series_data = [((x - self.mean) / safe_std).astype(np.float32) for x in self.series_data]

        # BATADAL/PSM/SMD/SMAP 统一不做额外裁剪，保留原始标准化分布。

    def _build_window_index(self):
        # 每段序列的可滑窗数量，长度小于 window_size 的序列会被自动忽略。
        self.window_counts = [max(0, len(x) - self.window_size + 1) for x in self.series_data]
        self.cum_windows = np.cumsum(self.window_counts, dtype=np.int64)

        total = int(self.cum_windows[-1]) if len(self.cum_windows) > 0 else 0
        if total <= 0:
            raise ValueError(
                f'没有可用窗口，请检查 window_size={self.window_size} 是否过大。'
            )

    def _locate_window(self, index):
        # 将全局索引映射到 (第几段序列, 序列内起点)。
        if index < 0 or index >= len(self):
            raise IndexError('索引超出范围。')

        seq_idx = int(np.searchsorted(self.cum_windows, index, side='right'))
        prev_cum = 0 if seq_idx == 0 else int(self.cum_windows[seq_idx - 1])
        local_start = int(index - prev_cum)
        return seq_idx, local_start

    def __len__(self):
        # 所有序列可滑窗数之和。
        return int(self.cum_windows[-1]) if len(self.cum_windows) > 0 else 0

    def __getitem__(self, index):
        seq_idx, start = self._locate_window(index)
        series = self.series_data[seq_idx]

        # 切出一个窗口: (window_size, channels)
        window_data = series[start : start + self.window_size]
        tensor_data = torch.from_numpy(window_data)

        # 双路输入格式：
        # - x_2d: (Seq_Len, Channels) 例如给 Transformer/ViT 分支
        # - x_1d: (Channels, Seq_Len) 例如给 TCN/CNN1D 分支
        x_2d = tensor_data
        x_1d = tensor_data.transpose(0, 1)

        if self.mode == 'test':
            # 窗口标签取窗口最后一个时间点，和常见异常检测评估方式一致。
            label = self.series_labels[seq_idx][start + self.window_size - 1]
            label = torch.tensor(label, dtype=torch.float32)
            return x_1d, x_2d, label
        return x_1d, x_2d


def build_train_test_datasets(
    data_dir,
    window_size=100,
    dataset_type='auto',
    machine_id=None,
):
    """
    一键构建训练集和测试集，并自动把训练集统计量传给测试集。

    这是最常用的入口，适合在训练脚本里直接调用。
    SMAP/MSL 自动启用 normalize_per_series=True，减少跨通道量纲主导问题。
    """
    # SMAP/MSL 自动启用分通道归一化
    _ds_type_lower = str(dataset_type).lower()
    _use_per_series = _ds_type_lower in {'smap', 'msl'}

    train_dataset = IndustrialDataset(
        data_dir=data_dir,
        window_size=window_size,
        mode='train',
        dataset_type=dataset_type,
        machine_id=machine_id,
        normalize_per_series=_use_per_series,
    )

    # 若 auto 检测后已确定为 smap/msl，也启用分通道归一化
    _actual_type = getattr(train_dataset, 'dataset_type', _ds_type_lower)
    if _actual_type in {'smap', 'msl'} and not _use_per_series:
        _use_per_series = True

    _train_per_series_stats = getattr(train_dataset, 'per_series_stats', None) if _use_per_series else None

    test_dataset = IndustrialDataset(
        data_dir=data_dir,
        window_size=window_size,
        mode='test',
        train_mean=train_dataset.mean,
        train_std=train_dataset.std,
        dataset_type=dataset_type,
        machine_id=machine_id,
        normalize_per_series=_use_per_series,
        train_per_series_stats=_train_per_series_stats,
    )

    return train_dataset, test_dataset


def build_train_test_loaders(
    data_dir,
    window_size=100,
    batch_size=64,
    dataset_type='auto',
    machine_id=None,
    num_workers=0,
    pin_memory=False,
    shuffle_train=True,
    shuffle_test=False,
    drop_last_train=False,
    drop_last_test=False,
):
    """
    一键构建训练/测试 DataLoader。

    返回:
    - train_loader
    - test_loader
    - meta: 训练统计量和基础信息，便于保存到 checkpoint
    """
    train_dataset, test_dataset = build_train_test_datasets(
        data_dir=data_dir,
        window_size=window_size,
        dataset_type=dataset_type,
        machine_id=machine_id,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_test,
    )

    meta = {
        'mean': train_dataset.mean,
        'std': train_dataset.std,
        'dataset_type': train_dataset.dataset_type,
        'series_names': train_dataset.series_names,
        'window_size': train_dataset.window_size,
        'num_features': int(train_dataset.series_data[0].shape[1]),
    }

    return train_loader, test_loader, meta
    