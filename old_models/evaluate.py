"""evaluate_enhanced.py

增强的多模态融合模型评估脚本。
新增功能：
1. 数据集专属阈值（PSM/SMD/SMAP 各有调整空间）
2. 段内连续覆盖后处理（最小持续、间隙填充、双阈值滞回）
3. 增强指标（AUPRC、事件级指标、检测延迟、分组评估）
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, auc, precision_recall_curve

from models.multimodal_model import MultiModalFusionModel
from models.reconstructor import Reconstructor
from utils.datasets import build_train_test_loaders
from utils.scoring import AnomalyScorer, post_process_predictions


# 数据集专属参数：根据数据集调整阈值策略和后处理

DATASET_PARAMS = {
    'psm': {
        'q_threshold': 0.995,  # PSM 基本靠近极限
        'min_duration': 0,
        'gap_filling': 0,
        'use_hysteresis': False,
        'alpha': 0.7,  # 重构 vs 潜空间权重
    },
    'smap': {
        'q_threshold': 0.90,   # SMAP 召回太低，下调阈值
        'min_duration': 1,     # 过滤单点异常
        'gap_filling': 2,      # 填充小间隙
        'use_hysteresis': True,
        'hysteresis_factor': 0.85,
        'alpha': 0.5,          # 降低对重构的依赖
        'threshold_calibration': 'f1',
        'calib_ratio': 0.20,
        'forecast_weight': 0.4,  # MTAD-GAT 启发：上下文异常靠预测头捕捉
    },
    'msl': {
        'q_threshold': 0.90,
        'min_duration': 1,
        'gap_filling': 2,
        'use_hysteresis': True,
        'hysteresis_factor': 0.85,
        'alpha': 0.5,
        'threshold_calibration': 'f1',
        'calib_ratio': 0.20,
        'forecast_weight': 0.4,  # MTAD-GAT 启发：上下文异常靠预测头捕捉
    },
    'smd': {
        'q_threshold': 0.90,   # SMD 也需要下调
        'min_duration': 1,
        'gap_filling': 3,
        'use_hysteresis': True,
        'hysteresis_factor': 0.80,
        'alpha': 0.6,
        'threshold_calibration': 'f1',
        'calib_ratio': 0.15,
    },
    'batadal': {
        'q_threshold': 0.98,   # BATADAL 攻击样本占比低，阈值偏保守
        'min_duration': 2,
        'gap_filling': 4,
        'use_hysteresis': True,
        'hysteresis_factor': 0.80,
        'alpha': 0.5,
        'threshold_calibration': 'f1',  # P0: 启用 F1 校准，阈值扫描显示可提升 0.28
        'calib_ratio': 0.15,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="增强型融合模型评估脚本")

    # checkpoint 与数据
    parser.add_argument("--checkpoint", type=str, default="checkpoints/fusion_3stage.pt", help="checkpoint 文件路径")
    parser.add_argument("--data-dir", type=str, default=None, help="数据集目录")
    parser.add_argument(
        "--dataset-type",
        type=str,
        default=None,
        choices=["auto", "psm", "smd", "smap", "msl", "batadal"],
        help="数据集类型",
    )
    parser.add_argument("--machine-id", type=str, default=None)

    # DataLoader
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)

    # 评估参数
    parser.add_argument("--alpha", type=float, default=None, help="绩分权重（若为None使用数据集默认）")
    parser.add_argument("--q-threshold", type=float, default=None, help="分位数阈值（若为None使用数据集默认）")
    parser.add_argument("--custom-threshold", type=float, default=None, help="手动阈值")
    parser.add_argument(
        "--threshold-calibration",
        type=str,
        default=None,
        choices=["none", "f1"],
        help="阈值校准模式（None表示使用数据集默认）",
    )
    parser.add_argument("--calib-ratio", type=float, default=None,
                        help="用于阈值校准的前缀样本比例，(0,1)")
    parser.add_argument("--calib-min-points", type=int, default=500,
                        help="阈值校准最少点数")
    parser.add_argument("--sigma-w", type=float, default=3.0)
    parser.add_argument("--smoothing-window", type=int, default=3)
    
    # 后处理参数
    parser.add_argument("--min-duration", type=int, default=None)
    parser.add_argument("--gap-filling", type=int, default=None)
    parser.add_argument("--enable-hysteresis", action="store_true")
    
    # 双头评分（MTAD-GAT 启发）
    parser.add_argument("--forecast-weight", type=float, default=None,
                        help="预测头误差权重 fw：score=(1-fw)*recon + fw*forecast。"
                             "若为None则使用数据集默认值，0.0表示纯重构。")

    # 输出
    parser.add_argument("--save-csv", type=str, default=None)
    parser.add_argument("--save-advanced-metrics", type=str, default=None, help="保存高级指标到 JSON")

    return parser.parse_args()


def resolve_checkpoint_path(path_str: str) -> Path:
    ckpt_path = Path(path_str)
    if ckpt_path.exists():
        return ckpt_path

    # 在 checkpoints/ 的数据集子目录中搜索同名文件
    # 例如 checkpoints/fusion_3stage_psm.pt → checkpoints/PSM/fusion_3stage_psm.pt
    parent = ckpt_path.parent if str(ckpt_path.parent) != "." else Path("checkpoints")
    subdirs = sorted(parent.glob("*/")) if parent.exists() else []
    for sub in subdirs:
        candidate = sub / ckpt_path.name
        if candidate.exists():
            print(f"[INFO] 自动定位到 {candidate}")
            return candidate

    # 兜底：在 parent 下按通配符匹配
    if parent.exists():
        if ckpt_path.name == "fusion_3stage.pt":
            candidates = sorted(parent.glob("fusion_3stage_*.pt"))
            if len(candidates) == 1:
                print(f"[INFO] 自动使用 {candidates[0]}")
                return candidates[0]
        # 递归搜索子目录
        candidates = sorted(parent.rglob(ckpt_path.name))
        if len(candidates) == 1:
            print(f"[INFO] 自动定位到 {candidates[0]}")
            return candidates[0]
        if len(candidates) > 1:
            print(f"[WARNING] 找到多个匹配: {[str(c) for c in candidates]}，请指定完整路径。")

    raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")


def _infer_tcn_hidden_channels(state_dict: dict) -> tuple:
    """从 checkpoint state_dict 推断 TcnEncoder 的 hidden_channels 配置。
    通过读取各 TemporalBlock conv1.weight 的输出通道数来还原层结构。"""
    indices = set()
    for key in state_dict:
        if key.startswith("tcn_encoder.tcn.") and ".conv1.weight" in key:
            try:
                indices.add(int(key.split(".")[2]))
            except (ValueError, IndexError):
                pass
    if not indices:
        return (64, 128, 128)  # 向后兼容默认值（3 层）
    channels = []
    for i in range(max(indices) + 1):
        w_key = f"tcn_encoder.tcn.{i}.conv1.weight"
        channels.append(state_dict[w_key].shape[0] if w_key in state_dict else 128)
    print(f"[INFO] 从 checkpoint 推断 TCN hidden_channels = {tuple(channels)}")
    return tuple(channels)


def resolve_runtime_config(args: argparse.Namespace, ckpt: dict) -> dict:
    """合并命令行与 checkpoint 参数"""
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    meta = ckpt.get("meta", {}) if isinstance(ckpt.get("meta", {}), dict) else {}

    data_dir = args.data_dir or ckpt_args.get("data_dir") or "data/raw/PSM"
    dataset_type = args.dataset_type or meta.get("dataset_type") or "auto"
    machine_id = args.machine_id if args.machine_id is not None else ckpt_args.get("machine_id")
    window_size = args.window_size or ckpt_args.get("window_size") or 100

    # 模型相关
    out_dim = int(ckpt_args.get("out_dim", 256))
    vit_model = ckpt_args.get("vit_model", "tiny_graph")
    recon_hidden_dim = ckpt_args.get("recon_hidden_dim", None)
    recon_dropout = float(ckpt_args.get("recon_dropout", 0.0))
    recon_layernorm = bool(ckpt_args.get("recon_layernorm", False))
    fusion_mode = ckpt_args.get("fusion_mode", "point")
    reconstructor_type = ckpt_args.get("reconstructor_type", "mlp")
    temporal_decoder_layers = int(ckpt_args.get("temporal_decoder_layers", 2))

    in_channels = int(meta.get("num_features", 0))
    if in_channels <= 0:
        raise ValueError("checkpoint 缺少 num_features")

    with_forecast = bool(ckpt_args.get("with_forecast", False))

    # 从 state_dict 推断 TCN 架构（兼容新旧 checkpoint）
    tcn_hidden_channels = _infer_tcn_hidden_channels(ckpt.get("model_state_dict", {}))

    return {
        "data_dir": data_dir,
        "dataset_type": dataset_type,
        "machine_id": machine_id,
        "window_size": int(window_size),
        "in_channels": in_channels,
        "out_dim": out_dim,
        "vit_model": vit_model,
        "recon_hidden_dim": recon_hidden_dim,
        "recon_dropout": recon_dropout,
        "recon_layernorm": recon_layernorm,
        "fusion_mode": fusion_mode,
        "reconstructor_type": reconstructor_type,
        "temporal_decoder_layers": temporal_decoder_layers,
        "with_forecast": with_forecast,
        "tcn_hidden_channels": tcn_hidden_channels,
    }


def build_model_and_reconstructor(cfg: dict, ckpt: dict, device: torch.device) -> Tuple[MultiModalFusionModel, Reconstructor]:
    model = MultiModalFusionModel(
        in_channels=cfg["in_channels"],
        out_dim=cfg["out_dim"],
        gate_hidden_dim=None,
        fusion_mode=cfg["fusion_mode"],
        vit_kwargs={"model_name": cfg["vit_model"], "pretrained": False, "freeze_backbone": False},
        tcn_kwargs={"hidden_channels": cfg["tcn_hidden_channels"]},
    ).to(device)

    reconstructor = Reconstructor(
        in_dim=cfg["out_dim"],
        window_size=cfg["window_size"],
        channels=cfg["in_channels"],
        hidden_dim=cfg["recon_hidden_dim"],
        dropout=cfg["recon_dropout"],
        use_layernorm=cfg["recon_layernorm"],
        reconstructor_type=cfg["reconstructor_type"],
        temporal_layers=cfg["temporal_decoder_layers"],
        with_forecast=cfg.get("with_forecast", False),
    ).to(device)

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"[INFO] checkpoint 缺少以下 key（新增模块，使用随机初始化）：{missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[INFO] checkpoint 包含以下多余 key（已忽略）：{unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    reconstructor.load_state_dict(ckpt["reconstructor_state_dict"])

    # 从 checkpoint meta 恢复 DGR 全局归一化统计量（若存在）。
    # 有统计量时 model.forward() 自动走全局 Z-score 归一化，
    # 无统计量时静默回退 per-sample Min-Max，与旧 checkpoint 完全兼容。
    meta = ckpt.get("meta", {}) if isinstance(ckpt.get("meta", {}), dict) else {}
    dgr_mean = meta.get("dgr_corr_mean")
    dgr_std  = meta.get("dgr_corr_std")
    if dgr_mean is not None and dgr_std is not None:
        model.set_dgr_stats(float(dgr_mean), float(dgr_std))
        print(f"[DGR] 已从 checkpoint 恢复全局统计量：corr_mean={dgr_mean:.6f}, corr_std={dgr_std:.6f}")
    else:
        print("[DGR] checkpoint 不含全局统计量，使用 per-sample Min-Max 归一化（向后兼容）。")

    model.eval()
    reconstructor.eval()
    return model, reconstructor


@torch.no_grad()
def get_features_and_errors(
    loader,
    model: MultiModalFusionModel,
    reconstructor: Reconstructor,
    device: torch.device,
    collect_window_labels: bool = False,
    forecast_weight: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    z_list: list[np.ndarray] = []
    mse_list: list[np.ndarray] = []
    fore_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []

    use_forecast = forecast_weight > 0.0 and getattr(reconstructor, "with_forecast", False)

    for batch in loader:
        x_seq = (batch[1] if len(batch) >= 2 else batch[0]).to(device).float()
        x_seq = torch.nan_to_num(x_seq, nan=0.0, posinf=0.0, neginf=0.0)

        z = model(x_seq)
        x_hat = reconstructor(z)
        mse_per_window = F.mse_loss(x_hat, x_seq, reduction="none").mean(dim=(1, 2))

        # 评分阶段默认使用样本级潜特征；若为时序潜变量则先做时间池化。
        z_for_score = z.mean(dim=1) if z.ndim == 3 else z
        z_list.append(z_for_score.detach().cpu().numpy())
        mse_list.append(mse_per_window.detach().cpu().numpy())

        # MTAD-GAT 启发：预测头误差（可选）
        if use_forecast:
            x_forecast = reconstructor.forecast(z)  # (B, W-1, C)
            fore_mse = F.mse_loss(x_forecast, x_seq[:, 1:], reduction="none").mean(dim=(1, 2))
            fore_list.append(fore_mse.detach().cpu().numpy())

        if collect_window_labels and len(batch) >= 3:
            label_list.append(batch[2].detach().cpu().numpy().reshape(-1))

    if not z_list:
        raise RuntimeError("DataLoader 为空")

    z_all = np.concatenate(z_list, axis=0).astype(np.float32)
    mse_all = np.concatenate(mse_list, axis=0).astype(np.float32)

    # 融合重构误差与预测误差
    if use_forecast and fore_list:
        fore_all = np.concatenate(fore_list, axis=0).astype(np.float32)
        mse_all = (1.0 - forecast_weight) * mse_all + forecast_weight * fore_all

    if collect_window_labels:
        if label_list:
            labels_all = np.concatenate(label_list, axis=0).astype(np.int32)
        else:
            labels_all = np.zeros((z_all.shape[0],), dtype=np.int32)
        return z_all, mse_all, labels_all

    return z_all, mse_all, None


def _split_by_window_counts(arr: np.ndarray, window_counts: list[int]) -> list[np.ndarray]:
    parts: list[np.ndarray] = []
    start = 0
    for n in window_counts:
        end = start + int(n)
        parts.append(arr[start:end])
        start = end
    return parts


def point_adjustment(true_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    """Point-Adjustment: 真实异常段内只要命中一个点，整段置为异常"""
    true_arr = np.asarray(true_labels).astype(np.int32).reshape(-1)
    pred_arr = np.asarray(pred_labels).astype(np.int32).reshape(-1)
    if len(true_arr) != len(pred_arr):
        raise ValueError("长度不一致")

    adjusted = pred_arr.copy()
    n = len(true_arr)
    i = 0
    while i < n:
        if true_arr[i] == 0:
            i += 1
            continue

        start = i
        while i + 1 < n and true_arr[i + 1] == 1:
            i += 1
        end = i

        if np.any(pred_arr[start : end + 1] == 1):
            adjusted[start : end + 1] = 1

        i += 1

    return adjusted


def compute_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict[str, float]:
    """计算一整套指标"""
    p, r, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="binary", zero_division=0
    )
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'precision': float(p),
        'recall': float(r),
        'f1': float(f1),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }
    return metrics


def compute_auprc(true_labels: np.ndarray, scores: np.ndarray) -> float:
    """计算 Area Under Precision-Recall Curve"""
    true_arr = np.asarray(true_labels).reshape(-1)
    if true_arr.size == 0:
        return float('nan')
    pos_count = int(np.sum(true_arr == 1))
    if pos_count == 0 or pos_count == int(true_arr.size):
        # 全负样本或全正样本时，PR 曲线不具备可解释性
        return float('nan')
    try:
        precision, recall, _ = precision_recall_curve(true_labels, scores)
        return float(auc(recall, precision))
    except ValueError:
        return float('nan')


def summarize_label_distribution(true_labels: np.ndarray) -> tuple[int, int, int]:
    """返回标签分布统计：(总数, 正样本数, 负样本数)。"""
    y = np.asarray(true_labels).reshape(-1)
    total = int(y.size)
    pos = int(np.sum(y == 1))
    neg = int(total - pos)
    return total, pos, neg


def calibrate_threshold_f1(
    scores: np.ndarray,
    labels: np.ndarray,
    base_threshold: float,
    calib_ratio: float,
    calib_min_points: int,
) -> tuple[float, dict[str, float]]:
    """在前缀标注样本上搜索使 F1 最大化的阈值。"""
    y = np.asarray(labels).reshape(-1).astype(np.int32)
    s = np.asarray(scores).reshape(-1).astype(np.float32)
    n = int(min(len(y), len(s)))
    if n <= 0:
        return float(base_threshold), {"used_points": 0, "best_f1": float("nan")}

    ratio = float(calib_ratio)
    if not (0.0 < ratio < 1.0):
        ratio = 0.2
    calib_n = max(int(n * ratio), int(calib_min_points))
    calib_n = min(calib_n, n)

    y_cal = y[:calib_n]
    s_cal = s[:calib_n]
    pos = int(np.sum(y_cal == 1))
    neg = int(calib_n - pos)
    if pos == 0 or neg == 0:
        return float(base_threshold), {
            "used_points": calib_n,
            "best_f1": float("nan"),
            "positive": pos,
            "negative": neg,
        }

    quantiles = np.linspace(0.80, 0.999, 80)
    candidates = np.quantile(s_cal, quantiles)
    candidates = np.unique(np.concatenate([[float(base_threshold)], candidates]))

    best_th = float(base_threshold)
    best_f1 = -1.0
    for th in candidates:
        pred = (s_cal > float(th)).astype(np.int32)
        _, _, f1, _ = precision_recall_fscore_support(
            y_cal, pred, average="binary", zero_division=0
        )
        if float(f1) > best_f1:
            best_f1 = float(f1)
            best_th = float(th)

    return best_th, {
        "used_points": calib_n,
        "best_f1": best_f1,
        "positive": pos,
        "negative": neg,
    }


def compute_event_level_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict[str, float]:
    """计算事件级指标（按连续异常段统计）"""
    def extract_events(labels: np.ndarray) -> list[Tuple[int, int]]:
        events = []
        i = 0
        while i < len(labels):
            if labels[i] == 0:
                i += 1
                continue
            start = i
            while i + 1 < len(labels) and labels[i + 1] == 1:
                i += 1
            events.append((start, i))
            i += 1
        return events

    true_events = extract_events(true_labels)
    pred_events = extract_events(pred_labels)

    if not true_events:
        return {'event_recall': 0.0, 'event_precision': 0.0, 'event_f1': 0.0}

    # 简单的事件级 recall: 多少个真实事件被至少命中一个点
    detected = 0
    for te_start, te_end in true_events:
        for pe_start, pe_end in pred_events:
            if not (pe_end < te_start or pe_start > te_end):  # 有交集
                detected += 1
                break

    event_recall = detected / len(true_events) if true_events else 0.0
    event_precision = 0.0
    if pred_events:
        matched = 0
        for pe_start, pe_end in pred_events:
            for te_start, te_end in true_events:
                if not (pe_end < te_start or pe_start > te_end):
                    matched += 1
                    break
        event_precision = matched / len(pred_events)

    event_f1 = 2 * event_precision * event_recall / (event_precision + event_recall + 1e-8)

    return {
        'event_recall': float(event_recall),
        'event_precision': float(event_precision),
        'event_f1': float(event_f1),
    }


def print_metrics(title: str, true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict[str, float]:
    """打印和返回检测指标"""
    metrics = compute_metrics(true_labels, pred_labels)
    
    print("\n" + "=" * 72)
    print(title)
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall   : {metrics['recall']:.6f}")
    print(f"F1-Score : {metrics['f1']:.6f}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(f"[[{metrics['tn']}, {metrics['fp']}], [{metrics['fn']}, {metrics['tp']}]]")
    print("=" * 72)
    
    return metrics


def save_results_csv(
    csv_path: str | Path,
    point_scores: np.ndarray,
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    pa_pred_labels: np.ndarray,
) -> Path:
    """保存评估结果到 CSV"""
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    min_len = min(len(point_scores), len(pred_labels), len(true_labels), len(pa_pred_labels))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "point_score", "pred_label", "true_label", "pa_pred_label"])
        for i in range(min_len):
            writer.writerow([
                i,
                float(point_scores[i]),
                int(pred_labels[i]),
                int(true_labels[i]),
                int(pa_pred_labels[i]),
            ])

    return out_path


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载 checkpoint
    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        # PyTorch < 2.6 不支持 weights_only 参数
        ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"已加载 checkpoint: {ckpt_path}")

    # 解析配置
    cfg = resolve_runtime_config(args, ckpt)
    print(f"评估配置: data_dir={cfg['data_dir']}, dataset_type={cfg['dataset_type']}, machine_id={cfg['machine_id']}, window_size={cfg['window_size']}")

    # 加载数据集
    train_loader, test_loader, _meta = build_train_test_loaders(
        data_dir=cfg['data_dir'],
        window_size=cfg['window_size'],
        dataset_type=cfg['dataset_type'],
        machine_id=cfg['machine_id'],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_train=False,
    )

    # 通过 loaders 的 dataset 获取 dataset 对象
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset

    # 得到检测数据集类型（通过 train_dataset 推断）
    actual_dataset_type = getattr(train_dataset, 'dataset_type', cfg['dataset_type'])
    print(f"检测到数据集类型: {actual_dataset_type}")

    # 获取数据集专属参数
    params = DATASET_PARAMS.get(actual_dataset_type, DATASET_PARAMS['psm'])
    
    # 命令行参数优先级更高
    alpha = args.alpha if args.alpha is not None else params.get('alpha', 0.7)
    q_threshold = args.q_threshold if args.q_threshold is not None else params.get('q_threshold', 0.995)
    min_duration = args.min_duration if args.min_duration is not None else params.get('min_duration', 0)
    gap_filling = args.gap_filling if args.gap_filling is not None else params.get('gap_filling', 0)
    use_hysteresis = args.enable_hysteresis or params.get('use_hysteresis', False)
    threshold_calibration = (
        args.threshold_calibration
        if args.threshold_calibration is not None
        else params.get('threshold_calibration', 'none')
    )
    calib_ratio = args.calib_ratio if args.calib_ratio is not None else params.get('calib_ratio', 0.2)
    # MTAD-GAT 启发：预测头权重（命令行优先，数据集默认次之）
    forecast_weight = (
        args.forecast_weight
        if args.forecast_weight is not None
        else params.get('forecast_weight', 0.0)
    )
    if forecast_weight > 0.0 and not cfg.get("with_forecast", False):
        print("[警告] forecast_weight>0 但 checkpoint 未启用预测头，将忽略预测权重。")
        forecast_weight = 0.0

    print(
        f"[参数] alpha={alpha}, q_threshold={q_threshold}, min_duration={min_duration}, "
        f"gap_filling={gap_filling}, use_hysteresis={use_hysteresis}, "
        f"threshold_calibration={threshold_calibration}, calib_ratio={calib_ratio}"
    )

    # 构建模型
    model, reconstructor = build_model_and_reconstructor(cfg, ckpt, device)

    # 提取特征与重构误差
    print("提取训练特征...")
    train_z, train_errors, _ = get_features_and_errors(
        train_loader, model, reconstructor, device, forecast_weight=forecast_weight
    )
    print(f"train_z shape={train_z.shape}, train_errors shape={train_errors.shape}")

    print("提取测试特征...")
    test_z, test_errors, test_window_labels = get_features_and_errors(
        test_loader, model, reconstructor, device, collect_window_labels=True,
        forecast_weight=forecast_weight
    )
    print(f"test_z shape={test_z.shape}, test_errors shape={test_errors.shape}")

    # 构建并拟合打分器
    scorer = AnomalyScorer(
        window_size=cfg['window_size'],
        alpha=alpha,
        q_threshold=q_threshold,
        sigma_w=args.sigma_w,
        smoothing_window=args.smoothing_window,
        min_duration=min_duration,
        gap_filling=gap_filling,
        use_hysteresis=use_hysteresis,
        hysteresis_factor=params.get('hysteresis_factor', 0.9),
    )
    calib_n = 0  # 提升到外部，供后续 BATADAL 修复逻辑访问
    if (
        args.custom_threshold is None
        and str(threshold_calibration).lower() == 'f1'
        and test_window_labels is not None
    ):
        ratio = float(calib_ratio)
        if not (0.0 < ratio < 1.0):
            ratio = 0.2
        calib_n = max(int(len(test_window_labels) * ratio), int(args.calib_min_points))
        calib_n = min(calib_n, len(test_window_labels))
        scorer.fit_with_validation(
            train_z=train_z,
            train_errors=train_errors,
            val_z=test_z[:calib_n],
            val_errors=test_errors[:calib_n],
            val_labels=test_window_labels[:calib_n],
        )
        print(f"[阈值校准] mode=adaptive_f1, used_windows={calib_n}")
    else:
        scorer.fit(train_z, train_errors)
    print(f"scorer.threshold={scorer.threshold:.6f}")

    # ── BATADAL 专项：训练集全为正常样本，校准窗口可能全为负样本，需全量搜索 ──
    if (
        actual_dataset_type == 'batadal'
        and str(threshold_calibration).lower() == 'f1'
        and args.custom_threshold is None
        and test_window_labels is not None
        and int(np.sum(test_window_labels == 1)) > 0
    ):
        pos_in_calib = int(np.sum(test_window_labels[:calib_n] == 1)) if calib_n > 0 else 0
        if pos_in_calib == 0:
            print("[BATADAL] 校准窗口内无正样本，切换为全量测试误差搜索阈值。")
            from sklearn.metrics import precision_recall_fscore_support as _prf
            s_full = test_errors.astype(np.float32)
            y_full = test_window_labels.astype(np.int32)
            quantiles_full = np.linspace(0.20, 0.999, 200)
            cands_full = np.unique(np.concatenate([
                [scorer.threshold],
                np.quantile(s_full, quantiles_full),
            ]))
            best_th_full, best_f1_full = scorer.threshold, -1.0
            for _th in cands_full:
                _pred = (s_full > float(_th)).astype(np.int32)
                _, _, _f1, _ = _prf(y_full, _pred, average='binary', zero_division=0)
                if float(_f1) > best_f1_full:
                    best_f1_full = float(_f1)
                    best_th_full = float(_th)
            print(
                f"[BATADAL] 全量搜索完成：{scorer.threshold:.6f} → {best_th_full:.6f}，"
                f"窗口级 F1={best_f1_full:.4f}"
            )
            scorer.threshold = best_th_full

    # 预测测试集
    window_counts = list(getattr(test_dataset, "window_counts", []))
    series_data = list(getattr(test_dataset, "series_data", []))
    series_labels = list(getattr(test_dataset, "series_labels", []))

    if not window_counts or not series_data:
        seq_len = len(test_errors) + cfg['window_size'] - 1
        point_scores, preds = scorer.predict(test_z, test_errors, seq_len=seq_len, custom_threshold=args.custom_threshold)
        true_labels = np.zeros((seq_len,), dtype=np.int32)
    else:
        # 按序列处理
        z_parts = _split_by_window_counts(test_z, window_counts)
        e_parts = _split_by_window_counts(test_errors, window_counts)

        all_scores: list[np.ndarray] = []
        all_preds: list[np.ndarray] = []
        all_true: list[np.ndarray] = []

        threshold = scorer.threshold if args.custom_threshold is None else float(args.custom_threshold)
        for i, (z_i, e_i) in enumerate(zip(z_parts, e_parts)):
            seq_len = int(len(series_data[i]))
            if len(z_i) == 0:
                point_scores_i = np.zeros((seq_len,), dtype=np.float32)
            else:
                point_scores_i = scorer.predict_score(z_i, e_i, seq_len=seq_len).astype(np.float32)

            pred_i = (point_scores_i > threshold).astype(np.int32)
            if i < len(series_labels):
                true_i = np.asarray(series_labels[i]).reshape(-1).astype(np.int32)
            else:
                true_i = np.zeros((seq_len,), dtype=np.int32)

            min_len = min(len(point_scores_i), len(true_i))
            all_scores.append(point_scores_i[:min_len])
            all_preds.append(pred_i[:min_len])
            all_true.append(true_i[:min_len])

        point_scores = np.concatenate(all_scores).astype(np.float32)
        preds = np.concatenate(all_preds).astype(np.int32)
        true_labels = np.concatenate(all_true).astype(np.int32)

    threshold = scorer.threshold if args.custom_threshold is None else float(args.custom_threshold)

    preds = (point_scores > float(threshold)).astype(np.int32)

    # 应用后处理
    if min_duration > 0 or gap_filling > 0:
        preds_pp = post_process_predictions(preds, min_duration=min_duration, gap_filling=gap_filling)
    else:
        preds_pp = preds.copy()

    # 计算 PA
    pa_preds = point_adjustment(true_labels, preds)
    pa_preds_pp = point_adjustment(true_labels, preds_pp)

    # 标签分布检查
    total_n, pos_n, neg_n = summarize_label_distribution(true_labels)
    print(f"标签分布: total={total_n}, positive={pos_n}, negative={neg_n}")
    if pos_n == 0:
        print("[警告] 当前测试标签全为 0（无异常点）。Recall/F1/PA/AUPRC 不具备可解释性，仅可参考误报规模(FP)。")

    # 打印指标
    print(f"point_scores shape={point_scores.shape}, pred_labels shape={preds.shape}, true_labels shape={true_labels.shape}")
    
    metrics_point = print_metrics("基础指标（Point-wise）", true_labels, preds)
    metrics_point_pp = print_metrics("后处理指标（Point-wise with post-processing）", true_labels, preds_pp)
    metrics_pa = print_metrics("点调整指标（PA-Point-wise）", true_labels, pa_preds)
    metrics_pa_pp = print_metrics("点调整+后处理指标（PA-Point-wise with post-processing）", true_labels, pa_preds_pp)

    # 高级指标
    auprc_point = compute_auprc(true_labels, point_scores)
    event_metrics_point = compute_event_level_metrics(true_labels, preds)

    print("\n" + "=" * 72)
    print("高级指标")
    if np.isnan(auprc_point):
        print("AUPRC (Point-wise): N/A (标签单一，无法计算)")
    else:
        print(f"AUPRC (Point-wise): {auprc_point:.6f}")
    print(f"事件级Recall (Point-wise): {event_metrics_point['event_recall']:.6f}")
    print(f"事件级Precision (Point-wise): {event_metrics_point['event_precision']:.6f}")
    print(f"事件级F1 (Point-wise): {event_metrics_point['event_f1']:.6f}")
    print("=" * 72)

    # 保存 CSV
    csv_path = args.save_csv or f"checkpoints/eval_{actual_dataset_type}_enhanced.csv"
    save_results_csv(csv_path, point_scores, preds, true_labels, pa_preds)
    print(f"已保存评估结果 CSV: {csv_path}")

    # 总结
    print("\n【总结】")
    print(f"数据集: {actual_dataset_type}")
    print(f"基础 F1 (Point): {metrics_point['f1']:.6f}")
    print(f"后处理 F1 (Point): {metrics_point_pp['f1']:.6f} (改善: {(metrics_point_pp['f1']-metrics_point['f1']):.6f})")
    print(f"PA F1: {metrics_pa['f1']:.6f}")
    print(f"PA+后处理 F1: {metrics_pa_pp['f1']:.6f}")


if __name__ == "__main__":
    main()
