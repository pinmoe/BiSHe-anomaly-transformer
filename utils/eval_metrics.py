"""utils/eval_metrics.py
工业时序异常检测评估指标集。

支持的指标：
  - 逐点 Precision / Recall / F1
  - PA-F1（Point-Adjust，与 TranAD 等文献对齐，需声明 Recall 高估）
  - AUPRC（Area Under Precision-Recall Curve，不依赖阈值，最客观）
  - 事件级指标（Event-level Precision / Recall / F1）
  - 检测延迟（Detection Latency，单位：时间步）
"""
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    precision_recall_curve, auc
)


# --------------------------------------------------------------------------- #
# 基础逐点指标
# --------------------------------------------------------------------------- #

def pointwise_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """
    计算逐点 Precision / Recall / F1 / Accuracy。

    参数
    ----
    gt   : 0/1 标签数组，形状 (N,)
    pred : 0/1 预测数组，形状 (N,)

    返回
    ----
    dict，键为 'accuracy' 'precision' 'recall' 'f1'
    """
    return {
        'accuracy':  accuracy_score(gt, pred),
        'precision': precision_score(gt, pred, zero_division=0),
        'recall':    recall_score(gt, pred, zero_division=0),
        'f1':        f1_score(gt, pred, zero_division=0),
    }


# --------------------------------------------------------------------------- #
# PA-F1（Point-Adjust 协议）
# --------------------------------------------------------------------------- #

def point_adjust(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Point-Adjust 调整：若预测窗口命中某连续异常段中任意一点，
    则该段内所有点预测置为 1。

    注意：PA 协议会人为提高 Recall，需在论文中明确声明。
    """
    pred_pa = pred.copy()
    in_anomaly = False
    segment_start = 0

    for i in range(len(gt)):
        if gt[i] == 1 and not in_anomaly:
            in_anomaly = True
            segment_start = i
        if gt[i] == 0 and in_anomaly:
            # 检查该段内是否有任意预测为 1
            if pred[segment_start:i].any():
                pred_pa[segment_start:i] = 1
            in_anomaly = False
    # 处理末尾未闭合的异常段
    if in_anomaly:
        if pred[segment_start:].any():
            pred_pa[segment_start:] = 1

    return pred_pa


def pa_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """
    计算 Point-Adjust 协议下的 Precision / Recall / F1。
    """
    pred_pa = point_adjust(gt, pred)
    return {
        'pa_precision': precision_score(gt, pred_pa, zero_division=0),
        'pa_recall':    recall_score(gt, pred_pa, zero_division=0),
        'pa_f1':        f1_score(gt, pred_pa, zero_division=0),
    }


# --------------------------------------------------------------------------- #
# AUPRC（不依赖阈值）
# --------------------------------------------------------------------------- #

def compute_auprc(gt: np.ndarray, score: np.ndarray) -> float:
    """
    计算 Precision-Recall 曲线下面积（AUPRC）。

    参数
    ----
    gt    : 0/1 标签数组，形状 (N,)
    score : 连续异常分数（越大越异常），形状 (N,)

    返回
    ----
    float，AUPRC 值
    """
    precision, recall, _ = precision_recall_curve(gt, score)
    return auc(recall, precision)


# --------------------------------------------------------------------------- #
# 事件级指标
# --------------------------------------------------------------------------- #

def _get_events(labels: np.ndarray) -> list:
    """从 0/1 标签数组中提取连续异常段列表，每段为 (start, end) 闭区间。"""
    events = []
    in_event = False
    start = 0
    for i, v in enumerate(labels):
        if v == 1 and not in_event:
            in_event = True
            start = i
        elif v == 0 and in_event:
            events.append((start, i - 1))
            in_event = False
    if in_event:
        events.append((start, len(labels) - 1))
    return events


def event_level_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """
    事件级 Precision / Recall / F1。

    - Event Recall：真实异常事件中被检测到的比例（事件内有任意预测点即算检测到）。
    - Event Precision：预测为异常的点中，属于真实异常事件的比例（事件粒度）。
    - Event F1：调和平均。

    参数
    ----
    gt   : 0/1 真实标签，形状 (N,)
    pred : 0/1 预测标签，形状 (N,)
    """
    gt_events = _get_events(gt)
    if len(gt_events) == 0:
        return {'event_precision': 0.0, 'event_recall': 0.0, 'event_f1': 0.0}

    # Event Recall：被命中的真实事件数 / 总真实事件数
    detected = 0
    for (s, e) in gt_events:
        if pred[s:e + 1].any():
            detected += 1
    event_recall = detected / len(gt_events)

    # Event Precision：预测到的异常段中命中真实事件的比例
    pred_events = _get_events(pred)
    if len(pred_events) == 0:
        event_precision = 0.0
    else:
        tp_pred = 0
        gt_set = set()
        for (s, e) in gt_events:
            gt_set.update(range(s, e + 1))
        for (s, e) in pred_events:
            if any(i in gt_set for i in range(s, e + 1)):
                tp_pred += 1
        event_precision = tp_pred / len(pred_events)

    denom = event_precision + event_recall
    event_f1 = (2 * event_precision * event_recall / denom) if denom > 0 else 0.0

    return {
        'event_precision': event_precision,
        'event_recall':    event_recall,
        'event_f1':        event_f1,
    }


# --------------------------------------------------------------------------- #
# 检测延迟
# --------------------------------------------------------------------------- #

def detection_latency(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    计算每个异常事件的检测延迟（时间步数），返回均值。

    延迟 = 首次预测为异常的时间步 - 异常段开始时间步。
    若整个异常段均未被检测到，该事件延迟记为段长（最坏情况）。

    参数
    ----
    gt   : 0/1 真实标签，形状 (N,)
    pred : 0/1 预测标签，形状 (N,)

    返回
    ----
    float，平均检测延迟（时间步）
    """
    events = _get_events(gt)
    if len(events) == 0:
        return 0.0

    latencies = []
    for (s, e) in events:
        hit_indices = np.where(pred[s:e + 1] == 1)[0]
        if len(hit_indices) > 0:
            latencies.append(int(hit_indices[0]))  # 相对于段起点的偏移
        else:
            latencies.append(e - s + 1)  # 未检测到，记最大延迟

    return float(np.mean(latencies))


# --------------------------------------------------------------------------- #
# 一键汇总所有指标
# --------------------------------------------------------------------------- #

def full_evaluation(gt: np.ndarray, pred: np.ndarray, score: np.ndarray = None) -> dict:
    """
    一次性计算所有评估指标。

    参数
    ----
    gt    : 0/1 真实标签
    pred  : 0/1 预测标签
    score : 连续异常分数（可选，用于计算 AUPRC）

    返回
    ----
    dict，包含所有指标键值对
    """
    results = {}
    results.update(pointwise_metrics(gt, pred))
    results.update(pa_metrics(gt, pred))
    results.update(event_level_metrics(gt, pred))
    results['detection_latency'] = detection_latency(gt, pred)
    if score is not None:
        results['auprc'] = compute_auprc(gt, score)
    return results
