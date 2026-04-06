import numpy as np
import torch


def _to_tensor(x):
    """将 numpy 或 torch 输入统一转换为 float32 的 torch.Tensor"""
    if isinstance(x, torch.Tensor):
        return x.float()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    raise TypeError('输入必须是 torch.Tensor 或 numpy.ndarray。')


def window_to_corr_image(window, eps=1e-6):
    """
    将单个滑窗转换为相关性图。

    参数:
    - window: 形状 (W, C)，W 是时间步，C 是传感器通道数
    - eps: 防止除零的极小值

    返回:
    - corr: 形状 (C, C) 的相关性矩阵，数值范围约为 [-1, 1]
    """
    x = _to_tensor(window)
    if x.ndim != 2:
        raise ValueError(f'window 维度必须是 2D (W, C)，当前是 {tuple(x.shape)}。')

    w = x.shape[0]
    if w < 2:
        raise ValueError('window 的时间长度 W 必须 >= 2，才能计算相关系数。')

    # 沿时间维做中心化，得到每个通道的波动。
    x_centered = x - x.mean(dim=0, keepdim=True)

    # 协方差: (C, W) @ (W, C) -> (C, C)
    cov = (x_centered.transpose(0, 1) @ x_centered) / max(w - 1, 1)

    # 标准差向量，后续用于把协方差标准化为相关系数。
    var = torch.diag(cov)
    std = torch.sqrt(torch.clamp(var, min=0.0) + eps)
    denom = torch.outer(std, std) + eps

    corr = cov / denom
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = torch.clamp(corr, -1.0, 1.0)
    return corr


def normalize_image(image, eps=1e-6):
    """
    对相关性图做 Min-Max 归一化，输出范围 [0, 1]。

    支持输入形状:
    - (C, C)
    - (B, C, C)
    """
    x = _to_tensor(image)
    if x.ndim not in (2, 3):
        raise ValueError(f'image 只支持 2D 或 3D，当前是 {tuple(x.shape)}。')

    if x.ndim == 2:
        x_min = x.min()
        x_max = x.max()
        return (x - x_min) / (x_max - x_min + eps)

    # 3D 时按样本分别归一化，避免 batch 内互相影响。
    x_min = x.amin(dim=(1, 2), keepdim=True)
    x_max = x.amax(dim=(1, 2), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def build_dgr_batch(x_2d_batch, normalize=True, eps=1e-6,
                    corr_mean=None, corr_std=None):
    """
    将一批时序窗口转换为 DGR 图像 batch。

    参数:
    - x_2d_batch: 形状 (B, W, C)，来自数据集里的 x_2d
    - normalize: 是否将每张图归一化到 [0, 1]（按样本 Min-Max，向后兼容模式）
    - corr_mean: 全局相关矩阵均值（由 compute_dgr_stats 在训练后计算）
    - corr_std:  全局相关矩阵标准差（由 compute_dgr_stats 在训练后计算）
      当二者均不为 None 时，改用全局 Z-score 归一化后映射到 [0,1]，
      能保留正常/异常相关矩阵之间的全局幅度差异。

    返回:
    - dgr_batch: 形状 (B, 1, C, C)，可直接喂给 ViT 图像分支
    """
    x = _to_tensor(x_2d_batch)
    if x.ndim != 3:
        raise ValueError(f'x_2d_batch 维度必须是 3D (B, W, C)，当前是 {tuple(x.shape)}。')

    _, w, _ = x.shape
    if w < 2:
        raise ValueError('窗口长度 W 必须 >= 2，才能计算相关系数。')

    # 批量中心化与协方差计算。
    x_centered = x - x.mean(dim=1, keepdim=True)  # (B, W, C)
    cov = (x_centered.transpose(1, 2) @ x_centered) / max(w - 1, 1)  # (B, C, C)

    var = torch.diagonal(cov, dim1=1, dim2=2)  # (B, C)
    std = torch.sqrt(torch.clamp(var, min=0.0) + eps)
    denom = std.unsqueeze(2) * std.unsqueeze(1) + eps  # (B, C, C)

    corr = cov / denom
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = torch.clamp(corr, -1.0, 1.0)

    if corr_mean is not None and corr_std is not None:
        # 全局 Z-score 归一化，保留正常/异常相关图的全局幅度信号。
        corr_mean_t = torch.tensor(corr_mean, dtype=corr.dtype, device=corr.device)
        corr_std_t = torch.tensor(corr_std, dtype=corr.dtype, device=corr.device)
        corr = (corr - corr_mean_t) / (corr_std_t + 1e-6)
        corr = torch.clamp(corr, -3.0, 3.0)
        corr = (corr + 3.0) / 6.0            # 映射到 [0, 1]
    elif normalize:
        corr = normalize_image(corr, eps=eps)  # 旧行为，按样本 Min-Max，向后兼容

    # 增加通道维，得到图像格式 (B, 1, C, C)
    return corr.unsqueeze(1)


def compute_dgr_stats(x_batch: torch.Tensor) -> tuple[float, float]:
    """计算相关矩阵的全局均值与标准差，供训练脚本累计后保存到 checkpoint。

    参数:
    - x_batch: 形状 (B, W, C)

    返回:
    - (全局均值, 全局标准差)，可累计后传入 build_dgr_batch 的 corr_mean/corr_std。
    """
    corr = build_dgr_batch(x_batch, normalize=False).squeeze(1)  # (B, C, C)
    return float(corr.mean()), float(corr.std())


if __name__ == '__main__':
    # 冒烟测试：验证单窗口和 batch 转换的输出形状。
    single_window = torch.randn(100, 25)
    single_img = window_to_corr_image(single_window)
    single_img_norm = normalize_image(single_img)

    batch_windows = torch.randn(8, 100, 25)
    batch_imgs = build_dgr_batch(batch_windows, normalize=True)

    print('single_img shape:', tuple(single_img.shape))
    print('single_img_norm range:', float(single_img_norm.min()), float(single_img_norm.max()))
    print('batch_imgs shape:', tuple(batch_imgs.shape))
