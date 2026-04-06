from collections.abc import Callable

import torch
import torch.nn.functional as F


def mse_loss(x_hat: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
    """重构误差项：L_mse = mean((x_hat - x)^2)。"""
    if x_hat.shape != x_target.shape:
        raise ValueError(
            f"x_hat 与 x_target 形状必须一致，当前分别为 {tuple(x_hat.shape)} 与 {tuple(x_target.shape)}"
        )
    return F.mse_loss(x_hat, x_target, reduction="mean")


def idempotent_loss(
    z: torch.Tensor,
    x_hat: torch.Tensor,
    re_encoder: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """等幂约束项：L_idem = mean((z - Enc(x_hat))^2)。"""
    if z.ndim not in (2, 3):
        raise ValueError(f"z 需要是二维或三维张量，当前形状为 {tuple(z.shape)}")

    z_re = re_encoder(x_hat)
    if z_re.shape != z.shape:
        raise ValueError(
            f"re_encoder(x_hat) 输出形状必须与 z 一致，当前分别为 {tuple(z_re.shape)} 与 {tuple(z.shape)}"
        )
    return F.mse_loss(z_re, z, reduction="mean")


def joint_loss(
    x_hat: torch.Tensor,
    x_target: torch.Tensor,
    z: torch.Tensor,
    re_encoder: Callable[[torch.Tensor], torch.Tensor],
    lambda_idem: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """联合损失：L = L_mse + lambda * L_idem。"""
    if not torch.isfinite(torch.tensor(lambda_idem)):
        raise ValueError(f"lambda_idem 必须是有限数值，当前为 {lambda_idem}")
    if lambda_idem < 0.0:
        raise ValueError(f"lambda_idem 不能为负数，当前为 {lambda_idem}")

    loss_mse = mse_loss(x_hat, x_target)
    loss_idem = idempotent_loss(z, x_hat, re_encoder)
    total = loss_mse + float(lambda_idem) * loss_idem
    return total, loss_mse, loss_idem


# ─────────────────────────────────────────────────────────────────────────────
#  新增：判别性联合优化损失（重构 + 对比 + 分离 + 时序约束）
# ─────────────────────────────────────────────────────────────────────────────

def pseudo_label_contrastive_loss(
    z: torch.Tensor,
    scores: torch.Tensor,
    margin: float = 1.0,
    pos_ratio: float = 0.1,
    neg_ratio: float = 0.5,
    temperature: float = 0.2,
) -> torch.Tensor:
    """伪标签对比损失（SupCon 风格）。

    利用当前重构误差分布自动挖掘伪标签：高误差样本作为伪异常（正类），
    低误差样本作为伪正常（负类）。同类互为正例，异类互为负例。

    Args:
        z: (B, D) 融合潜向量
        scores: (B,) 当前重构误差分数（无需梯度）
        margin: 正负样本间距阈值（保留参数，InfoNCE 模式未直接使用）
        pos_ratio: 高分段作为伪异常的比例
        neg_ratio: 低分段作为伪正常的比例
        temperature: InfoNCE 温度系数
    Returns:
        标量损失值
    """
    if z.ndim == 3:
        # 时序潜变量先做时间池化，用于样本级判别损失。
        z = z.mean(dim=1)
    if z.ndim != 2:
        raise ValueError(f"z 需要是 (B,D) 或 (B,W,D)，当前形状为 {tuple(z.shape)}")

    B = z.shape[0]
    n_pos = max(1, int(B * pos_ratio))
    n_neg = max(1, int(B * neg_ratio))

    # 按分数升序排列：低分→伪正常，高分→伪异常
    sorted_idx = torch.argsort(scores, descending=False)
    neg_idx = sorted_idx[:n_neg]
    pos_idx = sorted_idx[-n_pos:]

    # L2 归一化嵌入
    z_pos = F.normalize(z[pos_idx], dim=1)   # (n_pos, D)
    z_neg = F.normalize(z[neg_idx], dim=1)   # (n_neg, D)

    # 拼接: [z_pos; z_neg]
    z_all = torch.cat([z_pos, z_neg], dim=0)
    n_total = n_pos + n_neg

    # 相似度矩阵，屏蔽对角自相似度（使用大负数而非 -inf，避免 -inf*0=NaN）
    sim = torch.mm(z_all, z_all.T) / temperature
    diag_mask = torch.eye(n_total, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(diag_mask, -1e9)   # -1e9 保证 exp≈0，同时避免 inf*0=NaN

    # 同类正例掩码：pos-pos 和 neg-neg 互为正例
    pos_pair_mask = torch.zeros(n_total, n_total, dtype=torch.bool, device=z.device)
    if n_pos >= 2:
        pos_pair_mask[:n_pos, :n_pos] = True
    if n_neg >= 2:
        pos_pair_mask[n_pos:, n_pos:] = True
    pos_pair_mask.fill_diagonal_(False)

    n_pos_pairs = pos_pair_mask.float().sum(dim=1)
    has_pos_pairs = n_pos_pairs > 0
    if not has_pos_pairs.any():
        # 无有效正例对时返回零（维持计算图）
        return z.sum() * 0.0

    # InfoNCE：-inf 对角在 exp 后为 0，不参与分母归一化
    log_prob = F.log_softmax(sim, dim=1)   # (n_total, n_total)

    # 每行平均正例对数概率
    loss_per_row = -(log_prob * pos_pair_mask.float()).sum(dim=1) / n_pos_pairs.clamp(min=1)
    return loss_per_row[has_pos_pairs].mean()


def separation_loss(
    z: torch.Tensor,
    scores: torch.Tensor,
    margin: float = 2.0,
    num_bins: int = 5,
) -> torch.Tensor:
    """分离约束损失。

    显式约束高分样本和低分样本在潜空间的类间距离：
        L = max(0, margin - ||mu_pos - mu_neg||) + 0.1 * (var_pos + var_neg)

    Args:
        z: (B, D) 潜向量
        scores: (B,) 异常分数
        margin: 强制质心分离的 margin 阈值
        num_bins: 每组样本数约为 B // num_bins
    Returns:
        标量损失值
    """
    if z.ndim == 3:
        z = z.mean(dim=1)
    if z.ndim != 2:
        raise ValueError(f"z 需要是 (B,D) 或 (B,W,D)，当前形状为 {tuple(z.shape)}")

    B = z.shape[0]
    n = max(1, B // num_bins)

    sorted_idx = torch.argsort(scores, descending=False)
    neg_idx = sorted_idx[:n]    # 低分：伪正常
    pos_idx = sorted_idx[-n:]   # 高分：伪异常

    z_pos = z[pos_idx]   # (n, D)
    z_neg = z[neg_idx]   # (n, D)

    mu_pos = z_pos.mean(dim=0)   # (D,)
    mu_neg = z_neg.mean(dim=0)   # (D,)

    # 质心距离分离损失
    dist = (mu_pos - mu_neg).norm()
    sep = F.relu(margin - dist)

    # 组内方差惩罚（促进类内紧凑）
    var_pos = ((z_pos - mu_pos.unsqueeze(0)) ** 2).mean()
    var_neg = ((z_neg - mu_neg.unsqueeze(0)) ** 2).mean()

    return sep + 0.1 * (var_pos + var_neg)


def temporal_consistency_loss(
    point_scores: torch.Tensor,
    window_ids: torch.Tensor,
    tv_weight: float = 0.1,
    smoothness_weight: float = 0.05,
) -> torch.Tensor:
    """时序连续性损失。

    约束相邻时间点的异常分数变化平滑，减少碎片化判决。
    按 window_ids 排序还原时序，计算一阶差分的 TV 损失与平滑损失：
        L = tv_weight * mean(|Δs|) + smoothness_weight * mean(Δs²)

    Args:
        point_scores: (B,) 点级异常分数
        window_ids: (B,) 样本所属时间窗 ID（用于还原时序）
        tv_weight: 总变差权重（允许突变但惩罚高频噪声）
        smoothness_weight: 平滑性权重（强制缓慢变化）
    Returns:
        标量损失值
    """
    if point_scores.shape[0] < 2:
        return point_scores.sum() * 0.0

    sort_idx = torch.argsort(window_ids)
    sorted_scores = point_scores[sort_idx]

    diff = sorted_scores[1:] - sorted_scores[:-1]
    tv_loss = diff.abs().mean() * tv_weight
    smooth_loss = (diff ** 2).mean() * smoothness_weight

    return tv_loss + smooth_loss


def joint_loss_v2(
    x_hat: torch.Tensor,
    x_target: torch.Tensor,
    z: torch.Tensor,
    re_encoder: Callable[[torch.Tensor], torch.Tensor],
    scores: torch.Tensor | None = None,
    lambda_idem: float = 1.0,
    lambda_contrast: float = 0.5,
    lambda_sep: float = 0.3,
    lambda_tv: float = 0.0,
    warmup_steps: int = 0,
    current_step: int = 0,
) -> tuple[torch.Tensor, dict]:
    """扩展版联合损失，支持伪标签对比学习、分离约束和时序连续性。

    总损失 = L_mse + lambda_idem*L_idem
           + lambda_contrast*L_contrast (若 scores 不为 None 且 lambda_contrast>0)
           + lambda_sep*L_sep           (若 scores 不为 None 且 lambda_sep>0)
           + lambda_tv*L_tv             (若 scores 不为 None 且 lambda_tv>0)

    Args:
        x_hat: (B, W, C) 重构输出
        x_target: (B, W, C) 原始输入
        z: (B, D) 融合潜向量
        re_encoder: 二次编码函数（用于等幂约束）
        scores: (B,) 当前 batch 的异常分数（用于伪标签构建，建议 detach）
        lambda_idem: 等幂约束权重
        lambda_contrast: 伪标签对比损失权重
        lambda_sep: 分离约束损失权重
        lambda_tv: 时序连续性损失权重
    Returns:
        (total_loss, loss_dict)，其中 loss_dict 各值均为 Tensor
    """
    loss_mse = mse_loss(x_hat, x_target)
    loss_idem = idempotent_loss(z, x_hat, re_encoder)
    total = loss_mse + lambda_idem * loss_idem

    loss_contrast = z.new_tensor(0.0)
    loss_sep = z.new_tensor(0.0)
    loss_tv = z.new_tensor(0.0)
    in_warmup = current_step < warmup_steps  # warm-up 期间跳过伪标签对比损失，避免噪声梯度

    if scores is not None and not in_warmup:
        if lambda_contrast > 0.0:
            loss_contrast = pseudo_label_contrastive_loss(z, scores)
            total = total + lambda_contrast * loss_contrast

        if lambda_sep > 0.0:
            loss_sep = separation_loss(z, scores)
            total = total + lambda_sep * loss_sep

        if lambda_tv > 0.0:
            window_ids = torch.arange(z.shape[0], device=z.device, dtype=z.dtype)
            loss_tv = temporal_consistency_loss(scores, window_ids)
            total = total + lambda_tv * loss_tv

    return total, {
        'mse': loss_mse,
        'idem': loss_idem,
        'contrast': loss_contrast,
        'sep': loss_sep,
        'tv': loss_tv,
        'in_warmup': in_warmup,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MTAD-GAT 启发：预测头损失 + 重构平滑性正则（修正版 TV）
# ─────────────────────────────────────────────────────────────────────────────

def forecast_mse_loss(
    x_forecast: torch.Tensor,
    x_next: torch.Tensor,
) -> torch.Tensor:
    """1-step 超前预测损失：L_forecast = mean((x_forecast - x_next)²)。

    受 MTAD-GAT 的 forecasting head 启发：纯重构头对"上下文型异常"
    （值域正常但趋势突变）天然盲区，叠加预测误差能有效检测此类异常。

    Args:
        x_forecast: (B, W-1, C)  Reconstructor.forecast(z) 的输出
        x_next    : (B, W-1, C)  对应真实值，即 x_seq[:, 1:, :]
    Returns:
        标量损失
    """
    if x_forecast.shape != x_next.shape:
        raise ValueError(
            f"x_forecast 与 x_next 形状必须一致，"
            f"当前分别为 {tuple(x_forecast.shape)} 与 {tuple(x_next.shape)}"
        )
    return F.mse_loss(x_forecast, x_next, reduction="mean")


def reconstruction_smoothness_loss(x_hat: torch.Tensor) -> torch.Tensor:
    """重构输出的样本内时序平滑损失（修正版时序连续性正则）。

    警告：对含合法高频切换信号的数据集（如 BATADAL），
    请将 lambda_tv 设为 0.0，否则会惩罚正常切换信号，
    可能导致正常的传感器快速切换被误判为异常。

    原 temporal_consistency_loss 以 batch 内打乱后的样本索引为"时序"代理，
    在 shuffle=True 的训练 DataLoader 下等价于惩罚批内分数方差，语义错误。

    本函数改为直接对重构输出 x_hat: (B, W, C) 计算样本内时间差分：
        L = mean(|x_hat[:, t+1] - x_hat[:, t]|)

    物理含义：防止重构头产生高频毛刺（对正常段的过度拟合），
    迫使模型还原平滑的背景趋势而非记忆噪声细节，
    从而提高异常时刻的重构误差与正常时刻的对比度。

    Args:
        x_hat: (B, W, C) 重构输出
    Returns:
        标量损失
    """
    if x_hat.ndim != 3:
        raise ValueError(f"x_hat 需要 3D 输入 (B,W,C)，当前形状为 {tuple(x_hat.shape)}")
    if x_hat.shape[1] < 2:
        return x_hat.sum() * 0.0
    diff = x_hat[:, 1:, :] - x_hat[:, :-1, :]   # (B, W-1, C)
    return diff.abs().mean()
