"""
run_inference.py — 只跑推理，把结果存成 npy，不画图。
后续改图直接跑 plot_figures.py，不需要重新推理。

用法（在项目根目录）：
  PYTHONPATH=$(pwd) python pics/run_inference.py \
      --out_dir pics/results/
"""

import argparse, os, sys
import numpy as np
import torch

_script_dir   = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
for _p in [_project_root, _script_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model.AnomalyTransformer import AnomalyTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir",    default="pics/results", help="推理结果保存目录")
parser.add_argument("--win_size",   type=int, default=100)
parser.add_argument("--d_model",    type=int, default=512)
parser.add_argument("--d_ff",       type=int, default=512)
parser.add_argument("--n_heads",    type=int, default=8)
parser.add_argument("--e_layers",   type=int, default=3)
parser.add_argument("--batch_size", type=int, default=256)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── 数据集配置 ────────────────────────────────────────────────────────────
DATASETS = {
    "HAI":  {"data_path": "data/HAI",  "input_c": 59, "prefix": "HAI"},
    "MSL":  {"data_path": "data/MSL",  "input_c": 55, "prefix": "MSL"},
    "SKAB": {"data_path": "data/SKAB", "input_c": 8,  "prefix": "SKAB"},
}
# E1~E4 的 checkpoint 路径
CKPTS = {
    "HAI": {
        "E1": "checkpoints/E1_HAI/HAI_checkpoint.pth",
        "E2": "checkpoints/E2_HAI/HAI_checkpoint.pth",
        "E3": "checkpoints/E3_HAI/HAI_checkpoint.pth",
        "E4": "checkpoints/E4_HAI/HAI_checkpoint.pth",
    },
    "MSL": {
        "E1": "checkpoints/E1_MSL/MSL_checkpoint.pth",
        "E2": "checkpoints/E2_MSL/MSL_checkpoint.pth",
        "E3": "checkpoints/E3_MSL/MSL_checkpoint.pth",
        "E4": "checkpoints/E4_MSL/MSL_checkpoint.pth",
    },
    "SKAB": {
        "E1": "checkpoints/E1_SKAB/SKAB_checkpoint.pth",
        "E2": "checkpoints/E2_SKAB/SKAB_checkpoint.pth",
        "E3": "checkpoints/E3_SKAB/SKAB_checkpoint.pth",
        "E4": "checkpoints/E4_SKAB/SKAB_checkpoint.pth",
    },
}

# ── 工具函数 ──────────────────────────────────────────────────────────────
def find_file(data_path, prefix, suffix):
    for name in [f"{prefix}{suffix}", suffix.lstrip("_")]:
        p = os.path.join(data_path, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"找不到 {data_path}/{prefix}{suffix}")

def build_windows(data, win):
    n = data.shape[0] - win + 1
    idx = np.arange(win)[None, :] + np.arange(n)[:, None]
    return data[idx]

def find_attn_win(win_labels):
    anom_idx = np.where(win_labels == 1)[0]
    if len(anom_idx) == 0:
        return len(win_labels) // 2
    gaps      = np.where(np.diff(anom_idx) > 1)[0]
    seg_start = anom_idx[0]
    seg_end   = anom_idx[gaps[0]] if len(gaps) > 0 else anom_idx[-1]
    return int((seg_start + seg_end) // 2)

def load_model(ckpt_path, dgr_mode, C):
    model = AnomalyTransformer(
        win_size=args.win_size, enc_in=C, c_out=C,
        d_model=args.d_model, n_heads=args.n_heads,
        e_layers=args.e_layers, d_ff=args.d_ff,
        dropout=0.0, activation="gelu",
        output_attention=True, dgr_mode=dgr_mode,
    ).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model

def run_inference(model, windows_np, attn_win_idx):
    N, eps = windows_np.shape[0], 1e-8
    scores       = np.zeros(N, dtype=np.float32)
    saved_series = None
    with torch.no_grad():
        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            x   = torch.tensor(windows_np[start:end]).to(DEVICE)
            _, series_list, prior_list, _ = model(x)
            sl = series_list[-1].cpu().float().numpy()
            pl = prior_list[-1].cpu().float().numpy()
            kl_sp = (sl * (np.log(sl + eps) - np.log(pl + eps))).sum(-1)
            kl_ps = (pl * (np.log(pl + eps) - np.log(sl + eps))).sum(-1)
            scores[start:end] = (kl_sp + kl_ps).mean(1)[:, -1]
            if start <= attn_win_idx < end:
                saved_series = sl[attn_win_idx - start]   # [H, W, W]
    return scores, saved_series

# ── 主循环 ────────────────────────────────────────────────────────────────
DGR_MODES = {"E1": "none", "E2": "dynamic", "E3": "multiscale", "E4": "static"}

for ds, cfg in DATASETS.items():
    print(f"\n{'='*60}\n  {ds}\n{'='*60}")
    prefix = cfg["prefix"]
    C      = cfg["input_c"]

    test_data  = np.load(find_file(cfg["data_path"], prefix, "_test.npy")).astype(np.float32)
    test_label = np.load(find_file(cfg["data_path"], prefix, "_test_label.npy")).astype(np.int32)
    T = test_data.shape[0]
    print(f"  {T} steps, {C} ch, anomaly={test_label.mean():.4f}")

    mean = test_data.mean(0, keepdims=True)
    std  = test_data.std(0,  keepdims=True) + 1e-8
    windows    = build_windows((test_data - mean) / std, args.win_size)
    win_labels = test_label[args.win_size - 1:]
    attn_win   = find_attn_win(win_labels)
    print(f"  attn_win={attn_win}  label={win_labels[attn_win]}")

    # 保存 labels（画图用）
    np.save(os.path.join(args.out_dir, f"{ds}_win_labels.npy"), win_labels)
    np.save(os.path.join(args.out_dir, f"{ds}_attn_win.npy"),   np.array(attn_win))

    for exp, dgr_mode in DGR_MODES.items():
        ckpt = CKPTS[ds][exp]
        print(f"  {exp}...", flush=True)
        model  = load_model(ckpt, dgr_mode, C)
        scores, attn_mat = run_inference(model, windows, attn_win)
        del model
        torch.cuda.empty_cache()

        np.save(os.path.join(args.out_dir, f"{ds}_{exp}_scores.npy"),   scores)
        np.save(os.path.join(args.out_dir, f"{ds}_{exp}_attn_E1E2.npy"), attn_mat)  # [H,W,W]

    print(f"  {ds} 推理完成，结果已保存至 {args.out_dir}/")

print("\n全部推理完成。")