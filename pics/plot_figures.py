import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", default="pics/results")
parser.add_argument("--out_dir",    default="pics")
parser.add_argument("--dpi",        type=int, default=300)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

COLORS = {"E1": "#4C72B0", "E2": "#DD8452", "E3": "#55A868", "E4": "#C44E52"}
LABELS = {
    "E1": "E1 (Gaussian Prior)",
    "E2": "E2 (Dynamic DGR Prior)",
    "E3": "E3 (Multi-scale DGR Prior)",
    "E4": "E4 (Static DGR Prior)",
}
DATASETS = ["HAI", "MSL", "SKAB"]

def load(ds, name):
    return np.load(os.path.join(args.result_dir, f"{ds}_{name}.npy"))

# ── Fig 1: Attention heatmaps, 2 rows x 3 cols ───────────────────────────
print("Plotting Fig 1...")
fig1, axes = plt.subplots(2, 3, figsize=(15, 8))

for col, ds in enumerate(DATASETS):
    for row, exp in enumerate(["E1", "E2"]):
        ax  = axes[row][col]
        mat = load(ds, f"{exp}_attn_E1E2").mean(0)
        im  = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=mat.max())
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if row == 0:
            ax.set_title(ds, fontsize=12, fontweight="bold")
        ax.set_ylabel(f"{exp}\nQuery", fontsize=8)
        ax.set_xlabel("Key", fontsize=8)

plt.tight_layout()
out1 = os.path.join(args.out_dir, "fig4-1_attention_heatmap.png")
fig1.savefig(out1, dpi=args.dpi, bbox_inches="tight")
print(f"  -> {out1}")
plt.close(fig1)

# ── Fig 2: Anomaly score curves, 3 rows x 1 col ──────────────────────────
print("Plotting Fig 2...")
fig2, axes2 = plt.subplots(3, 1, figsize=(14, 11))

for row, ds in enumerate(DATASETS):
    ax         = axes2[row]
    win_labels = load(ds, "win_labels").astype(int)
    N          = len(win_labels)

    anom_pos = np.where(win_labels == 1)[0]
    if len(anom_pos) == 0:
        plot_start, plot_end = 0, min(N, 2000)
    else:
        gaps       = np.where(np.diff(anom_pos) > 50)[0]
        seg_start  = anom_pos[0]
        seg_end    = anom_pos[gaps[0]] if len(gaps) > 0 else anom_pos[-1]
        plot_start = max(0, seg_start - 300)
        plot_end   = min(N, seg_end   + 300)

    x_axis = np.arange(plot_start, plot_end)

    in_anom, anom_sx = False, None
    for xi in x_axis:
        if win_labels[xi] == 1 and not in_anom:
            anom_sx = xi;  in_anom = True
        elif win_labels[xi] == 0 and in_anom:
            ax.axvspan(anom_sx, xi, color="#DDDDDD", alpha=0.65, zorder=0)
            in_anom = False
    if in_anom:
        ax.axvspan(anom_sx, x_axis[-1], color="#DDDDDD", alpha=0.65, zorder=0)

    for exp in ["E1", "E2", "E3", "E4"]:
        s      = load(ds, f"{exp}_scores")[plot_start:plot_end]
        s_norm = (s - s.min()) / (s.max() - s.min() + 1e-8)
        ax.plot(x_axis, s_norm, color=COLORS[exp], linewidth=0.9,
                label=LABELS[exp], alpha=0.9)

    anom_patch = mpatches.Patch(color="#DDDDDD", alpha=0.65, label="Anomaly Segment")
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles + [anom_patch], lbls + ["Anomaly Segment"],
              loc="upper left", fontsize=8, framealpha=0.85)

    ax.text(0.01, 0.96, ds, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")
    ax.set_xlabel("Window Index", fontsize=9)
    ax.set_ylabel("Normalized Anomaly Score", fontsize=9)
    ax.set_xlim(plot_start, plot_end)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

plt.tight_layout()
out2 = os.path.join(args.out_dir, "fig4-2_anomaly_score_curves.png")
fig2.savefig(out2, dpi=args.dpi, bbox_inches="tight")
print(f"  -> {out2}")
plt.close(fig2)

print(f"\nDone.\n  {out1}\n  {out2}")