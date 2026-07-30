import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# Embedded data
# Safety-score distribution counts
# Bin width = 5
# Bins: 0–5, 5–10, ..., 95–100
# --------------------------------------------------
bins = np.arange(0, 105, 5)
bin_width = 5
bin_centers = bins[:-1] + bin_width / 2

counts = {
    "nuScenes": [
        0, 0, 0, 0, 1,
        0, 0, 1, 0, 0,
        0, 2, 1, 1, 3,
        0, 0, 1, 0, 0
    ],

    "Argoverse2": [
        0, 0, 1, 3, 9,
        6, 13, 16, 19, 16,
        17, 22, 19, 213, 474,
        146, 22, 2, 2, 0
    ],

    "Waymo": [
        0, 0, 0, 2, 2,
        0, 2, 6, 5, 6,
        3, 5, 9, 143, 63,
        22, 7, 5, 6, 0
    ]
}

labels = {
    "nuScenes": "nuScenes (n=10)",
    "Argoverse2": "Argoverse 2 (n=1,000)",
    "Waymo": "Waymo WOMD (n=286)"
}

colors = {
    "nuScenes": "#1f77b4",
    "Argoverse2": "#ff7f0e",
    "Waymo": "#2ca02c"
}

order = ["nuScenes", "Argoverse2", "Waymo"]

# --------------------------------------------------
# Tier colors
# --------------------------------------------------
tier_fill_colors = {
    "Emergency": "#d62728",
    "Intervention": "#ff7f0e",
    "Advisory": "#f2c300",
    "Silent": "#2ca02c"
}

tier_text_colors = {
    "Emergency": "#8B0000",
    "Intervention": "#B35A00",
    "Advisory": "#8A6D00",
    "Silent": "#006400"
}

tier_alpha = 0.12

# --------------------------------------------------
# Figure setup
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(4.5, 2.5), dpi=600)

# --------------------------------------------------
# Tier zone shading — no boundary lines, shading shows zones
# --------------------------------------------------
ax.axvspan(0,  20,  alpha=tier_alpha, color=tier_fill_colors["Emergency"],    zorder=0)
ax.axvspan(20, 40,  alpha=tier_alpha, color=tier_fill_colors["Intervention"], zorder=0)
ax.axvspan(40, 70,  alpha=tier_alpha, color=tier_fill_colors["Advisory"],     zorder=0)
ax.axvspan(70, 100, alpha=tier_alpha, color=tier_fill_colors["Silent"],       zorder=0)

# --------------------------------------------------
# Tier labels inside shaded regions
# --------------------------------------------------
ax.text(10,  0.095, "Emergency (0–20)",      ha="center", va="top", fontsize=4.2, color=tier_text_colors["Emergency"])
ax.text(30,  0.095, "Intervention (20–40)",  ha="center", va="top", fontsize=4.2, color=tier_text_colors["Intervention"])
ax.text(55,  0.095, "Advisory (40–70)",      ha="center", va="top", fontsize=4.2, color=tier_text_colors["Advisory"])
ax.text(85,  0.095, "Silent (70–100)",       ha="center", va="top", fontsize=4.2, color=tier_text_colors["Silent"])

# --------------------------------------------------
# Plot density curves from embedded bin counts
# --------------------------------------------------
for dataset in order:
    n = sum(counts[dataset])
    density = np.array(counts[dataset]) / (n * bin_width)
    ax.plot(bin_centers, density, linewidth=1.8, color=colors[dataset], label=labels[dataset], zorder=3)
    ax.fill_between(bin_centers, density, alpha=0.20, color=colors[dataset], zorder=2)

# --------------------------------------------------
# Overall mean line
# --------------------------------------------------
overall_mean = 68.4  # weighted mean from raw scenario data (1,296 scenarios)

ax.axvline(
    overall_mean,
    color="black",
    linestyle="--",
    linewidth=1.2,
    alpha=0.80,
    label=f"Overall mean ({overall_mean:.1f})",
    zorder=4
)

# --------------------------------------------------
# Titles and labels
# --------------------------------------------------
ax.set_title("Safety Score Distribution Across Datasets", fontsize=7)
ax.set_xlabel("Safety Score (0–100)", fontsize=6)
ax.set_ylabel("Density", fontsize=6)

ax.set_xlim(0, 100)
ax.set_ylim(0, 0.10)

ax.set_xticks(np.arange(0, 101, 20))

yticks = np.arange(0, 0.1001, 0.025)
ax.set_yticks(yticks)
ax.set_yticklabels(["0", "0.025", "0.050", "0.075", "0.100"])

# --------------------------------------------------
# Tick font style
# --------------------------------------------------
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)

# --------------------------------------------------
# Axis spine style
# --------------------------------------------------
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("black")
ax.spines["bottom"].set_color("black")
ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)

# --------------------------------------------------
# Grid and legend
# --------------------------------------------------
ax.grid(axis="y", linewidth=0.4, alpha=0.3)

ax.legend(
    loc="upper left",
    bbox_to_anchor=(0.01, 0.82),
    fontsize=4,
    frameon=True,
    borderaxespad=0.2
)

# --------------------------------------------------
# Layout and save
# --------------------------------------------------
plt.tight_layout()
plt.subplots_adjust(left=0.13, bottom=0.18)

plt.savefig(
    "fig2_score_distribution_final.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()
