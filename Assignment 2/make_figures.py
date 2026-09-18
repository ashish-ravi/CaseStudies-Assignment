"""Figures for the Task 2 report. Style matches the Task 1 figures."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import prep

OUT, IMG = prep.OUTPUT_DIR, prep.IMG_DIR
IMG.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({"savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 7,
                     "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
BLUE, ORANGE = "#0072B2", "#D55E00"

# ---- Figure 1: learning curves ------------------------------------------
lc_o = pd.read_csv(OUT / "learning_curve_olist.csv")
lc_f = pd.read_csv(OUT / "learning_curve_food.csv")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.0))
for ax, lc, ylabel, title, cap_note in [
    (axes[0], lc_o, "Average precision", "Olist: late delivery classification", 15000),
    (axes[1], lc_f, "MAE (minutes)", "Food delivery: trip time regression", 15000),
]:
    for model, colour, marker in [("XGBoost", BLUE, "o"), ("SVM", ORANGE, "s")]:
        d = lc[lc["model"] == model]
        ax.plot(d["n_train"], d["score"], marker=marker, markersize=3.2,
                linewidth=1.2, color=colour, label=model)
    # The training size Task 1 capped the support vector machines at.
    ax.axvline(cap_note, color="grey", linestyle=":", linewidth=0.9)
    ax.annotate("Task 1 SVM cap", xy=(cap_note, ax.get_ylim()[1]),
                xytext=(cap_note * 1.15, ax.get_ylim()[1]), fontsize=5.8,
                color="grey", va="top")
    ax.set_xscale("log")
    ax.set_xlabel("Training rows")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=6.5)
fig.tight_layout(w_pad=1.4)
fig.savefig(IMG / "learning_curves.pdf")
print("wrote learning_curves.pdf")

# ---- Figure 2: fairness by region ---------------------------------------
reg = pd.read_csv(OUT / "fairness_olist_region.csv")
reg = reg.sort_values("false_negative_rate")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 1.9))

ax = axes[0]
ax.barh(reg["group"], reg["false_negative_rate"], color=BLUE, height=0.62)
overall_fnr = 1 - pd.read_csv(OUT / "fairness_olist_summary.csv")["overall_recall"].iloc[0]
ax.axvline(overall_fnr, color="black", linestyle="--", linewidth=0.9)
# x in data units, y as a fraction of the axes, so the label clears the bars.
ax.text(overall_fnr + 0.006, 0.03, f"overall {overall_fnr:.2f}", fontsize=5.8,
        color="black", ha="left", va="bottom", transform=ax.get_xaxis_transform())
ax.set_xlabel("Missed late deliveries (false negative rate)")
ax.set_title("Who the model fails to warn")

ax = axes[1]
width = 0.38
ypos = range(len(reg))
ax.barh([y + width / 2 for y in ypos], reg["base_rate"], height=width,
        color=ORANGE, label="Actually late")
ax.barh([y - width / 2 for y in ypos], reg["selection_rate"], height=width,
        color=BLUE, label="Flagged by model")
ax.set_yticks(list(ypos))
ax.set_yticklabels(reg["group"])
ax.set_xlabel("Share of orders")
ax.set_title("Flagging rate against actual lateness")
ax.legend(frameon=False, fontsize=6.5)

fig.tight_layout(w_pad=1.4)
fig.savefig(IMG / "fairness_region.pdf")
print("wrote fairness_region.pdf")
