import pandas as pd
import matplotlib.pyplot as plt

rows = [

    ("CvT-13", "Hybrid", "CIFAR-10", 83.07),
    ("CvT-13", "Hybrid w/Extension", "CIFAR-10", 82.03),
    ("CvT-13", "Hybrid", "ImageNet-100", 80.56),
    ("CvT-13", "Hybrid w/Extension", "ImageNet-100", 79.94),
]

df = pd.DataFrame(rows, columns=["Model", "Curriculum", "Dataset", "Accuracy"])

def plot_grouped_bar_unified(df_subset, title, ylim=(0, 100)):
    plt.clf()
    df_subset = df_subset[df_subset["Model"] == "CvT-13"]

    pivot = df_subset.pivot(index="Dataset", columns="Curriculum", values="Accuracy")
    pivot = pivot.reindex(["CIFAR-10", "ImageNet-100"])

    cols = ["Hybrid", "Hybrid w/Extension"]
    cols = [c for c in cols if c in pivot.columns]
    pivot = pivot[cols].dropna(axis=1, how='all')

    ax = pivot.plot(kind="bar", figsize=(9, 5), width=0.6)
    ax.set_title(title)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_xlabel("Dataset")
    ax.set_ylim(ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Curriculum", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xticklabels(pivot.index, rotation=0)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3)

    plt.tight_layout()
    plt.show()

plot_grouped_bar_unified(df, "Hybrid vs Hybrid w/Extension Accuracy for CvT-13")
