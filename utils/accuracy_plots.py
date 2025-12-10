import pandas as pd
import matplotlib.pyplot as plt

rows = [
    ("ResNet-18", "None", "CIFAR-10", 93.95),
    ("ResNet-18", "LeRaC", "CIFAR-10", 87.46),
    ("ResNet-18", "Data-Level (Blur)", "CIFAR-10", 93.90),
    ("ResNet-18", "Hybrid", "CIFAR-10", 93.43),    
    ("ResNet-18", "None", "ImageNet-100", 79.80),
    ("ResNet-18", "LeRaC", "ImageNet-100", 80.64),
    ("ResNet-18", "Data-Level (Blur)", "ImageNet-100", 79.30),
    ("ResNet-18", "Hybrid", "ImageNet-100", 81.08),
    ("CvT-13", "None", "CIFAR-10", 84.17),
    ("CvT-13", "LeRaC", "CIFAR-10", 82.81),
    ("CvT-13", "Data-Level (Blur)", "CIFAR-10", 83.44),
    ("CvT-13", "Hybrid", "CIFAR-10", 83.07),
    ("CvT-13", "None", "ImageNet-100", 81.12),
    ("CvT-13", "LeRaC", "ImageNet-100", 76.68),
    ("CvT-13", "Data-Level (Blur)", "ImageNet-100", 82.84),
    ("CvT-13", "Hybrid", "ImageNet-100", 80.56),
    ("ConvNeXt-Tiny", "None", "CIFAR-10", 95.83),
    ("ConvNeXt-Tiny", "LeRaC", "CIFAR-10", 96.46),
    ("ConvNeXt-Tiny", "Data-Level (Blur)", "CIFAR-10", 95.99),
    ("ConvNeXt-Tiny", "Hybrid", "CIFAR-10", 94.44),    
    ("ConvNeXt-Tiny", "None", "ImageNet-100", 84.53),
    ("ConvNeXt-Tiny", "LeRaC", "ImageNet-100", 83.65),
    ("ConvNeXt-Tiny", "Data-Level (Blur)", "ImageNet-100", 84.18),
    ("ConvNeXt-Tiny", "Hybrid", "ImageNet-100", 82.73)    
]

df = pd.DataFrame(rows, columns=["Model", "Curriculum", "Dataset", "Accuracy"])

def plot_grouped_bar_unified(df_subset, title, ylim=(0, 100)):
    plt.clf()
    pivot = df_subset.pivot(index="Model", columns="Curriculum", values="Accuracy")
    pivot = pivot.reindex(["ResNet-18", "CvT-13", "ConvNeXt-Tiny"])
    pivot = pivot[["None", "Data-Level (Blur)", "LeRaC", "Hybrid"]].dropna(axis=1, how='all')
    
    ax = pivot.plot(kind="bar", figsize=(9, 5), width=0.75)
    ax.set_title(title)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_xlabel("Model")
    ax.set_ylim(ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Curriculum", bbox_to_anchor=(1.05, 1), loc="upper left")
    
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3)
    
    plt.tight_layout()
    plt.show()

plot_grouped_bar_unified(df[df["Dataset"]=="CIFAR-10"], "CIFAR-10 Accuracy by Model & Curriculum")
plot_grouped_bar_unified(df[df["Dataset"]=="ImageNet-100"], "ImageNet-100 Accuracy by Model & Curriculum")
