"""Script pembuatan visualisasi evaluasi, perbandingan model, dan analisis multikompetensi."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Atur style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def generate_all_figures(
    outputs_dir: str | Path = "outputs",
    figures_dir: str | Path = "figures"
):
    out_path = Path(outputs_dir)
    fig_path = Path(figures_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    # 1. Bar Chart Perbandingan Metrik Evaluasi Independen
    eval_csv = out_path / "evaluasi_metrik_perbandingan.csv"
    if eval_csv.exists():
        df_eval = pd.read_csv(eval_csv)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(df_eval))
        width = 0.18

        m_ndcg = ax.bar(x - 1.5 * width, df_eval["ndcg@5"], width, label="NDCG@5", color="#1f77b4")
        m_p1 = ax.bar(x - 0.5 * width, df_eval["precision@1"], width, label="Precision@1", color="#2ca02c")
        m_p5 = ax.bar(x + 0.5 * width, df_eval["precision@5"], width, label="Precision@5", color="#ff7f0e")
        m_map = ax.bar(x + 1.5 * width, df_eval["map@5"], width, label="MAP@5", color="#9467bd")

        ax.set_title("Perbandingan Metrik Relevansi Multi-Label pada Ground Truth Independen", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Nilai Metrik (0.0 - 1.0)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(df_eval["model"], rotation=15, ha="right", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=True, loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # Label nilai
        for bars in [m_ndcg, m_p1]:
            for b in bars:
                h = b.get_height()
                if h > 0:
                    ax.annotate(f"{h:.2f}",
                                xy=(b.get_x() + b.get_width() / 2, h),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        plt.savefig(fig_path / "01_perbandingan_metrik_evaluasi.png", dpi=200)
        plt.close()

        # 2. Scatter Plot Trade-off Relevansi vs Latensi
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        for idx, row in df_eval.iterrows():
            ax.scatter(row["avg_latency_ms"], row["ndcg@5"], s=180, color=colors[idx % len(colors)], zorder=3, edgecolors="black")
            ax.annotate(row["model"],
                        (row["avg_latency_ms"], row["ndcg@5"]),
                        xytext=(8, 4), textcoords="offset points",
                        fontsize=10, fontweight="bold")

        ax.set_title("Trade-off Kualitas Relevansi (NDCG@5) vs Latensi Komputasi (ms)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Rata-rata Latensi Inferensi per Buku (Milidetik, Skala Log)", fontsize=11)
        ax.set_ylabel("NDCG@5 (Relevansi Multi-Label)", fontsize=11)
        ax.set_xscale("log")
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(fig_path / "02_tradeoff_relevansi_latensi.png", dpi=200)
        plt.close()

    # 3. Distribusi Jumlah Kompetensi Terpilih per Model
    multi_csv = out_path / "buku_multikompetensi_per_model.csv"
    if multi_csv.exists():
        df_multi = pd.read_csv(multi_csv)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df_multi, x="model", y="jumlah_kompetensi_kandidat", ax=ax, palette="Set2")
        ax.set_title("Distribusi Jumlah Kompetensi Kandidat per Buku (973 Buku)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Model NLP", fontsize=11)
        ax.set_ylabel("Jumlah Kompetensi Terpilih per Buku", fontsize=11)
        ax.set_ylim(0, 7)
        plt.tight_layout()
        plt.savefig(fig_path / "03_distribusi_multikompetensi.png", dpi=200)
        plt.close()

    print(f"Seluruh visualisasi evaluasi berhasil disimpan di: {fig_path}")


if __name__ == "__main__":
    generate_all_figures()
