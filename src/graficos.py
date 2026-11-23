"""Estilo único dos gráficos salvos em results/ (cores fixas por papel, eixos discretos)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AZUL, LARANJA, CINZA = "#2a78d6", "#eb6834", "#9a9893"
TINTA, TINTA2, GRADE, FUNDO = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
COR_CLASSE = {0: AZUL, 1: LARANJA}  # 0 = não reclamação, 1 = reclamação
COR_MODELO = {"Rede neural": AZUL, "Baseline (TF-IDF + LR)": LARANJA}

plt.rcParams.update({
    "figure.facecolor": FUNDO, "axes.facecolor": FUNDO, "savefig.facecolor": FUNDO,
    "axes.edgecolor": GRADE, "axes.labelcolor": TINTA2, "xtick.color": TINTA2,
    "ytick.color": TINTA2, "text.color": TINTA, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 11, "legend.frameon": False, "lines.linewidth": 2,
})


def salvar(fig, caminho):
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
