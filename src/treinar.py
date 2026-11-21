"""Treina o modelo final (Rede 2) e salva tudo o que o app precisa em modelos/.

Uso:  python -m src.treinar --csv data/accident_data.csv
Leva ~2 minutos em CPU. O app (app.py) só lê os arquivos salvos; não precisa do dataset.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src import dados
from src.modelo import metricas, prever, treinar, captura_topk

M_ALISAMENTO = 100  # força do alisamento das taxas (igual ao notebook)


def taxa_alisada(s, n, p0, m=M_ALISAMENTO):
    return (s + m * p0) / (n + m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/accident_data.csv")
    ap.add_argument("--saida", default="modelos")
    a = ap.parse_args()
    out = Path(a.saida)
    out.mkdir(exist_ok=True)
    Path("figuras").mkdir(exist_ok=True)

    d = dados.features_calendario(dados.limpar(dados.carregar(a.csv)), corrigir_dia_semana=True)
    train, val, test = dados.dividir_aleatorio(d)
    prep = dados.Preparador().ajustar(train)
    Xc_tr, Xn_tr, y_tr = prep.transformar(train)
    Xc_va, Xn_va, y_va = prep.transformar(val)
    Xc_te, Xn_te, y_te = prep.transformar(test)
    print(f"treino {len(train)} | validação {len(val)} | teste {len(test)}")

    modelo, hist = treinar(prep.vocab, Xc_tr, Xn_tr, y_tr, Xc_va, Xn_va, y_va, semente=42)
    s_va, s_te = prever(modelo, Xc_va, Xn_va), prever(modelo, Xc_te, Xn_te)
    met = metricas(y_te, s_te)
    print("teste:", {k: round(v, 4) for k, v in met.items()})

    # Faixas de risco (decis) definidas na validação
    cortes = np.quantile(s_va, np.linspace(0.1, 0.9, 9))
    decil_va = np.searchsorted(cortes, s_va) + 1
    taxa_decil = pd.Series(y_va).groupby(decil_va).mean()

    # Calibração no teste
    cal = pd.DataFrame({"p": s_te, "y": y_te})
    cal["faixa"] = pd.qcut(cal["p"], 10, labels=False, duplicates="drop")
    cal = cal.groupby("faixa").agg(prevista=("p", "mean"), real=("y", "mean"))

    # Localização típica de cada distrito (treino) e área mais comum
    dist = train.groupby("District Area").agg(lat=("Latitude", "mean"), lon=("Longitude", "mean"),
                                              n=("grave", "size"), s=("grave", "sum"))
    dist["area_comum"] = train.groupby("District Area")["Urban_or_Rural_Area"].agg(
        lambda x: x.value_counts().index[0])
    p0 = float(train["grave"].mean())

    # Mapa e ranking com treino + validação (o teste fica de fora)
    tv = pd.concat([train, val])
    p0tv = float(tv["grave"].mean())
    rk = tv.groupby("District Area").agg(acidentes=("grave", "size"), graves=("grave", "sum"))
    rk["taxa_alisada"] = taxa_alisada(rk["graves"], rk["acidentes"], p0tv)
    rk.sort_values("taxa_alisada", ascending=False).to_csv(out / "distritos_risco.csv")
    STEP = 0.05
    cel = tv.assign(cel=(np.floor(tv["Latitude"] / STEP).astype(int).astype(str) + "_" +
                         np.floor(tv["Longitude"] / STEP).astype(int).astype(str)))
    mp = cel.groupby("cel").agg(n=("grave", "size"), s=("grave", "sum"),
                                lat=("Latitude", "mean"), lon=("Longitude", "mean"))
    mp = mp[mp["n"] >= 100].copy()
    mp["taxa"] = taxa_alisada(mp["s"], mp["n"], p0tv)
    mp.round(5).to_csv(out / "celulas_risco.csv")

    torch.save(modelo.state_dict(), out / "rede_final.pt")
    art = {
        "cats": dados.CATS, "num": prep.num, "maps": prep.maps, "mu": prep.mu, "sd": prep.sd,
        "vocab": prep.vocab, "cortes": cortes.tolist(),
        "taxa_decil": {int(k): float(v) for k, v in taxa_decil.items()},
        "media_geral": p0,
        "distritos": {k: {"lat": float(r.lat), "lon": float(r.lon), "n": int(r.n),
                          "area_comum": r.area_comum} for k, r in dist.iterrows()},
        "metricas_teste": {k: float(v) for k, v in met.items()},
        "calibracao_teste": cal.round(4).reset_index().to_dict(orient="records"),
        "historico_treino": hist,
        "deslocamento_anos": dados.DESLOCAMENTO_ANOS,
    }
    json.dump(art, open(out / "artefatos.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # Figuras para o README
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    fr = np.linspace(0.01, 1.0, 100)
    ax[0].plot(fr * 100, [captura_topk(y_te, s_te, f) * 100 for f in fr], "-", label="Rede neural")
    ax[0].plot(fr * 100, fr * 100, ":", label="Acaso")
    ax[0].set_xlabel("% dos acidentes priorizados (maior risco primeiro)")
    ax[0].set_ylabel("% dos graves/fatais capturados")
    ax[0].set_title("Curva de ganho (teste)")
    ax[0].legend()
    ax[1].plot([0, 0.3], [0, 0.3], ":", color="gray", label="calibração perfeita")
    ax[1].plot(cal["prevista"], cal["real"], "o-", label="Rede neural")
    ax[1].set_xlabel("probabilidade prevista (média do decil)")
    ax[1].set_ylabel("proporção real de graves/fatais")
    ax[1].set_title("Calibração por decil (teste)")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("figuras/ganho_calibracao.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6.5, 8))
    sc = plt.scatter(mp["lon"], mp["lat"], c=mp["taxa"] * 100, s=np.clip(np.sqrt(mp["n"]), 4, 60),
                     cmap="YlOrRd", alpha=0.85, vmin=mp["taxa"].quantile(0.05) * 100,
                     vmax=mp["taxa"].quantile(0.95) * 100)
    plt.colorbar(sc, label="% de graves/fatais (taxa alisada)")
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.title("Risco regional (células de ~5 km)")
    plt.gca().set_aspect(1.6)
    plt.savefig("figuras/mapa_risco.png", dpi=150, bbox_inches="tight")
    plt.close()

    h = pd.DataFrame(hist)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(h["epoca"], h["custo_treino"], "o-", label="custo (treino)")
    ax1.set_xlabel("época")
    ax1.set_ylabel("custo BCE (treino)")
    ax2 = ax1.twinx()
    ax2.plot(h["epoca"], h["pr_auc_val"], "s--", color="tab:orange", label="PR-AUC (validação)")
    ax2.set_ylabel("PR-AUC (validação)")
    fig.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    ax1.set_title("Treinamento da rede final (early stopping)")
    plt.tight_layout()
    plt.savefig("figuras/curva_treino.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("artefatos salvos em", out)


if __name__ == "__main__":
    main()
