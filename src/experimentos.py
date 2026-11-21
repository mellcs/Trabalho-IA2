"""Experimentos extras planejados pelo grupo e que não estavam no notebook.

1. Correção do dia da semana (datas deslocadas em 12 anos).
2. Várias sementes: a diferença Rede x Regressão logística se mantém?
3. Validação temporal: treina em 2007-2009 reais e testa em 2010.
4. Ablação com Number_of_Vehicles (quanto ele acrescentaria).

Uso:  python -m src.experimentos --csv data/accident_data.csv
Saída: docs/resultados_experimentos.md e docs/resultados_experimentos.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from src import dados
from src.modelo import metricas, prever, treinar


def regressao_logistica(Xc_tr, Xn_tr, y_tr, Xc_te, Xn_te):
    ohe = OneHotEncoder(handle_unknown="ignore").fit(Xc_tr)
    A = hstack([ohe.transform(Xc_tr), csr_matrix(Xn_tr)]).tocsr()
    B = hstack([ohe.transform(Xc_te), csr_matrix(Xn_te)]).tocsr()
    lr = LogisticRegression(class_weight="balanced", max_iter=300).fit(A, y_tr)
    return lr.predict_proba(B)[:, 1]


def rodar(d, divisao, num=dados.NUM, corrigir=True, sementes=(42,), com_lr=True):
    d = dados.features_calendario(d, corrigir_dia_semana=corrigir)
    train, val, test = divisao(d)
    prep = dados.Preparador(num).ajustar(train)
    Xc_tr, Xn_tr, y_tr = prep.transformar(train)
    Xc_va, Xn_va, y_va = prep.transformar(val)
    Xc_te, Xn_te, y_te = prep.transformar(test)
    out = {"n_treino": len(train), "n_val": len(val), "n_teste": len(test),
           "taxa_teste": float(y_te.mean()), "rede": []}
    for s in sementes:
        m, _ = treinar(prep.vocab, Xc_tr, Xn_tr, y_tr, Xc_va, Xn_va, y_va, semente=s, verbose=False)
        out["rede"].append(metricas(y_te, prever(m, Xc_te, Xn_te)))
    if com_lr:
        out["lr"] = metricas(y_te, regressao_logistica(Xc_tr, Xn_tr, y_tr, Xc_te, Xn_te))
    return out


def resumo(lista, chave):
    v = np.array([x[chave] for x in lista])
    return f"{v.mean():.4f} ± {v.std():.4f}" if len(v) > 1 else f"{v[0]:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/accident_data.csv")
    ap.add_argument("--sementes", type=int, default=5)
    a = ap.parse_args()
    t0 = time.time()
    d = dados.limpar(dados.carregar(a.csv))
    sementes = tuple(range(42, 42 + a.sementes))
    R = {}

    print("1/4 dia da semana original (errado), divisão aleatória")
    R["aleatorio_dow_original"] = rodar(d, dados.dividir_aleatorio, corrigir=False, sementes=sementes)
    print("2/4 dia da semana corrigido, divisão aleatória")
    R["aleatorio_dow_corrigido"] = rodar(d, dados.dividir_aleatorio, corrigir=True, sementes=sementes)
    print("3/4 validação temporal (treino 2007-2009, teste 2010)")
    R["temporal"] = rodar(d, dados.dividir_temporal, corrigir=True, sementes=sementes)
    print("4/4 ablação com Number_of_Vehicles")
    R["com_n_veiculos"] = rodar(d, dados.dividir_aleatorio, num=dados.NUM + ["Number_of_Vehicles"],
                                corrigir=True, sementes=sementes, com_lr=False)

    Path("docs").mkdir(exist_ok=True)
    json.dump(R, open("docs/resultados_experimentos.json", "w"), indent=2)

    nomes = {"aleatorio_dow_original": "Divisão aleatória, dia da semana original",
             "aleatorio_dow_corrigido": "Divisão aleatória, dia da semana corrigido",
             "temporal": "Temporal: treino 2007-2009, teste 2010 (dia corrigido)",
             "com_n_veiculos": "Aleatória + Number_of_Vehicles (ablação)"}
    linhas = [f"# Resultados dos experimentos extras\n",
              f"Rede 2 com {len(sementes)} sementes ({sementes[0]} a {sementes[-1]}): média ± desvio no conjunto de teste.\n",
              "| Cenário | Modelo | PR-AUC | ROC-AUC | Captura top 10% | Captura top 20% |",
              "|---|---|---|---|---|---|"]
    for k, r in R.items():
        linhas.append(f"| {nomes[k]} | Rede neural | {resumo(r['rede'], 'PR-AUC')} | "
                      f"{resumo(r['rede'], 'ROC-AUC')} | {resumo(r['rede'], 'Captura top 10%')} | "
                      f"{resumo(r['rede'], 'Captura top 20%')} |")
        if "lr" in r:
            linhas.append(f"| {nomes[k]} | Regressão logística | {r['lr']['PR-AUC']:.4f} | "
                          f"{r['lr']['ROC-AUC']:.4f} | {r['lr']['Captura top 10%']:.4f} | "
                          f"{r['lr']['Captura top 20%']:.4f} |")
    linhas.append(f"\nTamanhos (treino/val/teste) temporal: {R['temporal']['n_treino']}/"
                  f"{R['temporal']['n_val']}/{R['temporal']['n_teste']}, "
                  f"taxa de graves no teste temporal: {R['temporal']['taxa_teste']:.4f}")
    linhas.append(f"\nTempo total: {(time.time() - t0) / 60:.1f} min")
    Path("docs/resultados_experimentos.md").write_text("\n".join(linhas), encoding="utf-8")
    print("\n".join(linhas))


if __name__ == "__main__":
    main()
