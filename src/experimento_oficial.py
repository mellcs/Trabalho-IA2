"""Quanto os dados oficiais (STATS19) melhoram o modelo?

Compara, com a MESMA rede, a MESMA divisão e o MESMO conjunto de teste:
  1. só as colunas do dataset do Kaggle (modelo do notebook);
  2. + limite de velocidade e horário;
  3. + todas as variáveis oficiais do local (cruzamento, classe da via, faixa de pedestre...);
e repete o cenário 3 na validação temporal (treino 2007-2009, teste 2010).

Uso:  python -m src.experimento_oficial --csv data/accident_data.csv --oficial data/Accidents0515.csv
Saída: docs/resultados_dados_oficiais.md e .json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from src import dados
from src.experimentos import regressao_logistica, resumo
from src.modelo import metricas, prever, treinar


def rodar(d, divisao, cats, num, sementes):
    train, val, test = divisao(d)
    prep = dados.Preparador(num=num, cats=cats).ajustar(train)
    A, B, C = prep.transformar(train), prep.transformar(val), prep.transformar(test)
    out = {"n_teste": len(test), "rede": []}
    for s in sementes:
        m, _ = treinar(prep.vocab, *A, *B, semente=s, verbose=False)
        out["rede"].append(metricas(C[2], prever(m, C[0], C[1])))
    out["lr"] = metricas(C[2], regressao_logistica(A[0], A[1], A[2], C[0], C[1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/accident_data.csv")
    ap.add_argument("--oficial", default="data/Accidents0515.csv")
    ap.add_argument("--sementes", type=int, default=3)
    a = ap.parse_args()
    t0 = time.time()
    d = dados.features_calendario(dados.limpar(dados.carregar(a.csv)))
    d = dados.juntar_oficial(d, a.oficial)
    casou = float(d["casou_oficial"].mean())
    sem = tuple(range(42, 42 + a.sementes))
    vel_hora_c = dados.CATS + ["Speed_limit"]
    tudo_c = dados.CATS + dados.CATS_OFICIAIS
    num_of = dados.NUM + dados.NUM_OFICIAIS
    cen = {
        "Só o dataset do Kaggle (notebook)": (dados.dividir_aleatorio, dados.CATS, dados.NUM),
        "+ limite de velocidade e horário": (dados.dividir_aleatorio, vel_hora_c, num_of),
        "+ todas as variáveis oficiais do local (painel)": (dados.dividir_aleatorio, tudo_c, num_of),
        "Temporal (treino 2007-2009, teste 2010), variáveis oficiais": (dados.dividir_temporal, tudo_c, num_of),
    }
    R = {}
    for nome, (div, c, n) in cen.items():
        print(nome, flush=True)
        R[nome] = rodar(d, div, c, n, sem)
    Path("docs").mkdir(exist_ok=True)
    json.dump({"linhas_casadas": casou, "resultados": R},
              open("docs/resultados_dados_oficiais.json", "w"), indent=2, ensure_ascii=False)
    L = ["# Ganho com os dados oficiais (STATS19)\n",
         f"Junção com `Accidents0515.csv`: {casou:.2%} das linhas casadas (data real, coordenadas, gravidade, nº de veículos e de vítimas).",
         f"Rede com {len(sem)} sementes (média ± desvio) e regressão logística; mesmo conjunto de teste.\n",
         "| Cenário | Modelo | PR-AUC | ROC-AUC | Captura top 10% | Captura top 20% |", "|---|---|---|---|---|---|"]
    for nome, r in R.items():
        L.append(f"| {nome} | Rede neural | {resumo(r['rede'], 'PR-AUC')} | {resumo(r['rede'], 'ROC-AUC')} | "
                 f"{resumo(r['rede'], 'Captura top 10%')} | {resumo(r['rede'], 'Captura top 20%')} |")
        L.append(f"| {nome} | Regressão logística | {r['lr']['PR-AUC']:.4f} | {r['lr']['ROC-AUC']:.4f} | "
                 f"{r['lr']['Captura top 10%']:.4f} | {r['lr']['Captura top 20%']:.4f} |")
    L.append("\nVariáveis oficiais usadas: " + ", ".join(dados.CATS_OFICIAIS) + "; horário (seno/cosseno).")
    L.append("Não entram: nº de vítimas, nº de veículos, dados dos veículos e das vítimas, presença da polícia.")
    L.append(f"\nTempo total: {(time.time() - t0) / 60:.1f} min")
    Path("docs/resultados_dados_oficiais.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
