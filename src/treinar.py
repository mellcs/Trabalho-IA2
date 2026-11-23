"""Pipeline completo: dados -> EDA -> baseline -> redes -> avaliação -> artefatos do app.

Uso:  python -m src.treinar            (~15 min em CPU; o app não precisa disso, já usa models/)
      python -m src.treinar --rapido   (1 semente, ~5 min)

Saídas:
  models/   rede.pt, vocab.json, config.json, baseline.joblib  (usados pelo app)
  results/  gráficos .png, metricas.json, resultados.md, erros.md, motivos.md
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

from src import dados, motivos, texto
from src.graficos import AZUL, CINZA, COR_CLASSE, COR_MODELO, LARANJA, TINTA2, plt, salvar
from src.modelo import metricas, prever, treinar

R, M = Path("results"), Path("models")
# só para os gráficos de palavras frequentes (o modelo usa todas as palavras, inclusive "não")
STOP = set("""a o e de da do das dos que em um uma para com por no na nos nas os as se ao foi
me meu minha mas muito mais ja eu esta isso ate pois so tem ter como sem pelo pela ainda
dia dias q pra pro voce lo la ele ela era bem""".split())


def baseline_tfidf():
    """TF-IDF com unigramas e bigramas + regressão logística com classes balanceadas."""
    tf = TfidfVectorizer(preprocessor=texto.normalizar, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    lr = LogisticRegression(C=4, class_weight="balanced", max_iter=2000)
    return tf, lr


def fig_eda(av, d):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    tem = av["review_comment_message"].fillna("").str.strip() != ""
    cont = pd.DataFrame({"Com comentário": av[tem]["review_score"].value_counts(),
                         "Sem comentário": av[~tem]["review_score"].value_counts()}).sort_index()
    x = np.arange(5)
    ax[0].bar(x - 0.2, cont["Com comentário"], 0.38, color=AZUL, label="com comentário")
    ax[0].bar(x + 0.2, cont["Sem comentário"], 0.38, color=CINZA, label="sem comentário")
    ax[0].set_xticks(x, [f"nota {i}" for i in range(1, 6)])
    ax[0].set_title("Notas das 100 mil avaliações")
    ax[0].legend()
    p = d["negativo"].value_counts().sort_index()
    ax[1].bar(["Não reclamação\n(nota 4-5)", "Reclamação\n(nota 1-2)"], p.values,
              color=[COR_CLASSE[0], COR_CLASSE[1]], width=0.55)
    for i, v in enumerate(p.values):
        ax[1].text(i, v, f"{v:,}\n({v / p.sum():.0%})".replace(",", "."), ha="center", va="bottom", color=TINTA2)
    ax[1].set_ylim(0, p.max() * 1.25)
    ax[1].set_title("Base final para o modelo")
    n = d["texto"].map(lambda s: len(texto.tokenizar(s)))
    for k, nome in [(0, "não reclamação"), (1, "reclamação")]:
        ax[2].hist(n[d["negativo"] == k], bins=range(0, 50, 2), alpha=0.75, color=COR_CLASSE[k], label=nome)
    ax[2].set_xlabel("palavras no comentário")
    ax[2].set_title("Tamanho dos comentários")
    ax[2].legend()
    salvar(fig, R / "eda.png")
    return {"mediana_palavras_reclamacao": float(n[d["negativo"] == 1].median()),
            "mediana_palavras_nao_reclamacao": float(n[d["negativo"] == 0].median())}


def fig_palavras(d, k=15):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for j, (cls, nome) in enumerate([(1, "Reclamação"), (0, "Não reclamação")]):
        c = Counter(t for s in d[d["negativo"] == cls]["texto"] for t in set(texto.tokenizar(s)) if t not in STOP)
        top = c.most_common(k)[::-1]
        ax[j].barh([w for w, _ in top], [v / (d["negativo"] == cls).sum() * 100 for _, v in top],
                   color=COR_CLASSE[cls], height=0.6)
        ax[j].set_title(f"{nome}: palavras mais frequentes")
        ax[j].set_xlabel("% dos comentários da classe que usam a palavra")
    salvar(fig, R / "palavras_frequentes.png")


def fig_problema(m):
    t = m.groupby("entrega")["negativo"].agg(["mean", "size"]).reindex(
        ["Entregue no prazo", "Entregue com atraso", "Não entregue"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(t.index, t["mean"] * 100, color=[AZUL, LARANJA, LARANJA], width=0.55)
    for i, (v, n) in enumerate(zip(t["mean"], t["size"])):
        ax.text(i, v * 100 + 1.5, f"{v:.1%}\n({n:,} pedidos)".replace(",", "."), ha="center", color=TINTA2)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% de avaliações com nota 1 ou 2")
    ax.set_title("Quando a entrega falha, a reclamação dispara")
    salvar(fig, R / "atraso_reclamacao.png")
    return {k: {"taxa_reclamacao": float(r["mean"]), "pedidos": int(r["size"])} for k, r in t.iterrows()}


def tres_classes(av):
    """Teste rápido com a nota 3 como classe neutra (justifica deixá-la de fora)."""
    d = av[av["review_comment_message"].fillna("").str.strip() != ""].drop_duplicates("review_id").copy()
    d["chave"] = d["review_comment_message"].map(texto.normalizar)
    d["classe"] = pd.cut(d["review_score"], [0, 2, 3, 5], labels=["negativo", "neutro", "positivo"])
    d = d[d.groupby("chave")["classe"].transform("nunique") == 1].drop_duplicates("chave")
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(d, test_size=0.2, stratify=d["classe"], random_state=42)
    tf, lr = baseline_tfidf()
    lr.fit(tf.fit_transform(tr["review_comment_message"]), tr["classe"])
    rep = classification_report(te["classe"], lr.predict(tf.transform(te["review_comment_message"])),
                                output_dict=True)
    return {c: round(rep[c]["f1-score"], 3) for c in ["negativo", "neutro", "positivo"]}


def limiares_prioridade(y, p):
    """Faixas de prioridade definidas na VALIDAÇÃO.

    Alta  = probabilidade a partir da qual pelo menos 95% são reclamações de verdade.
    Baixa = probabilidade abaixo da qual no máximo 2% são reclamações.
    Média = o que fica no meio (vale uma leitura humana).
    """
    ordem = np.argsort(-p)
    ps, ys = p[ordem], y[ordem]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    ok = np.where(prec >= 0.95)[0]
    alta = float(ps[ok.max()]) if len(ok) else 0.9
    ordem = np.argsort(p)
    ps, ys = p[ordem], y[ordem]
    taxa = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    ok = np.where(taxa <= 0.02)[0]
    baixa = float(ps[ok.max()]) if len(ok) else 0.1
    return {"alta": round(max(alta, 0.5), 3), "baixa": round(min(baixa, 0.5), 3)}


def faixa(p, lim):
    return np.where(p >= lim["alta"], "Alta", np.where(p < lim["baixa"], "Baixa", "Média"))


def secao_motivos(te, ped, out):
    """Motivo das reclamações previstas no teste + conferência com a situação real da entrega."""
    neg = te[te["p_rede"] >= 0.5].merge(ped[["review_id", "entrega"]].drop_duplicates("review_id"),
                                       on="review_id", how="left")
    neg["motivo"] = neg["texto"].map(motivos.motivo)
    dist = neg["motivo"].value_counts()
    falha = neg.groupby("motivo")["entrega"].apply(lambda s: (s != "Entregue no prazo").mean())
    out["motivos"] = {m: {"comentarios": int(dist[m]), "fracao": float(dist[m] / len(neg)),
                          "pedido_atrasado_ou_nao_entregue": float(falha[m])} for m in dist.index}
    out["entrega_falhou_entre_reclamacoes_previstas"] = float((neg["entrega"] != "Entregue no prazo").mean())
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    dd = dist[::-1]
    ax.barh(dd.index, dd.values / len(neg) * 100, color=[CINZA if i == motivos.OUTROS else LARANJA for i in dd.index], height=0.6)
    for i, v in enumerate(dd.values):
        ax.text(v / len(neg) * 100 + 0.5, i, f"{v / len(neg):.0%}", va="center", color=TINTA2)
    ax.set_xlabel("% das reclamações detectadas no teste")
    ax.set_title("Motivo provável das reclamações")
    salvar(fig, R / "motivos.png")
    Mo = ["# Motivos das reclamações (conjunto de teste)\n",
          f"Comentários que a rede classificou como reclamação: {len(neg)}.\n",
          "Conferência independente: o texto não sabe se o pedido atrasou, mas a tabela de pedidos sabe.",
          "Se a regra de \"Entrega / atraso\" funciona, esses pedidos devem ter muito mais atraso ou não entrega.\n",
          "| Motivo | Comentários | % | Pedido atrasado ou não entregue |", "|---|---|---|---|"]
    for m_, r in out["motivos"].items():
        Mo.append(f"| {m_} | {r['comentarios']} | {r['fracao']:.1%} | {r['pedido_atrasado_ou_nao_entregue']:.1%} |")
    Mo.append(f"\nEntre todas as reclamações detectadas: {out['entrega_falhou_entre_reclamacoes_previstas']:.1%} "
              f"com entrega atrasada ou não feita.")
    (R / "motivos.md").write_text("\n".join(Mo), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default="data")
    ap.add_argument("--rapido", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    R.mkdir(exist_ok=True)
    M.mkdir(exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    sementes = [42] if a.rapido else [42, 43, 44]
    out = {}

    # 1. Dados
    av = dados.carregar(a.pasta)
    d, etapas = dados.limpar(av)
    etapas.to_csv(R / "etapas_limpeza.csv", index=False)
    print(etapas.to_string(index=False))
    out["etapas"] = etapas.to_dict(orient="records")
    out["taxa_reclamacao_base"] = float(d["negativo"].mean())

    # 2. EDA e o problema em números
    out["eda"] = fig_eda(av, d)
    fig_palavras(d)
    ped = dados.avaliacoes_com_pedidos(a.pasta)
    out["entrega_x_reclamacao"] = fig_problema(ped)
    out["tres_classes_f1_baseline"] = tres_classes(av)
    print("3 classes (baseline):", out["tres_classes_f1_baseline"])

    # 3. Divisão
    tr, va, te = dados.dividir(d)
    y_tr, y_va, y_te = (x["negativo"].to_numpy() for x in (tr, va, te))
    out["tamanhos"] = {"treino": len(tr), "validacao": len(va), "teste": len(te)}

    # 4. Baseline
    tf, lr = baseline_tfidf()
    A = tf.fit_transform(tr["texto"])
    lr.fit(A, y_tr)
    pb_va, pb_te = lr.predict_proba(tf.transform(va["texto"]))[:, 1], lr.predict_proba(tf.transform(te["texto"]))[:, 1]
    joblib.dump({"tfidf": tf, "lr": lr}, M / "baseline.joblib")

    # 5. Redes: duas arquiteturas x várias sementes; escolha pela validação
    voc = texto.Vocabulario().ajustar(tr["texto"])
    X_tr, X_va, X_te = (voc.codificar(x["texto"].tolist()) for x in (tr, va, te))
    out["vocabulario"] = len(voc)
    corridas = {}
    for arq in ["bilstm", "cnn"]:
        for s in sementes:
            mod, hist = treinar(arq, len(voc), X_tr, y_tr, X_va, y_va, semente=s)
            corridas[(arq, s)] = (mod, hist, prever(mod, X_va), prever(mod, X_te))
    f1_val = {arq: np.mean([metricas(y_va, corridas[(arq, s)][2])["F1 (reclamação)"] for s in sementes])
              for arq in ["bilstm", "cnn"]}
    escolhida = max(f1_val, key=f1_val.get)
    out["f1_validacao_por_arquitetura"] = f1_val
    out["arquitetura_escolhida"] = escolhida
    print("F1 validação:", f1_val, "-> escolhida:", escolhida)
    rede, hist, p_va, p_te = corridas[(escolhida, 42)]
    out["parametros_rede"] = int(sum(p.numel() for p in rede.parameters()))

    # 6. Prioridade (definida na validação)
    lim = limiares_prioridade(y_va, p_va)
    out["limiares_prioridade"] = lim

    # 7. TESTE (aberto uma vez)
    tabela = {}
    for arq in ["bilstm", "cnn"]:
        ms = [metricas(y_te, corridas[(arq, s)][3]) for s in sementes]
        tabela[f"Rede {'BiLSTM' if arq == 'bilstm' else 'CNN 1D'}"] = {
            k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms]))) for k in ms[0]}
    tabela["Baseline (TF-IDF + LR)"] = {k: (float(v), 0.0) for k, v in metricas(y_te, pb_te).items()}
    out["teste"] = tabela
    out["teste_modelo_final"] = metricas(y_te, p_te)
    out["teste_baseline"] = metricas(y_te, pb_te)
    out["matriz_confusao_rede"] = confusion_matrix(y_te, (p_te >= 0.5).astype(int)).tolist()
    out["matriz_confusao_baseline"] = confusion_matrix(y_te, (pb_te >= 0.5).astype(int)).tolist()
    fx = faixa(p_te, lim)
    out["faixas_teste"] = {f: {"comentarios": int((fx == f).sum()), "fracao": float((fx == f).mean()),
                               "taxa_reclamacao": float(y_te[fx == f].mean()),
                               "fracao_das_reclamacoes": float(y_te[fx == f].sum() / y_te.sum())}
                           for f in ["Alta", "Média", "Baixa"]}
    lidos = p_te >= 0.5
    out["fila"] = {"fracao_lida": float(lidos.mean()), "reclamacoes_encontradas": float(y_te[lidos].sum() / y_te.sum())}

    # gráficos de avaliação
    h = pd.DataFrame(hist)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(h["epoca"], h["perda_treino"], "o-", color=AZUL, label="treino")
    ax[0].plot(h["epoca"], h["perda_val"], "s--", color=LARANJA, label="validação")
    ax[0].set_title("Perda (BCE com peso de classe)")
    ax[0].set_xlabel("época")
    ax[0].legend()
    ax[1].plot(h["epoca"], h["f1_val"], "o-", color=AZUL, label="F1 (validação)")
    ax[1].plot(h["epoca"], h["recall_val"], "s--", color=CINZA, label="recall (validação)")
    best = int(h.loc[h["f1_val"].idxmax(), "epoca"])
    ax[1].axvline(best, color=TINTA2, lw=1, ls=":")
    ax[1].text(best, h["f1_val"].min(), f" melhor época ({best})", color=TINTA2)
    ax[1].set_title("Métricas da classe reclamação")
    ax[1].set_xlabel("época")
    ax[1].legend()
    salvar(fig, R / "curvas_treino.png")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for j, (nome, mat) in enumerate([("Rede neural", out["matriz_confusao_rede"]),
                                     ("Baseline (TF-IDF + LR)", out["matriz_confusao_baseline"])]):
        mat = np.array(mat)
        ax[j].imshow(mat, cmap="Blues")
        ax[j].grid(False)
        for (i, k), v in np.ndenumerate(mat):
            ax[j].text(k, i, f"{v:,}".replace(",", "."), ha="center", va="center",
                       color="white" if v > mat.max() / 2 else "#0b0b0b", fontsize=13)
        ax[j].set_xticks([0, 1], ["previsto: não", "previsto: reclamação"])
        ax[j].set_yticks([0, 1], ["real: não", "real: reclamação"])
        ax[j].set_title(nome)
    salvar(fig, R / "matriz_confusao.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for nome, p in [("Rede neural", p_te), ("Baseline (TF-IDF + LR)", pb_te)]:
        pr, rc, _ = precision_recall_curve(y_te, p)
        ax.plot(rc, pr, color=COR_MODELO[nome], label=f"{nome} (PR-AUC {metricas(y_te, p)['PR-AUC']:.3f})")
    ax.axhline(y_te.mean(), color=CINZA, ls=":", lw=1.5, label="acaso")
    ax.set_xlabel("recall (reclamações encontradas)")
    ax.set_ylabel("precisão")
    ax.set_title("Curva precisão-recall (teste)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    salvar(fig, R / "curva_pr.png")

    # 8. Análise de erros
    te = te.assign(p_rede=p_te, p_base=pb_te, prioridade=fx, motivo=te["texto"].map(motivos.motivo))
    fp = te[(te["negativo"] == 0) & (te["p_rede"] >= 0.5)].nlargest(10, "p_rede")
    fn = te[(te["negativo"] == 1) & (te["p_rede"] < 0.5)].nsmallest(10, "p_rede")
    so_rede = te[((te["p_rede"] >= 0.5) == te["negativo"].astype(bool)) & ((te["p_base"] >= 0.5) != te["negativo"].astype(bool))]
    so_base = te[((te["p_base"] >= 0.5) == te["negativo"].astype(bool)) & ((te["p_rede"] >= 0.5) != te["negativo"].astype(bool))]
    out["discordancias"] = {"rede_acerta_baseline_erra": len(so_rede), "baseline_acerta_rede_erra": len(so_base)}

    def linhas(df):
        return [f"| {r.p_rede:.2f} | {r.p_base:.2f} | {r.review_score} | {r.texto.replace('|', '/').replace(chr(10), ' ')} |"
                for r in df.itertuples()]
    cab = ["| P(rede) | P(baseline) | Nota real | Comentário |", "|---|---|---|---|"]
    E = ["# Análise de erros (conjunto de teste)\n",
         f"Rede {escolhida}, limiar 0,5. P = probabilidade de ser reclamação.\n",
         "## Falsos positivos mais confiantes (o modelo disse reclamação, a nota foi 4 ou 5)\n", *cab, *linhas(fp),
         "\n## Falsos negativos mais confiantes (o modelo disse não reclamação, a nota foi 1 ou 2)\n", *cab, *linhas(fn),
         f"\n## Onde rede e baseline discordam\n\nA rede acerta e o baseline erra em {len(so_rede)} comentários; "
         f"o contrário acontece em {len(so_base)}.\n", "### Exemplos em que a rede acertou e o baseline errou\n",
         *cab, *linhas(so_rede.head(10)), "\n### Exemplos em que o baseline acertou e a rede errou\n", *cab,
         *linhas(so_base.head(10))]
    (R / "erros.md").write_text("\n".join(E), encoding="utf-8")

    # 9. Motivos das reclamações previstas + conferência com a situação real da entrega
    secao_motivos(te, ped, out)

    # 10. Artefatos do app
    torch.save(rede.state_dict(), M / "rede.pt")
    json.dump(voc.para_dict(), open(M / "vocab.json", "w", encoding="utf-8"), ensure_ascii=False)
    cfg = {"arquitetura": escolhida, "vocab": len(voc), "limiar": 0.5, "limiares_prioridade": lim,
           "metricas_teste": out["teste_modelo_final"], "faixas_teste": out["faixas_teste"]}
    json.dump(cfg, open(M / "config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # 11. Resumo
    out["historico_treino"] = hist
    out["tempo_min"] = (time.time() - t0) / 60
    json.dump(out, open(R / "metricas.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=float)
    L = ["# Resultados no conjunto de teste\n",
         f"Teste: {len(te)} comentários ({y_te.mean():.1%} reclamações). Redes: média ± desvio de {len(sementes)} sementes. Limiar 0,5.\n",
         "| Modelo | F1 (reclamação) | Recall | Precisão | Acurácia | PR-AUC |", "|---|---|---|---|---|---|"]
    for nome, mm in tabela.items():
        f = lambda k: f"{mm[k][0]:.3f}" + (f" ± {mm[k][1]:.3f}" if mm[k][1] else "")
        L.append(f"| {nome} | {f('F1 (reclamação)')} | {f('Recall (reclamação)')} | {f('Precisão (reclamação)')} | {f('Acurácia')} | {f('PR-AUC')} |")
    L.append(f"\nArquitetura escolhida pela validação: **{escolhida}** (F1 médio na validação: "
             + ", ".join(f"{k} {v:.4f}" for k, v in f1_val.items()) + ").")
    L.append(f"\nFaixas de prioridade (limiares da validação: alta ≥ {lim['alta']}, baixa < {lim['baixa']}):\n")
    L.append("| Prioridade | % dos comentários | % que são reclamação | % de todas as reclamações |")
    L.append("|---|---|---|---|")
    for f_, r in out["faixas_teste"].items():
        L.append(f"| {f_} | {r['fracao']:.1%} | {r['taxa_reclamacao']:.1%} | {r['fracao_das_reclamacoes']:.1%} |")
    L.append(f"\nTempo total: {out['tempo_min']:.1f} min")
    (R / "resultados.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
