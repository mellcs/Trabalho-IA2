"""Gera os dados que o app usa sem precisar do dataset: caixa de entrada de exemplo e painel.

Uso:  python -m src.preparar_demo     (depois de python -m src.treinar)

- models/caixa_demo.json: mensagens para a caixa de entrada de demonstração
    * mensagens escritas por nós imitando WhatsApp, e-mail, SAC e Reclame Aqui (sem nota);
    * avaliações reais da Olist do CONJUNTO DE TESTE (nunca vistas no treino) de mar/2018,
      mês em que as reclamações de entrega dispararam.
- models/painel.json: histórico mês a mês do conjunto de teste inteiro (reclamações previstas
  pela rede, por motivo) e os alertas de alta brusca.
"""
import json
from pathlib import Path

import pandas as pd

from src import dados, motivos
from src.inferencia import Analisador

ALERTA_MIN = 30      # só alerta se o motivo tiver pelo menos 30 reclamações no mês
ALERTA_ALTA = 0.5    # ... e subir pelo menos 50% em relação ao mês anterior


def main(pasta="data", saida="models"):
    A = Analisador(saida)
    d, _ = dados.limpar(dados.carregar(pasta))
    _, _, te = dados.dividir(d)
    av = dados.carregar(pasta)[["review_id", "review_creation_date"]].drop_duplicates("review_id")
    te = te.merge(av, on="review_id", how="left")
    te["data"] = pd.to_datetime(te["review_creation_date"])
    te["p"] = A.probabilidades(te["texto"].tolist())
    te["reclamacao"] = te["p"] >= A.cfg["limiar"]
    te["motivo"] = [motivos.motivo(t) if r else "—" for t, r in zip(te["texto"], te["reclamacao"])]
    te["prioridade"] = [A.prioridade(p) for p in te["p"]]

    # ---------- painel: mês a mês (meses com pelo menos 50 avaliações no teste)
    te["mes"] = te["data"].dt.to_period("M").astype(str)
    cont = te.groupby("mes").size()
    meses = [m for m in cont.index if cont[m] >= 50]
    nomes_mot = [m for m in motivos.MOTIVOS] + [motivos.OUTROS]
    serie = []
    for m in meses:
        g = te[te["mes"] == m]
        serie.append({"mes": m, "mensagens": int(len(g)), "reclamacoes": int(g["reclamacao"].sum()),
                      "alta": int((g["prioridade"] == "Alta").sum()),
                      "por_motivo": {k: int((g["motivo"] == k).sum()) for k in nomes_mot}})
    alertas = []
    for a, b in zip(serie[:-1], serie[1:]):
        for k in nomes_mot:
            x, y = a["por_motivo"][k], b["por_motivo"][k]
            if y >= ALERTA_MIN and x > 0 and (y - x) / x >= ALERTA_ALTA:
                alertas.append({"mes": b["mes"], "motivo": k, "antes": x, "depois": y,
                                "alta": round((y - x) / x, 3)})
    rec = te[te["reclamacao"]]
    painel = {
        "periodo": [meses[0], meses[-1]], "mensagens": int(len(te)), "reclamacoes": int(te["reclamacao"].sum()),
        "faixas": {f: int((te["prioridade"] == f).sum()) for f in ["Alta", "Média", "Baixa"]},
        "motivos": {k: int((rec["motivo"] == k).sum()) for k in nomes_mot},
        "serie": serie, "alertas": alertas, "faixas_teste": A.cfg["faixas_teste"],
        "metricas_teste": A.cfg["metricas_teste"],
    }
    json.dump(painel, open(Path(saida) / "painel.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---------- caixa de entrada de demonstração
    canais = pd.read_csv("docs/exemplo_comentarios.csv")
    nomes = ["Carla M.", "João P.", "Ana L.", "Rafael S.", "Bruna T.", "Diego R.", "Lucas F.", "Paula V.",
             "Marcos A.", "Júlia C.", "Pedro H.", "Sofia B.", "Thiago N.", "Larissa G.", "Felipe O.",
             "Camila R.", "Bruno K.", "Renata D."]
    caixa = []
    for i, r in canais.iterrows():
        caixa.append({"id": f"c{i}", "canal": r["canal"], "cliente": nomes[i % len(nomes)],
                      "texto": r["comentario"], "nota": None, "data": "hoje"})
    mar = te[te["mes"] == "2018-03"].sample(n=min(110, (te["mes"] == "2018-03").sum()), random_state=7)
    for _, r in mar.iterrows():
        caixa.append({"id": r["review_id"][:10], "canal": "Avaliação", "cliente": f"Pedido #{r['order_id'][:6]}",
                      "texto": r["texto"], "nota": int(r["review_score"]), "data": r["data"].strftime("%d/%m/%Y")})
    json.dump(caixa, open(Path(saida) / "caixa_demo.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"painel: {len(serie)} meses, {len(alertas)} alertas; caixa: {len(caixa)} mensagens")
    for a in alertas:
        print("  alerta", a)


if __name__ == "__main__":
    main()
