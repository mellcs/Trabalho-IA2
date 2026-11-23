"""Carga, limpeza e divisão dos dados de avaliações da Olist.

Etapas (cada uma registra quantas linhas sobram, para mostrar no README e no notebook):
1. avaliações com comentário em texto;
2. remoção de review_id repetido;
3. remoção da nota 3 (neutra): no teste com 3 classes a classe neutra ficou com F1 muito
   baixo, e o objetivo do produto é separar reclamação de não reclamação;
4. remoção de textos duplicados (ex.: "ótimo" aparece centenas de vezes). Sem isso o mesmo
   texto cairia no treino e no teste ao mesmo tempo (vazamento) e inflaria as métricas.
   Textos iguais com rótulos diferentes são descartados.

Alvo: negativo = 1 se a nota for 1 ou 2; negativo = 0 se a nota for 4 ou 5.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.texto import normalizar

ARQ_AVALIACOES = "olist_order_reviews_dataset.csv"
ARQ_PEDIDOS = "olist_orders_dataset.csv"


def carregar(pasta="data"):
    return pd.read_csv(Path(pasta) / ARQ_AVALIACOES)


def limpar(av):
    """Devolve (base limpa, tabela com a contagem de cada etapa)."""
    etapas = [("Avaliações no arquivo", len(av))]
    txt = av["review_comment_message"].fillna("").str.strip()
    d = av[txt != ""].copy()
    d["texto"] = d["review_comment_message"].str.strip()
    etapas.append(("Com comentário em texto", len(d)))

    d = d.drop_duplicates("review_id")
    etapas.append(("Sem review_id repetido", len(d)))

    d = d[d["review_score"] != 3].copy()
    etapas.append(("Sem a nota 3 (neutra)", len(d)))

    d["negativo"] = (d["review_score"] <= 2).astype(int)
    # texto normalizado (minúsculas, sem acento, espaços únicos) para achar duplicados
    d["chave"] = d["texto"].map(normalizar)
    d = d[d["chave"] != ""]
    conflito = d.groupby("chave")["negativo"].transform("nunique") > 1
    d = d[~conflito]
    etapas.append(("Sem textos com rótulos conflitantes", len(d)))
    d = d.drop_duplicates("chave").reset_index(drop=True)
    etapas.append(("Sem textos duplicados (base final)", len(d)))

    tab = pd.DataFrame(etapas, columns=["etapa", "linhas"])
    return d[["review_id", "order_id", "review_score", "texto", "negativo"]], tab


def dividir(d, semente=42):
    """70% treino, 15% validação, 15% teste, estratificado pelo alvo.

    A divisão é feita depois de remover os duplicados, então nenhum texto aparece
    em duas partes. O teste só é usado uma vez, no final.
    """
    treino, resto = train_test_split(d, test_size=0.30, stratify=d["negativo"], random_state=semente)
    val, teste = train_test_split(resto, test_size=0.50, stratify=resto["negativo"], random_state=semente)
    return treino.reset_index(drop=True), val.reset_index(drop=True), teste.reset_index(drop=True)


def avaliacoes_com_pedidos(pasta="data"):
    """Junta TODAS as avaliações (com ou sem texto) com a situação da entrega do pedido.

    Serve para medir o problema: quanto a reclamação cresce quando a entrega atrasa.
    """
    av = carregar(pasta)
    ped = pd.read_csv(Path(pasta) / ARQ_PEDIDOS,
                      parse_dates=["order_delivered_customer_date", "order_estimated_delivery_date"])
    m = av.merge(ped[["order_id", "order_status", "order_delivered_customer_date",
                      "order_estimated_delivery_date"]], on="order_id", how="left")
    entregue = (m["order_status"] == "delivered") & m["order_delivered_customer_date"].notna()
    atrasou = m["order_delivered_customer_date"].dt.normalize() > m["order_estimated_delivery_date"]
    m["entrega"] = "Não entregue"
    m.loc[entregue & ~atrasou, "entrega"] = "Entregue no prazo"
    m.loc[entregue & atrasou, "entrega"] = "Entregue com atraso"
    m["negativo"] = (m["review_score"] <= 2).astype(int)
    return m
