"""Baixa os dois arquivos da Olist usados no projeto para data/ (sem precisar de login no Kaggle).

Uso:  python -m src.baixar_dados

Fonte: espelho público no GitHub do "Brazilian E-Commerce Public Dataset by Olist"
(original no Kaggle: olistbr/brazilian-ecommerce, licença CC BY-NC-SA 4.0).
"""
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/spdrio/Brazilian-E-Commerce-Public-Dataset-by-Olist/master/files/"
ARQUIVOS = ["olist_order_reviews_dataset.csv", "olist_orders_dataset.csv"]


def main(pasta="data"):
    p = Path(pasta)
    p.mkdir(exist_ok=True)
    for a in ARQUIVOS:
        destino = p / a
        if destino.exists():
            print("já existe:", destino)
            continue
        print("baixando", a, "...")
        urllib.request.urlretrieve(BASE + a, destino)
        print("ok:", destino, f"({destino.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
