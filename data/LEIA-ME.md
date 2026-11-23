# Dados

Os CSVs não estão no repositório. Para baixar os dois arquivos usados (sem login no Kaggle):

```bash
python -m src.baixar_dados
```

Isso salva aqui:

- `olist_order_reviews_dataset.csv`: 100 mil avaliações (nota de 1 a 5 e comentário em texto)
- `olist_orders_dataset.csv`: pedidos (status e datas de entrega), usado para medir o efeito do atraso e conferir os motivos

Fonte: *Brazilian E-Commerce Public Dataset by Olist* (Kaggle `olistbr/brazilian-ecommerce`, licença CC BY-NC-SA 4.0), baixado do espelho público no GitHub `spdrio/Brazilian-E-Commerce-Public-Dataset-by-Olist`.
