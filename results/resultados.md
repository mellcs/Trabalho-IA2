# Resultados no conjunto de teste

Teste: 4647 comentários (34.2% reclamações). Redes: média ± desvio de 3 sementes. Limiar 0,5.

| Modelo | F1 (reclamação) | Recall | Precisão | Acurácia | PR-AUC |
|---|---|---|---|---|---|
| Rede BiLSTM | 0.897 ± 0.001 | 0.927 ± 0.004 | 0.868 ± 0.002 | 0.927 ± 0.001 | 0.936 ± 0.003 |
| Rede CNN 1D | 0.895 ± 0.003 | 0.931 ± 0.004 | 0.862 ± 0.008 | 0.925 ± 0.002 | 0.934 ± 0.005 |
| Baseline (TF-IDF + LR) | 0.898 | 0.941 | 0.860 | 0.927 | 0.934 |

Arquitetura escolhida pela validação: **bilstm** (F1 médio na validação: bilstm 0.8935, cnn 0.8892).

Faixas de prioridade (limiares da validação: alta ≥ 0.974, baixa < 0.184):

| Prioridade | % dos comentários | % que são reclamação | % de todas as reclamações |
|---|---|---|---|
| Alta | 20.6% | 94.7% | 56.9% |
| Média | 20.5% | 65.6% | 39.3% |
| Baixa | 58.9% | 2.2% | 3.8% |

Tempo total: 15.5 min