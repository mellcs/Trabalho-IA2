# Resultados dos experimentos extras

Rede 2 com 5 sementes (42 a 46): média ± desvio no conjunto de teste.

| Cenário | Modelo | PR-AUC | ROC-AUC | Captura top 10% | Captura top 20% |
|---|---|---|---|---|---|
| Divisão aleatória, dia da semana original | Rede neural | 0.2155 ± 0.0012 | 0.6161 ± 0.0009 | 0.1791 ± 0.0026 | 0.3168 ± 0.0012 |
| Divisão aleatória, dia da semana original | Regressão logística | 0.2140 | 0.6150 | 0.1788 | 0.3138 |
| Divisão aleatória, dia da semana corrigido | Rede neural | 0.2153 ± 0.0005 | 0.6159 ± 0.0007 | 0.1785 ± 0.0010 | 0.3160 ± 0.0006 |
| Divisão aleatória, dia da semana corrigido | Regressão logística | 0.2142 | 0.6150 | 0.1795 | 0.3160 |
| Temporal: treino 2007-2009, teste 2010 (dia corrigido) | Rede neural | 0.2020 ± 0.0010 | 0.6051 ± 0.0008 | 0.1764 ± 0.0020 | 0.3057 ± 0.0007 |
| Temporal: treino 2007-2009, teste 2010 (dia corrigido) | Regressão logística | 0.2007 | 0.6051 | 0.1730 | 0.3031 |
| Aleatória + Number_of_Vehicles (ablação) | Rede neural | 0.2369 ± 0.0006 | 0.6539 ± 0.0005 | 0.1932 ± 0.0015 | 0.3496 ± 0.0020 |

Tamanhos (treino/val/teste) temporal: 438761/77429/144411, taxa de graves no teste temporal: 0.1408

Tempo total: 18.7 min