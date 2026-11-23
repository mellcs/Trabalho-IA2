# Ganho com os dados oficiais (STATS19)

Junção com `Accidents0515.csv`: 99.97% das linhas casadas (data real, coordenadas, gravidade, nº de veículos e de vítimas).
Rede com 3 sementes (média ± desvio) e regressão logística; mesmo conjunto de teste.

| Cenário | Modelo | PR-AUC | ROC-AUC | Captura top 10% | Captura top 20% |
|---|---|---|---|---|---|
| Só o dataset do Kaggle (notebook) | Rede neural | 0.2154 ± 0.0006 | 0.6155 ± 0.0007 | 0.1778 ± 0.0005 | 0.3158 ± 0.0006 |
| Só o dataset do Kaggle (notebook) | Regressão logística | 0.2142 | 0.6150 | 0.1795 | 0.3160 |
| + limite de velocidade e horário | Rede neural | 0.2241 ± 0.0007 | 0.6234 ± 0.0006 | 0.1891 ± 0.0010 | 0.3306 ± 0.0016 |
| + limite de velocidade e horário | Regressão logística | 0.2208 | 0.6217 | 0.1876 | 0.3288 |
| + todas as variáveis oficiais do local (painel) | Rede neural | 0.2307 ± 0.0009 | 0.6335 ± 0.0013 | 0.1977 ± 0.0017 | 0.3380 ± 0.0019 |
| + todas as variáveis oficiais do local (painel) | Regressão logística | 0.2261 | 0.6307 | 0.1953 | 0.3336 |
| Temporal (treino 2007-2009, teste 2010), variáveis oficiais | Rede neural | 0.2195 ± 0.0009 | 0.6258 ± 0.0009 | 0.1940 ± 0.0016 | 0.3301 ± 0.0010 |
| Temporal (treino 2007-2009, teste 2010), variáveis oficiais | Regressão logística | 0.2134 | 0.6230 | 0.1876 | 0.3222 |

Variáveis oficiais usadas: Speed_limit, Junction_Detail, Junction_Control, 1st_Road_Class, 2nd_Road_Class, Pedestrian_Crossing-Human_Control, Pedestrian_Crossing-Physical_Facilities, Special_Conditions_at_Site, Carriageway_Hazards; horário (seno/cosseno).
Não entram: nº de vítimas, nº de veículos, dados dos veículos e das vítimas, presença da polícia.

Tempo total: 15.7 min