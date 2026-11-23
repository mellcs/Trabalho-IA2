# Painel de risco de acidentes graves de trânsito (Reino Unido)

Trabalho 2 de Inteligência Artificial II (2026/02) · Faculdade Antonio Meneghetti (AMF) · Sistemas de Informação
Modalidade escolhida: **treinamento de rede neural**

**Integrantes:** Mell Campos, Brayan Martins e Nicolas Marin.

![Painel: consulta e simulação](figuras/app_consultar.png)

---

## 1. Problema

No Brasil, 37.150 pessoas morreram no trânsito em 2024, segundo o Ministério da Saúde. Gestores de trânsito têm verba limitada para fiscalização, iluminação, limites de velocidade e obras, e costumam decidir olhando onde acontecem **mais** acidentes. Só que, nos dados, os lugares com mais acidentes não são os que têm acidentes **mais graves** (seção 6). A pergunta que o projeto responde é:

> Em quais condições e locais um acidente de trânsito tem maior chance de ser **grave ou fatal**?

O objetivo **não** é prever com certeza a gravidade de um acidente específico. É **priorizar risco**: ordenar situações da mais para a menos arriscada e mostrar quanto dos acidentes graves se concentra nas situações de maior risco estimado.

## 2. Solução

1. Uma **rede neural (PyTorch)** com embeddings que estima a probabilidade de um acidente ser grave ou fatal a partir das condições.
2. Uma análise de **pontos críticos (DBSCAN)** e do **risco por região**.
3. Um **painel em Gradio** (`app.py`) para o gestor, com três funções:
   - **Consultar e simular:** informa as condições (distrito, área, tipo e classe da via, limite de velocidade, cruzamento e seu controle, iluminação, superfície, clima, mês, dia e hora) e recebe o risco, a faixa (1 a 10) e uma tabela "e se mudasse um fator?", que mostra quanto o risco estimado muda trocando a iluminação, o limite de velocidade, o controle do cruzamento, o tipo de via, a superfície ou o clima.
   - **Mapa e ranking:** mapa de risco por células de ~5 km e ranking dos distritos com maior proporção de acidentes graves.
   - **Priorizar uma lista:** o usuário envia um CSV com vários locais/situações e recebe a lista ordenada por risco, pronta para baixar.

| Mapa e ranking | Priorização de lista |
|---|---|
| ![Mapa e ranking](figuras/app_mapa_ranking.png) | ![Priorizar lista](figuras/app_priorizar.png) |

## 3. Dados

- **Fonte principal:** dataset do Kaggle [`deeplumiere/accident-analysis-dataset`](https://www.kaggle.com/datasets/deeplumiere/accident-analysis-dataset) (~174 MB, 670.679 registros).
- **Colunas:** `Index`, `Accident_Severity`, `Accident Date`, `Latitude`, `Longitude`, `Light_Conditions`, `District Area`, `Number_of_Casualties`, `Number_of_Vehicles`, `Road_Surface_Conditions`, `Road_Type`, `Urban_or_Rural_Area`, `Weather_Conditions`, `Vehicle_Type`.
- **Fonte complementar (dados oficiais):** `Accidents0515.csv` do Kaggle [`silicon99/dft-accident-data`](https://www.kaggle.com/datasets/silicon99/dft-accident-data) (~245 MB), que reproduz o STATS19 do Departamento de Transportes do Reino Unido (2005-2015). Dele vêm **limite de velocidade, horário, tipo e controle do cruzamento, classe da via, faixa de pedestre e condições do local**.

### Auditoria e limpeza

| Achado | Decisão |
|---|---|
| `Index` aparece como `2.01E+12` (notação científica) em 239.478 linhas, o que indica corrupção por planilha | `Index` descartado; cada linha é tratada como um registro |
| Nas 430.776 linhas com `Index` intacto, ele segue o padrão do STATS19 (dados oficiais do Reino Unido) e começa com o **ano real**: 2007, 2008, 2009 e 2010 aparecem como 2019, 2020, 2021 e 2022 | Os dados são de **2007 a 2010**, com o ano somado de 12. O **dia da semana** é calculado na data real (ver abaixo) |
| Exatamente 10.000 linhas com data de 2023 a 2025 (contra 144 a 182 mil por ano em 2019-2022) | Bloco excluído por suspeita de ter sido acrescentado ao dataset original |
| 52 linhas duplicadas e 26 sem coordenadas | Removidas |
| Nulos em clima, tipo de via, superfície e área | Categoria `Unknown` |

| Junção com os dados oficiais (data real, coordenadas, gravidade, nº de veículos e de vítimas) | **99,97%** das linhas casaram. Nas 420 mil linhas com `Index` intacto, a junção pelo código bate em 100% (mesma gravidade, coordenadas e data exatamente 12 anos antes), o que confirma o deslocamento das datas |

**Por que corrigir o dia da semana:** com a data do dataset, o pico de acidentes cai no sábado e o mínimo na segunda. Voltando 12 anos, o pico vai para a sexta e o mínimo para o domingo, que é o padrão conhecido do trânsito britânico. O mês não muda. A tabela que prova o deslocamento está no notebook (seção de auditoria).

**Base final: 660.601 registros.** Divisão estratificada 70/15/15 (`random_state=42`): 462.420 treino, 99.090 validação, 99.091 teste.

**Variável-alvo:** binária, `1` = acidente grave ou fatal (14,66% dos registros), `0` = leve. A distribuição original em três classes é 85,3% leve, 13,4% grave e 1,3% fatal.

**Variáveis de entrada do modelo final** (27):

- 7 categóricas do dataset do Kaggle: iluminação, superfície da pista, tipo de via, área urbana/rural, clima, tipo de veículo e distrito (422 distritos + código para categoria desconhecida);
- 9 categóricas dos dados oficiais: limite de velocidade, tipo de cruzamento, controle do cruzamento, classe da 1ª e da 2ª via, faixa de pedestre (controle humano e estrutura física), condições especiais do local e obstáculos na pista;
- 11 numéricas: latitude e longitude (padronizadas), mês, dia da semana e hora (seno e cosseno), fim de semana, limite de velocidade (padronizado) e indicador de hora válida.

O notebook usa só as 14 primeiras (7 categóricas e 7 numéricas); a seção 6 compara as duas versões.

**Fora do modelo:** `Index`, `Number_of_Casualties` (a gravidade é definida pelas vítimas, então seria vazamento do alvo), `Number_of_Vehicles` (só é conhecido depois do acidente; foi testado à parte, seção 6) e, dos dados oficiais, tudo o que só existe depois do acidente (vítimas, veículos, presença da polícia).

Vocabulários, médias e desvios são aprendidos **apenas no treino**, para não vazar informação da validação e do teste.

## 4. Abordagem

1. **Auditoria e limpeza** (seção anterior).
2. **Experimento inicial com 3 classes** (leve, grave, fatal). A classe fatal (1,3%) mostrou sinal muito fraco (PR-AUC de 0,03 para fatal), o que motivou o alvo binário.
3. **Baselines:** acaso, regressão logística e LightGBM.
4. **Rede neural** com embeddings e experimentos de função de custo e otimizador.
5. **Clusterização** (DBSCAN) e **risco por região**.
6. **Interpretação** (importância por permutação).
7. **Avaliação final no teste**, aberto uma única vez.
8. **Experimentos extras:** várias sementes, validação temporal e ablação com `Number_of_Vehicles`.
9. **Junção com os dados oficiais** (STATS19) e retreino com limite de velocidade, horário e cruzamento.
10. **Painel** em Gradio, com o modelo final.

### Métricas

Acurácia não serve aqui: um modelo que sempre responde "leve" acerta 85,3% e é inútil. Usamos:

- **PR-AUC** (referência: 0,147, a proporção de positivos) e **ROC-AUC**;
- **captura no top k%**: que fração dos acidentes graves/fatais está entre os k% de maior risco estimado. Ao acaso, seria k%.

## 5. Arquitetura da rede neural

```
16 variáveis categóricas ──► Embedding (dim = min(16, (n+1)//2)) cada ──┐
                                                                          ├─► concatenação (97)
11 variáveis numéricas ───────────────────────────────────────────────────┘
        │
        ▼
Linear(97→128) · ReLU · Dropout(0,3) · Linear(128→64) · ReLU · Dropout(0,3) · Linear(64→1)  ──► logit
```

- 28.323 parâmetros no modelo final (21.902 na versão do notebook, com 14 variáveis e concatenação de 50).
- Adam (`lr=1e-3`), batch de 1.024, até 15 épocas, **early stopping** com paciência de 3, monitorado pelo PR-AUC na validação (os pesos da melhor época são restaurados).
- `BCEWithLogitsLoss` (a saída é um logit; a sigmoide só é aplicada na predição). Sementes fixas.

![Curva de treinamento](figuras/curva_treino.png)

### Experimentos (validação, versão do notebook)

| Rede | Função de custo / otimizador | PR-AUC | Probabilidade média prevista |
|---|---|---|---|
| 1 | BCE com `pos_weight` (5,82), Adam | 0,2150 | 0,483 |
| **2** | **BCE sem pesos, Adam** | **0,2157** | **0,148** |
| 3 | Focal loss (γ=2, α=0,75), Adam | 0,2151 | 0,432 |
| 4 | BCE sem pesos, SGD com momentum (lr 0,05) | 0,2111 | n/d |

Proporção real de graves/fatais: 0,1466. As diferenças de PR-AUC entre as redes 1, 2 e 3 são menores do que a variação entre sementes. O que muda é a **calibração**: peso de classe e focal loss superestimam a probabilidade (média ~0,43 a 0,48), e a Rede 2 acerta a média. Por isso a Rede 2 é o modelo final. O Adam chegou em 3 épocas ao patamar que o SGD (sem ajuste fino) levou 15 épocas para alcançar.

## 6. Resultados

### Avaliação final no conjunto de teste

| Modelo | PR-AUC | ROC-AUC | Captura top 10% | Captura top 20% | Captura top 30% |
|---|---|---|---|---|---|
| Acaso | 0,146 | 0,500 | 10,0% | 20,2% | 30,0% |
| Regressão logística (notebook) | 0,214 | 0,615 | 18,0% | 31,6% | 43,6% |
| LightGBM (notebook) | 0,211 | 0,609 | 17,6% | 31,3% | 42,9% |
| Rede neural, Rede 2 (notebook) | 0,216 | 0,616 | 17,9% | 31,5% | 43,7% |
| **Rede neural + dados oficiais (modelo final, painel)** | **0,230** | **0,632** | **19,7%** | **33,5%** | **45,1%** |

- Os 10% de situações de maior risco estimado concentram **~20% dos acidentes graves e fatais**, o dobro do acaso; os 30% de maior risco concentram ~45%.
- **Dados oficiais:** limite de velocidade, horário e cruzamento elevam o ROC-AUC de 0,616 para 0,632 com a mesma rede, a mesma divisão e o mesmo teste (detalhes abaixo). O ganho é real (maior que a variação entre sementes), mas moderado: o que mais pesa na gravidade (velocidade real, idade, cinto) continua fora dos dados.
- **Rede x regressão logística:** com 5 sementes, a rede fica em PR-AUC 0,2153 ± 0,0005 contra 0,2142 da regressão. A diferença é consistente, mas pequena demais para ter importância prática (bootstrap pareado no teste: IC 95% de [-0,0001; 0,0030]). Modelos mais complexos não extraíram mais sinal que o modelo linear; a vantagem da rede é ser um único modelo bem calibrado que alimenta o painel.
- **Calibração (modelo final):** a probabilidade prevista acompanha a proporção real nos 10 decis de risco (de 5,4% previsto contra 6,7% real no decil mais baixo até 28,1% contra 28,9% no mais alto).

![Curva de ganho e calibração](figuras/ganho_calibracao.png)

### Experimentos extras (`src/experimentos.py`)

Rede 2, 5 sementes, média ± desvio no teste ([detalhes](docs/resultados_experimentos.md)):

| Cenário | PR-AUC | ROC-AUC | Captura top 10% |
|---|---|---|---|
| Divisão aleatória, dia da semana original | 0,2155 ± 0,0012 | 0,6161 ± 0,0009 | 17,9% |
| Divisão aleatória, dia da semana corrigido | 0,2153 ± 0,0005 | 0,6159 ± 0,0007 | 17,9% |
| **Temporal: treino 2007-2009, teste 2010** | **0,2020 ± 0,0010** | **0,6051 ± 0,0008** | **17,6%** |
| Aleatória + `Number_of_Vehicles` | 0,2369 ± 0,0006 | 0,6539 ± 0,0005 | 19,3% |

- **Dia da semana:** a correção não muda as métricas (o calendário pesa pouco), mas deixa o modelo e o painel coerentes com o dia real.
- **Validação temporal:** treinando só com o passado e testando no ano seguinte, o ROC-AUC cai de 0,616 para 0,605 e a captura no top 10% quase não muda. O padrão aprendido **se sustenta no tempo**.
- **`Number_of_Vehicles`:** aumenta o ROC-AUC para 0,654, mas só é conhecido depois do acidente. Fica fora do painel de prevenção e é uma ideia para um uso diferente (triagem de ocorrências, seção 10).

### Ganho com os dados oficiais (STATS19)

Mesma rede, mesma divisão e mesmo conjunto de teste; rede com 3 sementes (média ± desvio). Detalhes em [`docs/resultados_dados_oficiais.md`](docs/resultados_dados_oficiais.md).

| Cenário | PR-AUC | ROC-AUC | Captura top 10% |
|---|---|---|---|
| Só o dataset do Kaggle (notebook) | 0,2154 ± 0,0006 | 0,6155 ± 0,0007 | 17,8% |
| + limite de velocidade e horário | 0,2241 ± 0,0007 | 0,6234 ± 0,0006 | 18,9% |
| **+ todas as variáveis oficiais do local (painel)** | **0,2307 ± 0,0009** | **0,6335 ± 0,0013** | **19,8%** |
| Temporal (treino 2007-2009, teste 2010), variáveis oficiais | 0,2195 ± 0,0009 | 0,6258 ± 0,0009 | 19,4% |

- O ganho é **real, mas moderado** (+0,018 de ROC-AUC). A regressão logística chega a 0,631: o limite está nos dados, não no modelo.
- A gravidade depende sobretudo de fatores que nenhuma base de "condições do local" registra (velocidade real, cinto, idade, tipo de impacto).

### Pontos críticos e risco por região

**DBSCAN sobre os acidentes graves/fatais do treino** (distância haversine). Varremos `eps` ∈ {200, 500, 1000, 2000} m e `minPts` ∈ {10, 20, 50}, avaliando silhouette e Davies-Bouldin (só sobre pontos em clusters). As melhores métricas vinham de configurações que cobriam menos de 1% dos pontos, então escolhemos **`eps` = 2000 m e `minPts` = 50**: 90 clusters, 34,5% dos pontos em clusters, silhouette 0,513 e Davies-Bouldin 0,298.

**Os clusters marcam volume, não risco.** Na validação, 45,8% dos acidentes caem dentro dos clusters, mas a proporção de graves ali é **menor**:

| Validação | Graves/fatais |
|---|---|
| Global | 14,7% |
| Dentro dos clusters | 13,1% |
| Fora dos clusters | 16,0% |

Só 8,3% dos acidentes dentro dos clusters são rurais, contra 59,7% fora. Acidentes rurais têm proporção maior de graves (17,8% contra 12,9% nos urbanos).

**Risco por região** (células de ~5 km, taxa alisada): as regiões que parecem mais perigosas no treino continuam parecendo na validação.

| | Células | Spearman (treino × validação) | 20% de maior taxa no treino → validação | 20% de menor taxa → validação |
|---|---|---|---|---|
| Todas | 977 | 0,325 | 15,5% | 10,3% |
| Urbanas | 920 | 0,302 | 14,9% | 10,0% |
| Rurais | 337 | 0,373 | 18,5% | 9,2% |

O padrão regional persiste dentro de áreas urbanas e rurais, então não é só o efeito "rural contra urbano".

![Mapa de risco regional](figuras/mapa_risco.png)

### O que o modelo usa (importância por permutação, validação)

| Bloco | Queda de PR-AUC |
|---|---|
| Localização (distrito, coordenadas, rural/urbano) | 0,0470 |
| Condições da via e do ambiente (luz, tipo de via, superfície, clima) | 0,0300 |
| Calendário (mês, dia da semana) | 0,0095 |
| Tipo de veículo | 0,0000 |

Na validação, a proporção de graves/fatais é de **23,1% sem iluminação pública contra 13,6% de dia**, e de **8,5% em rotatórias contra 15,6% em pistas simples**. O tipo de veículo não discrimina a gravidade neste dataset (motocicletas entre 14,0% e 15,2%, carros 14,6%), o que contraria estatísticas de trânsito em geral; por isso evitamos conclusões sobre veículos.

## 7. Estrutura do repositório

```
├── app.py                         # painel Gradio (usa o modelo salvo em modelos/)
├── notebooks/
│   └── Trabalho_IA_G2_Acidentes_no_Reino_Unido.ipynb   # análise completa, com as saídas
├── src/
│   ├── dados.py                   # carga, limpeza, correção de datas, junção com os dados oficiais, features
│   ├── modelo.py                  # rede neural, treino com early stopping, métricas
│   ├── treinar.py                 # treina o modelo final e gera modelos/ e figuras/
│   ├── experimentos.py            # sementes, validação temporal, ablação
│   └── experimento_oficial.py     # ganho com os dados oficiais (STATS19)
├── modelos/                       # rede_final.pt + artefatos (vocabulários, faixas, mapa, ranking)
├── docs/
│   ├── resultados_experimentos.md # saída de src/experimentos.py
│   ├── resultados_dados_oficiais.md # saída de src/experimento_oficial.py
│   └── exemplo_lista.csv          # exemplo de entrada para a aba "Priorizar uma lista"
├── figuras/                       # gráficos e prints do painel
├── data/                          # coloque aqui o accident_data.csv e o Accidents0515.csv (não versionados)
└── requirements.txt
```

## 8. Como executar

Requisitos: Python 3.10+ e ~2 GB livres. CPU é suficiente.

### Painel (sem precisar do dataset)

O modelo treinado já está em `modelos/`.

```bash
git clone https://github.com/mellcs/Trabalho-IA2.git
cd Trabalho-IA2
pip install -r requirements.txt
python app.py              # abre em http://127.0.0.1:7860
python app.py --share      # opcional: link público temporário
```

### Reproduzir o treino e os experimentos

1. Baixe o dataset principal em <https://www.kaggle.com/datasets/deeplumiere/accident-analysis-dataset> e salve como `data/accident_data.csv`.
2. Baixe os dados oficiais em <https://www.kaggle.com/datasets/silicon99/dft-accident-data> (só o arquivo `Accidents0515.csv`) e salve como `data/Accidents0515.csv`.
3. Rode:

```bash
# modelo final do painel (com dados oficiais): ~3 min, gera modelos/ e figuras/
python -m src.treinar --csv data/accident_data.csv --oficial data/Accidents0515.csv

# comparação com e sem dados oficiais + validação temporal: ~20 min
python -m src.experimento_oficial --csv data/accident_data.csv --oficial data/Accidents0515.csv

# experimentos da versão do notebook (sementes, temporal, ablação): ~20 min
python -m src.experimentos --csv data/accident_data.csv
```

Sem `--oficial`, `src.treinar` treina a versão do notebook (só com o dataset principal).

### Notebook no Google Colab

1. Abra `notebooks/Trabalho_IA_G2_Acidentes_no_Reino_Unido.ipynb` no Colab (Arquivo → Abrir notebook → GitHub).
2. Envie o CSV ao Colab (painel de arquivos à esquerda) com o nome `accident_data.csv`.
3. **Ambiente de execução → Executar tudo.** A última célula instala o Gradio e abre a interface simples do notebook com link público.

O notebook documenta toda a análise com o dataset principal. O modelo do painel é treinado por `src/treinar.py` com a mesma arquitetura, divisão e sementes, acrescentando as variáveis dos dados oficiais.

Notas: sementes fixas (`random_state=42`, `torch.manual_seed(42)`); diferenças na quarta casa decimal podem aparecer conforme a versão das bibliotecas. Cada rede leva ~1 minuto em CPU. Se o Colab reiniciar a sessão e aparecer `NameError`, execute de novo desde o início.

## 9. Limitações

- **Sinal moderado.** ROC-AUC de 0,63. Mesmo com os dados oficiais, faltam as variáveis que mais determinam a gravidade (velocidade real, idade, uso de cinto). O projeto serve para **priorizar risco**, não para prever um acidente.
- **Associação, não causa.** A simulação "e se mudasse um fator?" mostra como o risco estimado muda, não garante o efeito de uma obra. Alguns resultados são contraintuitivos (pistas com gelo ou neve têm acidentes menos graves, provavelmente porque se dirige mais devagar; vias de 70 mph são em geral autoestradas, mais seguras que estradas rurais de 60 mph).
- **Dados antigos e com problemas.** Os acidentes são de 2007 a 2010, o `Index` está corrompido em ~36% das linhas e o bloco 2023-2025 foi excluído. O comportamento do tipo de veículo é inesperado.
- **Localização aproximada.** O painel usa o centro dos acidentes do distrito, perdendo a variação dentro dele.
- **Calibração nos extremos.** Verificada por decil; probabilidades acima de ~30% vêm de combinações raras e são extrapolação.
- **Sem ajuste fino de hiperparâmetros.** A arquitetura e o LightGBM usaram valores fixos.

## 10. Impacto e próximos passos

- **Uso real:** órgãos de trânsito podem usar o painel para escolher onde instalar iluminação, reforçar fiscalização em noites de fim de semana em vias rurais, ou ordenar uma lista de trechos candidatos a obras.
- **Brasil:** a PRF publica em dados abertos os acidentes nas rodovias federais desde 2007 (com localização e condições). O mesmo pipeline pode ser treinado com eles, por exemplo para as rodovias que passam por Santa Maria (BR-158, BR-287).
- **Dados mais recentes:** retreinar com os anos mais novos do STATS19.
- **Triagem de ocorrências:** um segundo modelo com variáveis conhecidas logo após o acidente (como `Number_of_Vehicles`, que sobe o ROC-AUC para 0,65) poderia ajudar a priorizar o envio de ambulâncias.
- **Mapa interativo** com entrada por coordenadas, em vez do centro do distrito.
- **Incerteza por região**, para marcar células com poucos acidentes.
