# Histórico do projeto e mudança de tema

Trabalho 2 de Inteligência Artificial II (2026/02) · AMF · Sistemas de Informação
**Integrantes:** Mell Campos, Brayan Martins e Nicolas Marin.

Este documento registra tudo o que fizemos com o primeiro tema (gravidade de acidentes de trânsito no Reino Unido), o que descobrimos e por que decidimos **trocar para um problema com impacto mais direto**: identificar automaticamente reclamações em avaliações de clientes de e-commerce brasileiro (PLN em português).

---

## Resumo em 30 segundos

| | Tema 1: acidentes (Reino Unido) | Tema 2: avaliações negativas (Olist) |
|---|---|---|
| Pergunta | Um acidente, nas condições do local, tende a ser grave ou fatal? | Este comentário de cliente é uma reclamação? Qual o motivo? |
| Dados | 660 mil acidentes (Kaggle) + base oficial STATS19 | ~100 mil pedidos e avaliações, em português |
| Melhor resultado | ROC-AUC **0,63** | em desenvolvimento |
| Quem usa | Secretaria de trânsito, para priorizar fiscalização e obras | Lojista ou marketplace, para responder primeiro quem está insatisfeito |
| Por que mudamos | A informação disponível explica pouco da gravidade (teto de 0,61 a 0,63 com qualquer modelo) | O texto carrega sinal forte e o uso é imediato |

---

## 1. Tema 1: acidentes de trânsito graves no Reino Unido

### 1.1 Problema e proposta

Gestores de trânsito têm verba limitada. A ideia era um **painel de priorização**: dadas as condições de um local (via, iluminação, clima, região, horário), estimar a chance de um acidente ali ser grave ou fatal e ordenar os locais por risco.

### 1.2 Dados e auditoria

Dataset do Kaggle `deeplumiere/accident-analysis-dataset` (670.679 linhas). A auditoria encontrou três problemas, todos tratados:

1. **Datas deslocadas em 12 anos.** Os acidentes reais são de 2007 a 2010, mas aparecem como 2019 a 2022. Confirmamos de duas formas: o código do acidente começa com o ano real, e a junção com a base oficial casou 100% dos campos.
2. **Coluna Index corrompida** em cerca de 36% das linhas. Ela foi descartada, e a junção usa data, coordenadas e contagens.
3. **Bloco sintético de 10 mil linhas datadas de 2023 a 2025.** Foi removido.

**Cuidados contra vazamento:**
- Ficaram fora tudo o que só se sabe depois do acidente: número de vítimas, número de veículos, dados das vítimas e presença da polícia.
- Vocabulários, médias e desvios foram aprendidos só no treino.
- O teste foi aberto uma única vez.

### 1.3 Modelo

- **Rede:** PyTorch com embeddings para as variáveis categóricas, camadas 128 → 64 → 1, ReLU e Dropout 0,3.
- **Treino:** Adam, BCE e early stopping pela PR-AUC de validação.
- **Comparações:** regressão logística e LightGBM.
- **Validações extras:** 5 sementes, validação temporal (treino 2007–2009, teste 2010) e calibração por decil.

### 1.4 Resultados (conjunto de teste, 14% de graves/fatais)

| Modelo | ROC-AUC | PR-AUC | Captura no top 10% |
|---|---|---|---|
| Rede neural (notebook) | 0,616 | 0,216 | 17,9% |
| Regressão logística | 0,615 | 0,214 | 18,0% |
| LightGBM | 0,609 | — | — |
| Rede, validação temporal | 0,605 | 0,202 | 17,6% |
| Rede + número de veículos (só ablação, vaza informação) | 0,654 | 0,237 | 19,3% |

Detalhes: [`docs/resultados_experimentos.md`](docs/resultados_experimentos.md) e o notebook em `notebooks/`.

### 1.5 Painel (Gradio)

O app (`app.py`) tem quatro funções:
- consulta de risco com faixa de 1 a 10;
- simulação "E se?", que troca um fator por vez;
- mapa e ranking de distritos, com taxa alisada;
- priorização de uma lista de locais via CSV.

Os pontos críticos foram encontrados com DBSCAN (haversine). Os clusters marcaram volume de acidentes, não risco, e isso está documentado no README.

### 1.6 Tentativa de melhorar: junção com a base oficial (STATS19)

Para tentar subir o AUC, juntamos o dataset do Kaggle com a base oficial do governo britânico (`Accidents0515.csv`, 99,97% das linhas casadas). Ela traz limite de velocidade, horário, tipo e controle de cruzamento, classe da via e faixa de pedestre. Mesma rede, mesma divisão e mesmo teste (3 sementes):

| Cenário | ROC-AUC | PR-AUC | Captura top 10% |
|---|---|---|---|
| Só o dataset do Kaggle | 0,6155 | 0,2154 | 17,8% |
| + limite de velocidade e horário | 0,6234 | 0,2241 | 18,9% |
| + todas as variáveis oficiais do local | **0,6335** | **0,2307** | **19,8%** |
| Temporal, com as variáveis oficiais | 0,6258 | 0,2195 | 19,4% |

O código, os resultados e o painel atualizado estão no branch [`acidentes-v2-dados-oficiais`](https://github.com/mellcs/Trabalho-IA2/tree/acidentes-v2-dados-oficiais), com os resultados em `docs/resultados_dados_oficiais.md`.

---

## 2. Por que trocamos de tema

1. **O teto é dos dados, não do modelo.** Rede neural, regressão logística e LightGBM param todos entre 0,61 e 0,63 de ROC-AUC, mesmo com uma segunda base oficial. O ganho com a base oficial foi real, mas pequeno: +0,018.
2. **O que decide a gravidade não está registrado.** Velocidade real, uso de cinto, idade, tipo de impacto e tempo de socorro não existem em nenhuma base de "condições do local".
3. **As variáveis que melhoram o modelo só existem depois do acidente.** O número de veículos sobe o AUC para 0,65, mas usar isso seria vazamento num painel de prevenção.
4. **Com AUC 0,63, o valor prático é limitado.** O modelo prioriza cerca de 2× melhor que o acaso nos 10% de maior risco, mas não consegue apontar com confiança um local ou situação específica. Para o critério de "resolver um problema real", achamos o resultado fraco.

**Conclusão:** em vez de forçar mais pontos de AUC num problema com pouco sinal, mudamos para um problema em que **a entrada carrega a informação que o modelo precisa** e o uso do resultado é imediato.

---

## 3. Tema 2: identificar reclamações em avaliações de clientes (em desenvolvimento)

### 3.1 Problema

Lojistas e marketplaces recebem muitas avaliações em texto livre e não conseguem ler todas. Reclamações importantes (produto não chegou, veio quebrado, veio diferente do anunciado) ficam sem resposta, e o cliente é perdido.

### 3.2 Solução proposta

Uma rede neural lê o comentário e identifica automaticamente as mensagens negativas. Nas negativas, o sistema mostra o motivo provável (entrega, defeito, produto errado ou atendimento). Assim a equipe responde primeiro quem está insatisfeito e o lojista corrige a causa.

### 3.3 Dados

- **Base:** Olist, e-commerce brasileiro (Kaggle `olistbr/brazilian-ecommerce`), com cerca de 100 mil pedidos de 2016 a 2018 e 99.224 avaliações com nota de 1 a 5 e comentário.
- **Alvo:** `negativo` = 1 se a nota for ≤ 2 e 0 se for ≥ 4.
- **Excluídos:** avaliações sem comentário, notas 3 e textos duplicados.
- **Base final:** cerca de 32 mil exemplos, 33% negativos.

### 3.4 Modelo previsto

- **Rede:** texto tokenizado, embedding, camada de sequência (LSTM ou CNN 1D) e saída sigmoide.
- **Desbalanceamento:** peso de classe na perda.
- **Treino:** early stopping na validação; o teste é usado só no fim.
- **Métrica principal:** F1 e recall da classe negativa, que é a que o negócio quer encontrar. A acurácia fica como métrica secundária.

### 3.5 Limitações já conhecidas

- **A nota funciona como rótulo (proxy).** Quem escreve no Olist já deu uma nota. O modelo é pensado para textos que chegam **sem** nota: SAC, WhatsApp, e-mail, redes sociais e sites de reclamação. O Olist serve como base de treino rotulada.
- **Viés de amostra.** Só uma parte dos clientes escreve comentário, então a amostra não representa toda a base.

---

## 4. O que aproveitamos do tema 1

- **Método:** auditoria dos dados antes de modelar, cuidado com vazamento, teste aberto uma vez só, comparação com modelos simples e validação com várias sementes.
- **Leitura honesta das métricas:** saber quando o limite é do dado e não do modelo foi o que motivou a troca.
- **Estrutura:** código em `src/`, artefatos em `modelos/`, painel em Gradio e README com resultados reproduzíveis.

## 5. Linha do tempo

| Data | Quem | O quê |
|---|---|---|
| 02–04/11 | Mell | README inicial, notebook e plano de experimentos |
| 21/11 | Brayan | Painel Gradio e código organizado em `src/` |
| 22/11 | Nicolas | Notebook executado com o dataset completo, experimentos extras e figuras |
| 23/11 | Brayan | Junção com a base oficial STATS19 (branch `acidentes-v2-dados-oficiais`) e este histórico |
| 23/11 | Grupo | Decisão de trocar para o tema 2 (avaliações Olist) |
