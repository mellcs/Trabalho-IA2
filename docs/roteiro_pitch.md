# Roteiro do pitch: Fila Certa (até 7 minutos)

Apresentação: `docs/Apresentacao_Fila_Certa.html`. Abra no navegador e use **F** para tela cheia e as **setas** para avançar.
Teclas úteis: **B** apaga a tela, **Z** zera o cronômetro (aparece no canto a partir do slide 1).

| Slide | Quem | Tempo | Acumulado |
|---|---|---|---|
| Capa + 01 Problema | Mell | 1:00 | 1:00 |
| 02 Solução | Mell | 0:50 | 1:50 |
| 03 Como funciona | Nicolas | 1:00 | 2:50 |
| 04 Demonstração | Brayan | 2:00 | 4:50 |
| 05 Resultados | Nicolas | 0:50 | 5:40 |
| 06 Valor | Brayan | 0:40 | 6:20 |
| 07 Impacto e próximos passos + final | Mell | 0:40 | 7:00 |

---

## Capa + 01 Problema (Mell, ~1 min)

> Pensem numa loja online pequena. Todo dia chegam dezenas, às vezes centenas, de mensagens: avaliação, WhatsApp, e-mail, Instagram. A maioria é "amei, recomendo". Mas no meio estão o "meu pedido não chegou" e o "veio quebrado", e ninguém consegue ler tudo na hora.
>
> Fomos ver isso nos dados reais da Olist, um marketplace brasileiro: 100 mil pedidos. Quando a entrega chega no prazo, **9,6%** dos clientes reclamam. Quando atrasa, são **63%**, ou seja 6,6 vezes mais. E quando o pedido nem chega, 78%.
>
> Esse cliente escreve, espera resposta, não recebe e não volta. E o lojista nem fica sabendo que o problema era a transportadora.

## 02 Solução (Mell, ~50 s)

> A Fila Certa resolve isso. Você cola a mensagem do cliente e a IA devolve três coisas: se é reclamação, com que prioridade responder e qual o motivo provável (entrega, defeito, produto errado, pedido incompleto ou atendimento).
>
> Uma pergunta que vocês devem estar pensando: "mas no Olist o cliente já deu a nota". Exatamente. Usamos a nota **só para ensinar a rede**: nota 1 e 2 é reclamação, 4 e 5 não. O uso real é nas mensagens que **chegam sem nota**: WhatsApp, e-mail, SAC, Reclame Aqui.

## 03 Como funciona (Nicolas, ~1 min)

> Por dentro, é uma rede neural que nós treinamos do zero, em PyTorch. O texto é normalizado: minúsculas, sem acento. Mantemos o "não", porque "chegou" e "não chegou" são opostos.
>
> Cada palavra vira um número do vocabulário (5.585 palavras) e depois um vetor, o embedding, que a rede aprende. A rede BiLSTM lê a frase e devolve a probabilidade de ser reclamação.
>
> Sobre os dados: das 100 mil avaliações, 41.726 têm texto. Tiramos a nota 3, que é neutra e confunde o modelo (F1 de 0,27 num teste com três classes), e os textos duplicados, como "ótimo" repetido centenas de vezes. Sobram 30.977. Dividimos em treino, validação e teste, e o teste só foi aberto no final.
>
> Como reclamação é só um terço da base, usamos peso de classe na perda. A métrica principal é o F1 e o recall da reclamação, porque é a classe que interessa achar.

## 04 Demonstração (Brayan, ~2 min)

Abra o app antes do pitch (`python app.py`) e deixe a aba aberta. Se falhar, avance os passos do slide 4, que mostram prints do app real.

1. **Reclamação clara:** clique no 1º exemplo ("Comprei faz duas semanas e até agora não chegou…").
   > Probabilidade alta, prioridade **alta**, motivo **entrega**. Embaixo aparecem as palavras que mais pesaram: "não", "chegou". A equipe vê *por que* a IA decidiu.
2. **Elogio:** 2º exemplo ("Produto excelente, chegou antes do prazo…").
   > Prioridade **baixa**: pode esperar ou receber um agradecimento automático.
3. **Caso misto:** 3º exemplo ("O produto é bonito, mas veio uma peça a menos e a cor é diferente da foto").
   > "Bonito" puxa para elogio e "diferente" puxa para reclamação. A rede dá 96%, mas fica na prioridade **média**, a faixa em que vale uma leitura humana. Motivo: produto diferente.
4. **Lote:** aba 2, clique no exemplo do CSV e em "Analisar lote".
   > Aqui são mensagens de WhatsApp, e-mail e Reclame Aqui, que não têm nota nenhuma. A IA devolve a fila ordenada por prioridade e o gráfico dos motivos. O lojista vê na hora que o problema mais frequente é a entrega.

## 05 Resultados (Nicolas, ~50 s)

> Em 4.647 comentários que a rede nunca viu, ela encontra **92% das reclamações**, com F1 de 0,90.
>
> Sendo honestos: um modelo clássico, TF-IDF com regressão logística, chega praticamente no mesmo número. Os comentários são curtos e expressões como "não recebi" já são capturadas como bigramas. Muitos erros nem são da leitura: o cliente escreve um elogio e dá nota 1. O exemplo do slide mostra isso.
>
> A rede tem a vantagem nas frases em que o contexto importa e é a base para evoluir para outros canais.

## 06 Valor (Brayan, ~40 s)

> Na prática: se a equipe ler primeiro só a faixa de prioridade alta, ela lê **21%** das mensagens e já encontra **57% de todas as reclamações**, e 95% do que ela lê é reclamação de verdade.
>
> E os motivos são confiáveis: conferimos com a tabela de pedidos. Nas mensagens que a IA classificou como "entrega", 59% dos pedidos de fato atrasaram ou não chegaram. Nos outros motivos, só 17%.

## 07 Impacto e próximos passos (Mell, ~40 s)

> O mesmo sistema serve para qualquer loja e qualquer canal. Os próximos passos são: validar com mensagens reais de WhatsApp e SAC de um lojista, integrar com a caixa de entrada e testar um modelo pré-treinado em português.
>
> As limitações estão assumidas: a nota é um proxy, o treino foi em e-commerce e o motivo é por regras.
>
> Fila Certa: a reclamação certa no topo da fila. Obrigado!

Perguntas prováveis e respostas: [`perguntas_pitch.md`](perguntas_pitch.md).
