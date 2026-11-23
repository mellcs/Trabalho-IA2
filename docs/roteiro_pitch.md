# Roteiro do pitch: Fila Certa

Apresentação: `docs/Apresentacao_Fila_Certa.html`. Abra no navegador, aperte **F** para tela cheia e use as **setas** (ou clique) para avançar. **B** apaga a tela.
O canto superior direito de cada slide mostra quem fala. O total fica perto de 7 minutos; a demo é a parte mais longa.

**Antes de começar:** rode `python app.py` e deixe o navegador aberto na Caixa de entrada. Se a demo falhar, os slides da demonstração têm prints do app real.

| Slides | Quem | Assunto |
|---|---|---|
| 1, 2 e 3 | **Mell** | Capa, a Ana e o mercado, dados reais e a virada |
| 4 e 5 | **Brayan** | A solução e a demonstração ao vivo |
| 6 e 7 | **Nicolas** | Como a rede funciona e os resultados |
| 8 e 9 | **Brayan** | Valor para quem usa e como vira negócio |
| 10 e 11 | **Mell** | Próximos passos, limitações e encerramento |

---

## Mell: slides 1, 2 e 3

**Slide 1 (capa)**
> Boa tarde. Nós somos a Mell, o Brayan e o Nicolas, e vamos apresentar a Fila Certa: uma caixa de entrada que lê as mensagens dos clientes e coloca as reclamações no topo, já com o motivo e a resposta.

**Slide 2 (a Ana)**
> Imaginem a Ana. Ela tem uma loja online e vende sozinha. Todo dia chegam mensagens pelo WhatsApp, e-mail, marketplace, Instagram, Reclame Aqui. A maioria é "amei, chegou certinho". Mas no meio estão o "meu pedido não chegou" e o "veio quebrado". Quando ela acha essa mensagem, o cliente já desistiu e já falou mal da loja.
>
> E a Ana não é exceção: o Brasil tem 1,9 milhão de lojas virtuais, e 87,5% são de pequeno porte. E cerca de 80% dos consumidores já desistiram de uma compra por causa de atendimento ruim.

**Slide 3 (dados reais)**
> Fomos ver isso nos dados reais da Olist, um marketplace brasileiro, com 100 mil pedidos. Quando a entrega chega no prazo, 9,6% dos clientes reclamam. Quando atrasa, 63%. Quando nem chega, 78%. Ou seja, quando a entrega falha, o cliente escreve.
>
> E aqui está o ponto principal: o difícil não é saber se uma mensagem é reclamação. Lendo, qualquer um sabe. O caro é ler todas para achar as poucas urgentes, e saber o que fazer com cada uma. Brayan.

## Brayan: slides 4 e 5

**Slide 4 (a solução)**
> A Fila Certa é uma caixa de entrada que se organiza sozinha. Ela faz quatro coisas. **Prioriza:** uma rede neural lê cada mensagem e coloca as urgentes no topo. **Explica:** mostra o motivo e destaca as palavras que pesaram. **Sugere a resposta:** prazo, ação e um texto pronto para revisar. E **avisa** quando algo começa a dar errado.

**Slide 5 (demonstração ao vivo)** — troque para o navegador com o app.
1. **Caixa de entrada.**
   > Aqui chegaram 128 mensagens de vários canais. A rede já leu todas: as 38 urgentes estão no topo. E olhem esse aviso verde: 56 mensagens podem esperar, porque nessa faixa só 2% são reclamações. Com um clique, elas recebem um agradecimento automático.
   Clique em **Agradecer as 56** e depois em **Desfazer** no aviso.
2. **Abrir a primeira mensagem.**
   > Ao abrir, a parte que pesou fica destacada no texto. A rede diz que é reclamação, prioridade alta, responder em até 2 horas. O motivo é entrega, e ela já diz o que fazer: consultar a transportadora. Embaixo, a resposta pronta. Eu reviso, clico em "Copiar e resolver" e ela já me leva para a próxima.
   Clique em **Copiar e resolver**.
3. **Painel.** Clique em **Painel** no menu.
   > No painel, o lojista vê o que mais gera reclamação e recebe alertas. Nos dados reais, em março de 2018 as reclamações de entrega subiram 86% em um mês. O painel avisa antes de virar crise. E aqui a calculadora: com 150 mensagens por dia, a equipe economiza cerca de 65 horas por mês.
4. **Testar (opcional, se sobrar tempo).** Clique em **Testar mensagem**, cole algo que o professor sugerir e clique em **Analisar**.
5. **Celular (opcional).** Avance o slide para mostrar a versão de celular.
> Nicolas, como a rede faz isso?

## Nicolas: slides 6 e 7

**Slide 6 (como funciona)**
> A rede é nossa, treinada do zero em PyTorch. O texto é normalizado: minúsculas, sem acento. Cada palavra vira um número do vocabulário, que tem 5.585 palavras, e depois um vetor de 100 dimensões, o embedding, que a rede aprende. Uma BiLSTM lê a frase nos dois sentidos, guardando o contexto, e devolve a probabilidade de ser reclamação.
>
> Quatro decisões importantes. O "não" fica, porque "chegou" e "não chegou" são opostos. Como só um terço da base é reclamação, usamos peso de classe na perda. Early stopping, para a rede não decorar o treino. E sem vazamento: tiramos os duplicados antes de dividir, e o teste só foi aberto no final.
>
> Das 100 mil avaliações, 41 mil têm texto. Tirando a nota 3 e os duplicados, sobram 31 mil, divididos em treino, validação e teste.

**Slide 7 (resultados)**
> Em 4.647 mensagens que a rede nunca viu, ela encontra 92% das reclamações.
>
> Sendo honestos: o modelo clássico, TF-IDF com regressão logística, chega praticamente no mesmo número. As mensagens são curtas e "não recebi" já é um par de palavras que ele captura. A rede ganha quando o sentido não está numa palavra só: "Pelo valor dele eu esperava um produto melhor" a rede pega, e o clássico erra.
>
> E ela erra também: "Chegou rápido, mas parou de funcionar" sai como elogio, porque "chegou rápido" pesa muito. E alguns erros nem são erros: "Não foi entregue o meu pedido", com nota 5. O rótulo é que está errado. Brayan.

## Brayan: slides 8 e 9

**Slide 8 (valor)**
> O resultado prático: a Ana lê só 41% das mensagens, as de prioridade alta e média, e encontra 96% das reclamações. As outras 59% recebem um agradecimento automático. Na calculadora, são cerca de 65 horas por mês, uns 1.600 reais. E os motivos são confiáveis: nas reclamações classificadas como entrega, 59% dos pedidos realmente atrasaram ou não chegaram, contra 17% nos outros motivos.

**Slide 9 (negócio)**
> Como isso vira negócio: uma assinatura mensal por volume de mensagens, para as lojas pequenas e médias que não têm equipe de SAC. A entrada é por integrações: WhatsApp Business, e-mail, marketplaces. A API já existe no app.
>
> E um ponto importante: não é um chatbot. A Fila Certa não responde sozinha. Ela decide a ordem, explica e prepara a resposta, mas quem envia é uma pessoa. Mell.

## Mell: slides 10 e 11

**Slide 10 (próximos passos)**
> Os próximos passos são: fazer um piloto com um lojista real, retreinar com as correções que a equipe marca no botão "A IA errou?", conectar o WhatsApp Business e comparar com um modelo pré-treinado em português.
>
> E as limitações que assumimos: a nota é um proxy, o treino foi com e-commerce de 2016 a 2018, o motivo é por regras de palavras e só 42% dos clientes escrevem comentário.

**Slide 11 (encerramento)**
> Fila Certa: a reclamação certa, no topo da fila. O código está no GitHub, pelo QR code. Obrigado!

---

Perguntas prováveis e respostas curtas: [`perguntas_pitch.md`](perguntas_pitch.md).
