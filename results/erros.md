# Análise de erros (conjunto de teste)

Rede bilstm, limiar 0,5. P = probabilidade de ser reclamação.

## Falsos positivos mais confiantes (o modelo disse reclamação, a nota foi 4 ou 5)

| P(rede) | P(baseline) | Nota real | Comentário |
|---|---|---|---|
| 1.00 | 1.00 | 4 | Não recebi o produto, enviei mensagem não me responderam, liguei estou esperando até agora o retorno |
| 1.00 | 1.00 | 4 | Até agora não recebi o produto. Ninguém me dá um retorno |
| 1.00 | 0.99 | 4 | Não comprei esse produto e nem e o numero do meu cartão. Aguardo retorno |
| 1.00 | 0.98 | 5 | Estou achando muito demorado há entrega, ficaram de me entregar até o dia 02/02/18 e até hoje não chegou |
| 1.00 | 1.00 | 4 | Enviaram o produto errado pra mim! Na loja virtual estava mostrando o produto original, eu comprei, e chegou um produto totalmente diferente! Quero outro, ou meu dinheiro de volta!!! |
| 1.00 | 0.99 | 4 | comprei tres produtos so recebi um ate agora e os outros dois cade |
| 1.00 | 0.96 | 4 | comprei 02 unidades (teria que receber 20 panos de prato barrado), mas recebi somente 01 unidade (recebi 10 panos de prato barrado.) Dando é que paguei o valor de 95,80. |
| 1.00 | 0.99 | 5 | Não foi entregue o meu pedido |
| 1.00 | 0.95 | 5 | Veio uma tesoura na medida 5.5 e a outra veio 6. 0 Já tinha adquirido outra vez as tesouras mas desta vez veio errada cada uma de uma medida e o estojo veio com uma qualidade inferior a outra vez |
| 1.00 | 0.97 | 4 | Foi me entregue 1 produto sendo que eu comprei 2 produtos do mesmo . Ainda esperando a entrega se não ocorrerá ida a justiça. |

## Falsos negativos mais confiantes (o modelo disse não reclamação, a nota foi 1 ou 2)

| P(rede) | P(baseline) | Nota real | Comentário |
|---|---|---|---|
| 0.00 | 0.01 | 2 | Linda.entrega rápida... parabéns |
| 0.00 | 0.00 | 2 | muito bom gostei do atendimento veio bem antes do prazo |
| 0.00 | 0.03 | 1 | Super recomendo lannister. Entrega tudo no prazo. 🖒🖒👏👏👏 |
| 0.00 | 0.02 | 1 | Bom produto: superou as minhas espectativas |
| 0.00 | 0.00 | 1 | Ótima loija produto entregue dentro do prazo,Produto muito bom |
| 0.00 | 0.02 | 1 | Tudo de acordo com o combinado. |
| 0.00 | 0.02 | 1 | Minha mercadoria chegou antes do prazo e é exelente |
| 0.00 | 0.08 | 1 | pena que é muito pouco o produto e o frete é muito caro |
| 0.00 | 0.09 | 1 | Muito prazo e nenhum cumprido |
| 0.00 | 0.21 | 2 | Pois por baratheon é mais rápido.  Por exemplo  Na compra no baratheon agente compra hoje.  Exemplo.   Na sexta-feira chega o produto que agente comprou de Essa loja chegou faiz na data que foi co |

## Onde rede e baseline discordam

A rede acerta e o baseline erra em 79 comentários; o contrário acontece em 85.

### Exemplos em que a rede acertou e o baseline errou

| P(rede) | P(baseline) | Nota real | Comentário |
|---|---|---|---|
| 0.51 | 0.37 | 1 | minha opinião é que vocês deviam acompanhar semanalmente ou quinzenalmente a entrega |
| 0.43 | 0.52 | 5 | Enfico a qualquer um de meus familiares e amigos |
| 0.43 | 0.70 | 4 | acuso positivamente o recebimento do produto. |
| 0.44 | 0.80 | 5 | Apenas o repositor para o produto em questão veio errado, veio um para tintas de um cartucho diferente do modelo que usamos (920XL) mas contornei com aplicação direta. Muito bom produto. Recomendo!!! |
| 0.81 | 0.47 | 2 | Pelo valor dele eu esperava um produto melhor. |
| 0.08 | 0.50 | 4 | Só foi feita uma tentativa de entrega, tive que retirar no correio e em um prazo também bem apertado |
| 0.20 | 0.79 | 5 | bom serviço ainda estou sperando a tinta preta que nao chegou |
| 0.41 | 0.53 | 5 | porém os correios deixa a desejar... se tivessem uma trasportadora para fazer entregas seria melhor... |
| 0.31 | 0.61 | 5 | Fiquei satisfeita com a entrega bem antes do prazo, agora o produto (tecido) não surpreendeu, deixou a desejar, mas, não é de todo ruim. Descartável, mas, por hora aproveitável. |
| 0.73 | 0.35 | 1 | Pinceis não atendeu minha expectativas.. |

### Exemplos em que o baseline acertou e a rede errou

| P(rede) | P(baseline) | Nota real | Comentário |
|---|---|---|---|
| 0.63 | 0.45 | 5 | Recomendo inclusive minhas colegas já pediram estão aguardando a entrega fiquei muito contente vou fazer mais pedidos novos nos próximos meses passei para minhas filhas aliás a família toda . |
| 0.54 | 0.14 | 4 | A loja postou o produto de e mediato o correio demorou um pouco |
| 0.08 | 0.59 | 1 | Não posso avaliar por enquanto, o produto foi postado e ocorreu uma greve no Pais. Obrigado. |
| 0.21 | 0.56 | 2 | Comprei o relógio com a borda azul, conforme o anúncio aqui no site e chegou com a borda amarela e botões. |
| 0.41 | 0.61 | 1 | O descrição do do produto no site .sobre o tamanho não confere com produto recebido....muito pequena ..... |
| 0.41 | 0.77 | 2 | FALTOU CADEIRAS |
| 0.42 | 0.79 | 1 | A targaryen é um intermediário entre lojas de e-commerce e o consumidor. TOTALMENTE desorganizada. DOU NOTA ZERO a targaryen. |
| 0.49 | 0.81 | 1 | Como faço pra conferir aonde está meu produto? |
| 0.25 | 0.56 | 1 | Pois a tela do encosto da cadeira estava rasgada. |
| 0.21 | 0.79 | 1 | Tive que receber o produto ... Pois não acataram o meu cancelamento feito antes da emissão da nota.... Então vou pensar duas vezes agora antes de comprar pelo site |