"""O que fazer com cada mensagem: prazo de resposta, ação interna e resposta sugerida.

É a parte "de negócio" do produto: a rede diz SE é reclamação e com que urgência; as
regras de motivo dizem O QUÊ; este módulo transforma isso em uma ação para a equipe.
Os textos são modelos para o atendente revisar antes de enviar (não são enviados sozinhos).
"""

PRAZO = {
    "Alta": "Responder em até 2 horas",
    "Média": "Responder em até 24 horas",
    "Baixa": "Agradecimento automático",
}

ACAO = {
    "Entrega / atraso": "Consultar o rastreio com a transportadora e informar a nova previsão.",
    "Produto com defeito / qualidade": "Abrir troca com coleta e registrar o defeito para o fornecedor.",
    "Produto diferente / errado": "Enviar o item correto, agendar a coleta e revisar o anúncio.",
    "Pedido incompleto": "Conferir a separação do pedido e enviar o que faltou com prioridade.",
    "Atendimento / reembolso": "Dar retorno sobre o reembolso ou cancelamento ainda hoje.",
    "Outros / não identificado": "Ler a mensagem e pedir mais detalhes ao cliente.",
    "—": "Nenhuma ação necessária.",
}

RESPOSTA = {
    "Entrega / atraso": (
        "Olá! Sentimos muito pela demora na entrega do seu pedido. Já acionamos a transportadora "
        "para localizar o pacote e vamos te passar o rastreio atualizado em até 24 horas. Se preferir, "
        "podemos reenviar o produto ou devolver o valor. Conte com a gente!"),
    "Produto com defeito / qualidade": (
        "Olá! Lamentamos que o produto tenha chegado com problema. Pode nos mandar uma foto por aqui? "
        "Assim que recebermos, já providenciamos a troca sem custo, com coleta no seu endereço."),
    "Produto diferente / errado": (
        "Olá! Pedimos desculpas por enviar um item diferente do que você comprou. Vamos enviar o produto "
        "correto e agendar a coleta do item errado, sem nenhum custo para você."),
    "Pedido incompleto": (
        "Olá! Sentimos muito, seu pedido chegou incompleto. Pode confirmar quais itens faltaram? "
        "Enviamos o restante com prioridade e te passamos o rastreio."),
    "Atendimento / reembolso": (
        "Olá! Pedimos desculpas pela demora no retorno. Seu caso já está com a nossa equipe e você recebe "
        "uma resposta definitiva sobre o reembolso ainda hoje."),
    "Outros / não identificado": (
        "Olá! Obrigado por nos contar e sentimos muito pela experiência. Pode nos dar mais detalhes? "
        "Queremos resolver o quanto antes."),
    "—": "Olá! Que bom que deu tudo certo. Obrigado pela confiança e volte sempre!",
}


def plano(prioridade, motivo):
    m = motivo if motivo in RESPOSTA else "Outros / não identificado"
    return {"prazo": PRAZO[prioridade], "acao": ACAO[m], "resposta": RESPOSTA[m]}
