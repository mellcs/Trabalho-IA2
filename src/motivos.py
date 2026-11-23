"""Motivo provável da reclamação, por regras de palavras-chave.

Por que regras e não outra rede: não existe rótulo de "motivo" no dataset. Regras são
transparentes (o lojista vê por que o comentário caiu em "Entrega"), fáceis de ajustar
e foram conferidas com um dado independente: a situação real da entrega do pedido
(ver results/motivos.md). Um comentário pode citar mais de um motivo; o principal é o
que soma mais pontos: cada termo vale 1, e os termos de PESO_FORTE valem 2 ou 3
(empate: a ordem abaixo).

Os termos são comparados com o texto normalizado (minúsculas, sem acento).
"""
import re

from src.texto import normalizar

MOTIVOS = {
    "Entrega / atraso": [
        r"nao (me )?(chegou|chegaram|recebi|recebemos|foi entregue|entregue|entregaram|veio)",
        r"ainda nao", r"atra[sz]", r"nao chega", r"a espera", r"nao postou", r"postad", r"frete", r"prazo", r"demor", r"aguard", r"esperando", r"correio",
        r"transportadora", r"rastre", r"extravi", r"entreg", r"chegou so", r"so chegou",
    ],
    "Produto com defeito / qualidade": [
        r"quebrad", r"defeit", r"estragad", r"danific", r"nao funciona", r"parou de funcionar",
        r"nao liga", r"qualidade", r"rasgad", r"vazand", r"amassad", r"riscad", r"trincad",
        r"pessimo produto", r"fragil", r"descascando", r"mal acabad", r"furad", r"manchad",
        r"rasg", r"arranh", r"estrag", r"nao reconhec", r"vazamento", r"enferruj", r"solt",
    ],
    "Produto diferente / errado": [
        r"diferente", r"errad", r"outro produto", r"nao (e|era) o", r"nao corresponde",
        r"anunci", r"falsificad", r"falso", r"nao e original", r"cor ", r"tamanho", r"modelo",
        r"trocad", r"descricao", r"descrit", r"foto", r"sabor", r"marca", r"replica",
        r"enviaram (o|a|um|uma) ", r"mandaram (o|a|um|uma) ", r"medida", r"expectativa", r"enganad",
    ],
    "Pedido incompleto": [
        r"falt", r"incomplet", r"so veio", r"veio so", r"apenas (um|uma|1)", r"so (um|uma|1) ",
        r"veio (um|uma|1) ", r"recebi (apenas|so) ", r"comprei (2|dois|duas|3|tres)", r"de (dois|duas|2) ", r"comprado (2|dois|duas)",
        r"unidade", r"pecas?", r"somente", r"vazi[oa]", r"nao (esta|estava|veio) dentro", r"sem (o|a|os|as) ",
        r"e nao \d+", r"comprei \d+",
    ],
    "Atendimento / reembolso": [
        r"atendimento", r"resposta", r"respond", r"contato", r"\bsac\b", r"reembols", r"estorn",
        r"devolu", r"devolv", r"cancel", r"dinheiro", r"ressarc", r"descaso", r"ninguem",
        r"suporte", r"email", r"e mail", r"telefone", r"procon", r"reclame aqui", r"estoque",
        r"disponivel", r"nota fiscal", r"informac", r"vendedor", r"credibilidade", r"desleixo",
    ],
}
# termos que, sozinhos, já definem o motivo valem mais (ex.: "não chegou" é entrega mesmo
# que o cliente também diga que "ninguém responde")
PESO_FORTE = {
    r"nao (me )?(chegou|chegaram|recebi|recebemos|foi entregue|entregue|entregaram|veio)": 3,
    r"quebrad": 2, r"defeit": 2, r"parou de funcionar": 2, r"nao funciona": 2,
    r"diferente": 2, r"errad": 2, r"falsificad": 2,
    r"falt": 2, r"so veio": 2, r"veio so": 2,
    r"reembols": 2, r"estorn": 2,
}
OUTROS = "Outros / não identificado"
_RE = {m: [re.compile(t) for t in termos] for m, termos in MOTIVOS.items()}


def contagens(texto):
    s = " " + normalizar(texto) + " "
    return {m: sum(PESO_FORTE.get(r.pattern, 1) for r in regs if r.search(s)) for m, regs in _RE.items()}


def motivo(texto):
    c = contagens(texto)
    melhor = max(c.values())
    if melhor == 0:
        return OUTROS
    return next(m for m in MOTIVOS if c[m] == melhor)


def termos_encontrados(texto):
    """Lista os termos que dispararam (para mostrar no app e explicar a decisão)."""
    s = " " + normalizar(texto) + " "
    achados = []
    for m, regs in _RE.items():
        for r in regs:
            hit = r.search(s)
            if hit:
                achados.append((m, hit.group(0).strip()))
    return achados
