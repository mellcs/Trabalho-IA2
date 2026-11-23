"""Carrega o modelo salvo em models/ e analisa comentários (usado pelo app e pelo notebook)."""
import json
from pathlib import Path

import numpy as np
import torch

from src import motivos, texto
from src.modelo import criar, prever


class Analisador:
    def __init__(self, pasta="models"):
        p = Path(pasta)
        self.cfg = json.load(open(p / "config.json", encoding="utf-8"))
        self.voc = texto.Vocabulario.de_dict(json.load(open(p / "vocab.json", encoding="utf-8")))
        self.rede = criar(self.cfg["arquitetura"], len(self.voc))
        self.rede.load_state_dict(torch.load(p / "rede.pt", map_location="cpu"))
        self.rede.eval()
        self.lim = self.cfg["limiares_prioridade"]

    def probabilidades(self, textos):
        return prever(self.rede, self.voc.codificar(list(textos)))

    def prioridade(self, p):
        if p >= self.lim["alta"]:
            return "Alta"
        if p < self.lim["baixa"]:
            return "Baixa"
        return "Média"

    def palavras_decisivas(self, txt, k=5):
        """Explicação por oclusão: tira uma palavra por vez e mede quanto a decisão muda.

        A mudança é medida no logit (antes da sigmoide), porque perto de 0% ou 100% a
        probabilidade quase não se mexe. O resultado é a participação de cada palavra no
        total (soma dos valores absolutos = 100%). Sinal +: empurra para "reclamação";
        sinal -: empurra para "não reclamação".
        """
        toks = texto.tokenizar(txt)[: self.voc.max_len]
        if len(toks) < 2:
            return []
        lg = lambda ps: np.log(np.clip(ps, 1e-7, 1 - 1e-7) / (1 - np.clip(ps, 1e-7, 1 - 1e-7)))
        base = lg(self.probabilidades([" ".join(toks)]))[0]
        variantes = [" ".join(toks[:i] + toks[i + 1:]) for i in range(len(toks))]
        efeito = base - lg(self.probabilidades(variantes))
        total = np.abs(efeito).sum()
        if total < 1e-6:
            return []
        part = efeito / total
        ordem = np.argsort(-np.abs(part))[:k]
        return [(toks[i], float(part[i])) for i in ordem if abs(part[i]) >= 0.05]

    def analisar(self, txt):
        p = float(self.probabilidades([txt])[0])
        reclamacao = p >= self.cfg["limiar"]
        return {
            "probabilidade": p,
            "reclamacao": bool(reclamacao),
            "prioridade": self.prioridade(p),
            "motivo": motivos.motivo(txt) if reclamacao else "—",
            "termos": motivos.termos_encontrados(txt) if reclamacao else [],
            "palavras": self.palavras_decisivas(txt),
        }

    def analisar_lote(self, textos):
        textos = [str(t) for t in textos]
        ps = self.probabilidades(textos)
        lin = []
        for t, p in zip(textos, ps):
            rec = p >= self.cfg["limiar"]
            lin.append({"comentario": t, "prob_reclamacao": round(float(p), 3),
                        "reclamacao": "Sim" if rec else "Não", "prioridade": self.prioridade(p),
                        "motivo": motivos.motivo(t) if rec else "—"})
        return lin
