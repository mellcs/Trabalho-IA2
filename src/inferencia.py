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

    def contribuicoes(self, txt):
        """Explicação por oclusão: tira uma palavra por vez e mede quanto a decisão muda.

        A mudança é medida no logit (antes da sigmoide), porque perto de 0% ou 100% a
        probabilidade quase não se mexe. Devolve, para cada token, a participação dele no
        total (soma dos valores absolutos = 1). Sinal +: empurra para "reclamação";
        sinal -: empurra para "não reclamação".
        """
        toks = texto.tokenizar(txt)[: self.voc.max_len]
        if len(toks) < 2:
            return [(t, 0.0) for t in toks]
        lg = lambda ps: np.log(np.clip(ps, 1e-7, 1 - 1e-7) / (1 - np.clip(ps, 1e-7, 1 - 1e-7)))
        base = lg(self.probabilidades([" ".join(toks)]))[0]
        variantes = [" ".join(toks[:i] + toks[i + 1:]) for i in range(len(toks))]
        efeito = base - lg(self.probabilidades(variantes))
        total = np.abs(efeito).sum()
        part = efeito / total if total > 1e-6 else np.zeros_like(efeito)
        return [(t, float(e)) for t, e in zip(toks, part)]

    def palavras_decisivas(self, txt, k=5):
        c = self.contribuicoes(txt)
        ordem = sorted(range(len(c)), key=lambda i: -abs(c[i][1]))[:k]
        return [c[i] for i in ordem if abs(c[i][1]) >= 0.05]

    def destacar(self, txt):
        """Divide o texto ORIGINAL em trechos com o peso de cada palavra (para colorir na tela)."""
        import re
        c = self.contribuicoes(txt)
        seg, pos, k = [], 0, 0
        for m in re.finditer(r"\w+", txt):
            if m.start() > pos:
                seg.append({"t": txt[pos:m.start()], "w": 0.0})
            n = len(texto.tokenizar(m.group(0)))
            w = sum(c[j][1] for j in range(k, min(k + n, len(c))))
            k += n
            seg.append({"t": m.group(0), "w": round(w, 4)})
            pos = m.end()
        if pos < len(txt):
            seg.append({"t": txt[pos:], "w": 0.0})
        return seg

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
