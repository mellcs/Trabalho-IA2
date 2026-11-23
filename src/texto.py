"""Normalização, tokenização e vocabulário.

Decisões:
- Minúsculas e remoção de acentos: os clientes escrevem "nao", "não", "NÃO". Sem acento,
  as três formas viram o mesmo token, e o vocabulário fica menor e mais robusto.
- Mantemos palavras e números; pontuação vira separador. "não" é mantido (sem lista de
  stopwords), porque a negação muda o sentido: "chegou" x "não chegou".
- Vocabulário montado só com o treino; palavra que aparece menos de 2 vezes vira <UNK>.
  Índice 0 = <PAD> (preenchimento), 1 = <UNK> (palavra desconhecida).
"""
import re
import unicodedata
from collections import Counter

import numpy as np

PAD, UNK = 0, 1
_TOKEN = re.compile(r"[a-z0-9]+")


def normalizar(s):
    s = unicodedata.normalize("NFKD", str(s).lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(_TOKEN.findall(s))


def tokenizar(s):
    return normalizar(s).split()


class Vocabulario:
    def __init__(self, min_freq=2, max_len=60):
        self.min_freq = min_freq
        self.max_len = max_len  # 60 tokens cobrem mais de 99% dos comentários
        self.itos = ["<PAD>", "<UNK>"]
        self.stoi = {}

    def ajustar(self, textos):
        cont = Counter(t for s in textos for t in tokenizar(s))
        self.itos += sorted(w for w, c in cont.items() if c >= self.min_freq)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        return self

    def __len__(self):
        return len(self.itos)

    def codificar(self, textos):
        """Lista de textos -> matriz (n, max_len) de índices, com zeros no fim."""
        X = np.zeros((len(textos), self.max_len), dtype=np.int64)
        for i, s in enumerate(textos):
            ids = [self.stoi.get(t, UNK) for t in tokenizar(s)][: self.max_len]
            X[i, : len(ids)] = ids
        return X

    def para_dict(self):
        return {"min_freq": self.min_freq, "max_len": self.max_len, "itos": self.itos}

    @classmethod
    def de_dict(cls, d):
        v = cls(d["min_freq"], d["max_len"])
        v.itos = d["itos"]
        v.stoi = {w: i for i, w in enumerate(v.itos)}
        return v
