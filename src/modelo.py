"""Redes neurais para classificar comentários (PyTorch), treinadas do zero.

Duas arquiteturas, comparadas na validação:

BiLSTM (modelo principal)
    tokens -> Embedding(100) -> BiLSTM(64 por direção) -> pooling (máximo + média) -> Dropout -> Linear -> logit
    Por quê: a LSTM lê a frase em ordem e guarda contexto, então "não chegou" é diferente de
    "chegou". Bidirecional = lê da esquerda para a direita e ao contrário, e cada palavra
    "enxerga" o que vem antes e depois. O pooling resume a frase inteira em um vetor de
    tamanho fixo, qualquer que seja o tamanho do comentário.

CNN 1D (comparação)
    tokens -> Embedding(100) -> Conv1d com janelas de 2, 3 e 4 palavras -> max pooling -> Dropout -> Linear
    Por quê: cada filtro aprende a reconhecer uma expressão curta ("não recebi", "veio
    quebrado") em qualquer posição do texto. É mais rápida que a LSTM.

Perda: BCEWithLogitsLoss com pos_weight = (nº de positivos / nº de negativos no treino),
ou seja, cada erro na classe "reclamação" (33% da base) pesa cerca de 2x mais.
Saída: um logit; a sigmoide transforma em probabilidade de ser reclamação.
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)


class BiLSTM(nn.Module):
    def __init__(self, vocab, emb=100, oculto=64, dropout=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, oculto, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.saida = nn.Linear(4 * oculto, 1)  # 2 direções x (máximo + média)

    def forward(self, x):
        mascara = (x != 0).unsqueeze(-1).float()          # 1 nas palavras, 0 no preenchimento
        h, _ = self.lstm(self.drop(self.emb(x)))          # (lote, tokens, 2*oculto)
        maximo = (h - 1e4 * (1 - mascara)).max(dim=1).values
        media = (h * mascara).sum(1) / mascara.sum(1).clamp(min=1)
        return self.saida(self.drop(torch.cat([maximo, media], dim=1))).squeeze(1)


class CNN1D(nn.Module):
    def __init__(self, vocab, emb=100, filtros=100, janelas=(2, 3, 4), dropout=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb, filtros, k, padding=k - 1) for k in janelas])
        self.drop = nn.Dropout(dropout)
        self.saida = nn.Linear(filtros * len(janelas), 1)

    def forward(self, x):
        e = self.drop(self.emb(x)).transpose(1, 2)         # (lote, emb, tokens)
        z = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.saida(self.drop(torch.cat(z, dim=1))).squeeze(1)


ARQUITETURAS = {"bilstm": BiLSTM, "cnn": CNN1D}


def criar(nome, vocab):
    return ARQUITETURAS[nome](vocab)


def metricas(y, p, limiar=0.5):
    """Métricas da classe negativa (reclamação = 1)."""
    yhat = (p >= limiar).astype(int)
    return {
        "F1 (reclamação)": f1_score(y, yhat),
        "Recall (reclamação)": recall_score(y, yhat),
        "Precisão (reclamação)": precision_score(y, yhat),
        "Acurácia": accuracy_score(y, yhat),
        "PR-AUC": average_precision_score(y, p),
        "ROC-AUC": roc_auc_score(y, p),
    }


@torch.no_grad()
def prever(modelo, X, lote=1024):
    modelo.eval()
    out = [torch.sigmoid(modelo(torch.from_numpy(X[i:i + lote]))).numpy()
           for i in range(0, len(X), lote)]
    return np.concatenate(out)


def treinar(nome, vocab, X_tr, y_tr, X_va, y_va, epocas=15, paciencia=3, lote=128,
            lr=1e-3, semente=42, verbose=True):
    """Adam + BCE com peso de classe + early stopping pelo F1 da validação.

    Guarda os pesos da melhor época e para quando o F1 de validação não melhora por
    `paciencia` épocas seguidas (evita overfitting).
    """
    torch.manual_seed(semente)
    np.random.seed(semente)
    modelo = criar(nome, vocab)
    pos_weight = torch.tensor((y_tr == 0).sum() / (y_tr == 1).sum(), dtype=torch.float32)
    perda_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    Xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32))
    melhor, melhor_estado, sem_melhora, hist = -1, None, 0, []
    for ep in range(1, epocas + 1):
        t0 = time.time()
        modelo.train()
        ordem = torch.randperm(len(Xt))
        soma = 0.0
        for i in range(0, len(Xt), lote):
            idx = ordem[i:i + lote]
            opt.zero_grad()
            perda = perda_fn(modelo(Xt[idx]), yt[idx])
            perda.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)  # evita gradientes explosivos na LSTM
            opt.step()
            soma += perda.item() * len(idx)
        p_va = prever(modelo, X_va)
        with torch.no_grad():
            perda_va = perda_fn(torch.logit(torch.from_numpy(p_va).clamp(1e-6, 1 - 1e-6)),
                                torch.from_numpy(y_va.astype(np.float32))).item()
        m = metricas(y_va, p_va)
        hist.append({"epoca": ep, "perda_treino": soma / len(Xt), "perda_val": perda_va,
                     "f1_val": m["F1 (reclamação)"], "recall_val": m["Recall (reclamação)"]})
        if verbose:
            print(f"[{nome}] época {ep:2d}  perda treino {hist[-1]['perda_treino']:.4f}  "
                  f"perda val {perda_va:.4f}  F1 val {m['F1 (reclamação)']:.4f}  ({time.time() - t0:.0f}s)",
                  flush=True)
        if m["F1 (reclamação)"] > melhor:
            melhor, melhor_estado, sem_melhora = m["F1 (reclamação)"], copy.deepcopy(modelo.state_dict()), 0
        else:
            sem_melhora += 1
            if sem_melhora >= paciencia:
                break
    modelo.load_state_dict(melhor_estado)
    return modelo, hist
