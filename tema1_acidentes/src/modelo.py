"""Rede neural com embeddings para variáveis categóricas (mesma arquitetura do notebook)."""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


def dim_emb(n):
    return min(16, (n + 1) // 2)


class RedeAcidentes(nn.Module):
    def __init__(self, vocab, n_num, ocultas=(128, 64), dropout=0.3):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n, dim_emb(n)) for n in vocab])
        d_in = sum(dim_emb(n) for n in vocab) + n_num
        camadas = []
        for h in ocultas:
            camadas += [nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(dropout)]
            d_in = h
        camadas.append(nn.Linear(d_in, 1))
        self.mlp = nn.Sequential(*camadas)

    def forward(self, xc, xn):
        e = [emb(xc[:, i]) for i, emb in enumerate(self.embs)]
        return self.mlp(torch.cat(e + [xn], dim=1)).squeeze(1)  # logit


@torch.no_grad()
def prever(modelo, xc, xn, lote=8192):
    modelo.eval()
    xc, xn = torch.as_tensor(xc, dtype=torch.long), torch.as_tensor(xn, dtype=torch.float32)
    out = [torch.sigmoid(modelo(xc[i:i + lote], xn[i:i + lote])) for i in range(0, len(xc), lote)]
    return torch.cat(out).numpy()


def treinar(vocab, Xc_tr, Xn_tr, y_tr, Xc_va, Xn_va, y_va, epocas=15, paciencia=3,
            semente=42, lr=1e-3, verbose=True):
    """Adam + BCE sem pesos (Rede 2), early stopping pelo PR-AUC da validação."""
    torch.manual_seed(semente)
    np.random.seed(semente)
    m = RedeAcidentes(vocab, n_num=Xn_tr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    perda = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(semente)
    dl = DataLoader(TensorDataset(torch.as_tensor(Xc_tr, dtype=torch.long),
                                  torch.as_tensor(Xn_tr, dtype=torch.float32),
                                  torch.as_tensor(y_tr, dtype=torch.float32)),
                    batch_size=1024, shuffle=True, generator=g)
    melhor, estado, sem_melhora, hist = -1.0, None, 0, []
    for ep in range(1, epocas + 1):
        t0, soma = time.time(), 0.0
        m.train()
        for xc, xn, yb in dl:
            opt.zero_grad()
            l = perda(m(xc, xn), yb)
            l.backward()
            opt.step()
            soma += l.item() * len(yb)
        ap = average_precision_score(y_va, prever(m, Xc_va, Xn_va))
        hist.append({"epoca": ep, "custo_treino": soma / len(dl.dataset), "pr_auc_val": ap})
        if verbose:
            print(f"  época {ep:2d} | custo {hist[-1]['custo_treino']:.4f} | "
                  f"PR-AUC val {ap:.4f} | {time.time() - t0:.0f}s")
        if ap > melhor:
            melhor, estado, sem_melhora = ap, copy.deepcopy(m.state_dict()), 0
        else:
            sem_melhora += 1
            if sem_melhora >= paciencia:
                break
    m.load_state_dict(estado)
    return m, hist


def captura_topk(y, score, frac):
    k = int(len(score) * frac)
    return y[np.argsort(-score)[:k]].sum() / y.sum()


def metricas(y, score):
    return {"PR-AUC": average_precision_score(y, score), "ROC-AUC": roc_auc_score(y, score),
            "Captura top 10%": captura_topk(y, score, 0.10),
            "Captura top 20%": captura_topk(y, score, 0.20),
            "Captura top 30%": captura_topk(y, score, 0.30)}
