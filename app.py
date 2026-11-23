"""Fila Certa: servidor do app (FastAPI) + interface web em web/.

Rodar:  python app.py                 abre em http://127.0.0.1:7860
        python app.py --porta 8000    outra porta
        python app.py --host 0.0.0.0  acessível por outros aparelhos da rede (ex.: celular no Wi-Fi)

Só lê os arquivos de models/; não precisa do dataset nem de retreinar.

API (a mesma que a interface usa, pronta para integrar com outros sistemas):
  POST /api/analisar  {"texto": "..."}                       -> análise completa de uma mensagem
  POST /api/lote      {"mensagens": [{"texto": "...", "canal": "...", "cliente": "..."}]}
  GET  /api/caixa     caixa de entrada de demonstração já analisada
  GET  /api/painel    histórico mês a mês, motivos e alertas
  GET  /api/config    métricas do modelo e limiares de prioridade
"""
import argparse
import json
import threading
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.inferencia import Analisador
from src.respostas import plano

AQUI = Path(__file__).parent
MODELOS, WEB = AQUI / "models", AQUI / "web"
A = Analisador(MODELOS)
LIMITE_LOTE = 2000          # mensagens por chamada de /api/lote
LIMITE_TEXTO = 2000         # caracteres por mensagem

app = FastAPI(title="Fila Certa", description="Triagem de reclamações de clientes com rede neural", version="1.0")


class Mensagem(BaseModel):
    texto: str
    canal: str | None = None
    cliente: str | None = None
    data: str | None = None
    nota: int | None = None
    id: str | None = None


class Lote(BaseModel):
    mensagens: list[Mensagem]


def resumo(m, p):
    """Campos que a lista da caixa de entrada precisa (sem a explicação palavra a palavra)."""
    rec = p >= A.cfg["limiar"]
    pr = A.prioridade(p)
    from src import motivos
    mot = motivos.motivo(m["texto"]) if rec else "—"
    return {**m, "probabilidade": round(float(p), 4), "reclamacao": bool(rec), "prioridade": pr,
            "motivo": mot, **plano(pr, mot)}


def analisar_varias(msgs):
    ps = A.probabilidades([m["texto"][:LIMITE_TEXTO] for m in msgs])
    return [resumo(m, p) for m, p in zip(msgs, ps)]


CAIXA = analisar_varias(json.load(open(MODELOS / "caixa_demo.json", encoding="utf-8")))
PAINEL = json.load(open(MODELOS / "painel.json", encoding="utf-8"))


@app.get("/api/config")
def config():
    return {"arquitetura": A.cfg["arquitetura"], "limiares": A.lim, "metricas_teste": A.cfg["metricas_teste"],
            "faixas_teste": A.cfg["faixas_teste"], "vocabulario": len(A.voc)}


@app.get("/api/caixa")
def caixa():
    return CAIXA


@app.get("/api/painel")
def painel():
    return PAINEL


@app.post("/api/analisar")
def analisar(m: Mensagem):
    txt = m.texto.strip()[:LIMITE_TEXTO]
    if not txt:
        raise HTTPException(400, "texto vazio")
    r = A.analisar(txt)
    base = resumo({**m.model_dump(), "texto": txt}, r["probabilidade"])
    base["destaques"] = A.destacar(txt)
    base["palavras"] = r["palavras"]
    base["termos"] = sorted({t for _, t in r["termos"]})
    return base


@app.post("/api/lote")
def lote(l: Lote):
    msgs = [x.model_dump() for x in l.mensagens if x.texto and x.texto.strip()][:LIMITE_LOTE]
    for x in msgs:
        x["texto"] = x["texto"].strip()
    return analisar_varias(msgs) if msgs else []


app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def inicio():
    return FileResponse(WEB / "index.html")


def iniciar_em_thread(porta=7860):
    """Para o Colab: sobe o servidor em segundo plano (ver notebook)."""
    t = threading.Thread(target=uvicorn.run, kwargs={"app": app, "host": "127.0.0.1", "port": porta,
                                                       "log_level": "warning"}, daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--porta", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--sem-navegador", action="store_true")
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.porta}"
    print(f"\n  Fila Certa rodando em {url}  (Ctrl+C para parar)\n")
    if not a.sem_navegador:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=a.host, port=a.porta, log_level="warning")
