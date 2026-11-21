"""Painel de risco de acidentes graves: interface Gradio para gestores de trânsito.

Uso:
    python app.py            # abre em http://127.0.0.1:7860
    python app.py --share    # gera também um link público temporário

Precisa só da pasta modelos/ (gerada por `python -m src.treinar`); não precisa do dataset.
"""
import argparse
import json
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.modelo import RedeAcidentes

PASTA = Path(__file__).parent / "modelos"

MESES = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
         "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]
DIAS = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]

PT = {
    "Light_Conditions": {"Daylight": "Dia (luz natural)",
                         "Darkness - lights lit": "Noite, com iluminação pública acesa",
                         "Darkness - no lighting": "Noite, sem iluminação pública",
                         "Darkness - lights unlit": "Noite, iluminação existente apagada",
                         "Darkness - lighting unknown": "Noite, iluminação desconhecida"},
    "Road_Type": {"Single carriageway": "Pista simples", "Dual carriageway": "Pista dupla",
                  "Roundabout": "Rotatória", "One way street": "Rua de mão única",
                  "Slip road": "Alça de acesso"},
    "Road_Surface_Conditions": {"Dry": "Seca", "Wet or damp": "Molhada ou úmida",
                                "Frost or ice": "Geada ou gelo", "Snow": "Neve",
                                "Flood over 3cm. deep": "Alagada (mais de 3 cm)"},
    "Weather_Conditions": {"Fine no high winds": "Tempo bom, sem ventos fortes",
                           "Fine + high winds": "Tempo bom, com ventos fortes",
                           "Raining no high winds": "Chuva, sem ventos fortes",
                           "Raining + high winds": "Chuva com ventos fortes",
                           "Snowing no high winds": "Neve, sem ventos fortes",
                           "Snowing + high winds": "Neve com ventos fortes",
                           "Fog or mist": "Neblina", "Other": "Outro"},
    "Urban_or_Rural_Area": {"Urban": "Urbana", "Rural": "Rural"},
}
NOME_FATOR = {"Light_Conditions": "Iluminação", "Road_Type": "Tipo de via",
              "Road_Surface_Conditions": "Superfície da pista",
              "Weather_Conditions": "Clima", "Urban_or_Rural_Area": "Área"}


# ---------------------------------------------------------------- modelo
class Preditor:
    def __init__(self, pasta=PASTA):
        self.art = json.load(open(pasta / "artefatos.json", encoding="utf-8"))
        a = self.art
        self.rede = RedeAcidentes(a["vocab"], n_num=len(a["num"]))
        self.rede.load_state_dict(torch.load(pasta / "rede_final.pt", map_location="cpu"))
        self.rede.eval()
        self.cortes = np.array(a["cortes"])
        self.celulas = pd.read_csv(pasta / "celulas_risco.csv")
        self.ranking = pd.read_csv(pasta / "distritos_risco.csv")
        self.distritos = sorted(a["distritos"])

    def _linha(self, distrito, luz, via, area, superficie, clima, mes, dia):
        a = self.art
        loc = a["distritos"][distrito]
        valores = {"Light_Conditions": luz, "Road_Surface_Conditions": superficie,
                   "Road_Type": via, "Urban_or_Rural_Area": area,
                   "Weather_Conditions": clima, "Vehicle_Type": "Car",  # sem efeito no modelo
                   "District Area": distrito}
        xc = [a["maps"][c].get(valores[c], 0) for c in a["cats"]]
        m, dw = MESES.index(mes) + 1, DIAS.index(dia)
        xn = [(loc["lat"] - a["mu"]["Latitude"]) / a["sd"]["Latitude"],
              (loc["lon"] - a["mu"]["Longitude"]) / a["sd"]["Longitude"],
              np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12),
              np.sin(2 * np.pi * dw / 7), np.cos(2 * np.pi * dw / 7), float(dw >= 5)]
        return xc, xn

    @torch.no_grad()
    def probabilidades(self, linhas):
        xc = torch.tensor([l[0] for l in linhas], dtype=torch.long)
        xn = torch.tensor([l[1] for l in linhas], dtype=torch.float32)
        return torch.sigmoid(self.rede(xc, xn)).numpy()

    def prob(self, *cenario):
        return float(self.probabilidades([self._linha(*cenario)])[0])

    def faixa(self, p):
        return int(np.searchsorted(self.cortes, p)) + 1


P = None


def preditor():
    global P
    if P is None:
        P = Preditor()
    return P


def opcoes(coluna):
    return [(pt, en) for en, pt in PT[coluna].items()]


# ---------------------------------------------------------------- aba 1: consultar
def consultar(distrito, luz, via, area, superficie, clima, mes, dia):
    pr = preditor()
    if not distrito:
        return "Escolha um distrito.", None
    cenario = [distrito, luz, via, area, superficie, clima, mes, dia]
    p = pr.prob(*cenario)
    fx = pr.faixa(p)
    media = pr.art["media_geral"]
    real = pr.art["taxa_decil"][str(fx)] if str(fx) in pr.art["taxa_decil"] else pr.art["taxa_decil"][fx]
    nivel = "ALTO" if fx >= 9 else ("MODERADO" if fx >= 6 else "BAIXO")
    texto = (f"## Risco {nivel}: {p:.1%} de chance de o acidente ser grave ou fatal\n"
             f"- Média de todos os acidentes: **{media:.1%}** (este cenário é "
             f"**{p / media:.1f}x** a média)\n"
             f"- Faixa de risco **{fx} de 10**. Nos dados de validação, **{real:.1%}** dos "
             f"acidentes desta faixa foram graves ou fatais.\n\n"
             "_Estimativa de risco a partir das condições, para priorização. "
             "Não é a previsão de um acidente específico._")

    # Simulação: troca um fator por vez e mede a variação
    pos = {"Light_Conditions": 1, "Road_Type": 2, "Urban_or_Rural_Area": 3,
           "Road_Surface_Conditions": 4, "Weather_Conditions": 5}
    alternativas, rotulos = [], []
    for col, i in pos.items():
        for en, pt in PT[col].items():
            if en == cenario[i]:
                continue
            c2 = list(cenario)
            c2[i] = en
            alternativas.append(pr._linha(*c2))
            rotulos.append((NOME_FATOR[col], pt))
    ps = pr.probabilidades(alternativas)
    sim = pd.DataFrame(rotulos, columns=["Fator", "Novo valor"])
    sim["Risco"] = [f"{x:.1%}" for x in ps]
    sim["Variação (p.p.)"] = np.round((ps.astype(float) - p) * 100, 1)
    sim = sim.sort_values("Variação (p.p.)").reset_index(drop=True)
    return texto, sim


# ---------------------------------------------------------------- aba 2: mapa e ranking
def mapa(distrito=None):
    pr = preditor()
    mp = pr.celulas
    fig, ax = plt.subplots(figsize=(6, 7.5))
    sc = ax.scatter(mp["lon"], mp["lat"], c=mp["taxa"] * 100, s=np.clip(np.sqrt(mp["n"]), 4, 60),
                    cmap="YlOrRd", alpha=0.85, vmin=mp["taxa"].quantile(0.05) * 100,
                    vmax=mp["taxa"].quantile(0.95) * 100)
    fig.colorbar(sc, ax=ax, label="% de acidentes graves/fatais (taxa alisada)")
    if distrito and distrito in pr.art["distritos"]:
        loc = pr.art["distritos"][distrito]
        ax.scatter([loc["lon"]], [loc["lat"]], marker="*", s=400, c="#1f4e79",
                   edgecolors="white", linewidths=1.5, zorder=5, label=distrito)
        ax.legend(loc="upper left")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Onde os acidentes tendem a ser mais graves\n(células de ~5 km)", fontsize=11)
    ax.set_aspect(1.6)
    fig.tight_layout()
    return fig


def ranking(minimo, quantos):
    r = preditor().ranking
    r = r[r["acidentes"] >= minimo].head(int(quantos)).reset_index(drop=True)
    r.insert(0, "Posição", range(1, len(r) + 1))
    r["taxa_alisada"] = (r["taxa_alisada"] * 100).round(1)
    return r.rename(columns={"District Area": "Distrito", "acidentes": "Acidentes",
                             "graves": "Graves/fatais", "taxa_alisada": "% graves (alisada)"})


# ---------------------------------------------------------------- aba 3: priorizar lista
COLUNAS_LISTA = ["local", "distrito", "iluminacao", "tipo_via", "area", "superficie",
                 "clima", "mes", "dia_semana"]


def _valor(col, v):
    """Aceita o valor original (inglês) ou o rótulo em português."""
    v = str(v).strip()
    if col in PT:
        reverso = {pt.lower(): en for en, pt in PT[col].items()}
        return reverso.get(v.lower(), v)
    return v


def priorizar(arquivo):
    pr = preditor()
    if arquivo is None:
        return None, None, "Envie um CSV (use o modelo de exemplo)."
    caminho = arquivo if isinstance(arquivo, str) else arquivo.name
    lista = pd.read_csv(caminho)
    faltando = [c for c in COLUNAS_LISTA if c not in lista.columns]
    if faltando:
        return None, None, f"Colunas faltando no CSV: {', '.join(faltando)}"
    erros, linhas = [], []
    for i, r in lista.iterrows():
        mes = MESES[int(r["mes"]) - 1] if str(r["mes"]).isdigit() else str(r["mes"]).capitalize()
        dia = str(r["dia_semana"]).capitalize()
        if r["distrito"] not in pr.art["distritos"]:
            erros.append(f"linha {i + 2}: distrito '{r['distrito']}' não encontrado")
            continue
        if mes not in MESES or dia not in DIAS:
            erros.append(f"linha {i + 2}: mês ou dia inválido")
            continue
        linhas.append((i, pr._linha(r["distrito"], _valor("Light_Conditions", r["iluminacao"]),
                                    _valor("Road_Type", r["tipo_via"]),
                                    _valor("Urban_or_Rural_Area", r["area"]),
                                    _valor("Road_Surface_Conditions", r["superficie"]),
                                    _valor("Weather_Conditions", r["clima"]), mes, dia)))
    if not linhas:
        return None, None, "Nenhuma linha válida.\n" + "\n".join(erros)
    ps = pr.probabilidades([l for _, l in linhas])
    res = lista.loc[[i for i, _ in linhas]].copy()
    res["risco_%"] = np.round(ps.astype(float) * 100, 1)
    res["faixa_1a10"] = [pr.faixa(p) for p in ps]
    res = res.sort_values("risco_%", ascending=False).reset_index(drop=True)
    res.insert(0, "prioridade", range(1, len(res) + 1))
    primeiras = ["prioridade", "local", "risco_%", "faixa_1a10"]
    res = res[primeiras + [c for c in res.columns if c not in primeiras]]
    saida = Path(tempfile.gettempdir()) / "lista_priorizada.csv"
    res.to_csv(saida, index=False)
    msg = f"{len(res)} itens priorizados."
    if erros:
        msg += " Ignorados:\n" + "\n".join(erros)
    return res, str(saida), msg


# ---------------------------------------------------------------- interface
def sobre_md():
    a = preditor().art
    m = a["metricas_teste"]
    return f"""
### Como funciona
Uma **rede neural (PyTorch)** com embeddings para as variáveis categóricas foi treinada com
cerca de 460 mil acidentes reais do Reino Unido. Ela estima a probabilidade de um acidente
ser **grave ou fatal** a partir de: distrito e localização, iluminação, tipo de via,
área urbana/rural, superfície da pista, clima, mês e dia da semana.

### Desempenho (conjunto de teste, nunca usado no treino)
| Métrica | Valor | Referência ao acaso |
|---|---|---|
| ROC-AUC | {m['ROC-AUC']:.3f} | 0,500 |
| PR-AUC | {m['PR-AUC']:.3f} | {a['media_geral']:.3f} |
| Graves/fatais capturados nos 10% de maior risco | {m['Captura top 10%']:.1%} | 10% |
| Graves/fatais capturados nos 20% de maior risco | {m['Captura top 20%']:.1%} | 20% |

### Limites
- O sinal é fraco (ROC-AUC ~0,62): o dataset não tem velocidade, idade ou uso de cinto. Serve
  para **priorizar**, não para prever um acidente.
- Associação não é causa: a simulação mostra como o risco estimado muda, não garante o efeito
  de uma obra.
- Os dados são de 2007 a 2010 (o dataset do Kaggle mostra os anos deslocados para 2019-2022).
- A localização usada é o centro dos acidentes do distrito.
"""


def criar_app():
    pr = preditor()
    padrao = "Birmingham" if "Birmingham" in pr.distritos else pr.distritos[0]
    with gr.Blocks(title="Risco de acidentes graves") as demo:
        gr.Markdown("# Painel de risco de acidentes graves de trânsito\n"
                    "Apoio à decisão para gestores de trânsito: **onde e em quais condições** "
                    "um acidente tende a ser mais grave, para priorizar fiscalização, "
                    "iluminação e obras.")
        with gr.Tab("1. Consultar e simular"):
            with gr.Row():
                with gr.Column(scale=1):
                    distrito = gr.Dropdown(pr.distritos, value=padrao, label="Distrito")
                    luz = gr.Dropdown(opcoes("Light_Conditions"), value="Daylight", label="Iluminação")
                    via = gr.Dropdown(opcoes("Road_Type"), value="Single carriageway", label="Tipo de via")
                    area = gr.Dropdown(opcoes("Urban_or_Rural_Area"), value="Urban", label="Área")
                    sup = gr.Dropdown(opcoes("Road_Surface_Conditions"), value="Dry",
                                      label="Superfície da pista")
                    clima = gr.Dropdown(opcoes("Weather_Conditions"), value="Fine no high winds",
                                        label="Clima")
                    mes = gr.Dropdown(MESES, value="Julho", label="Mês")
                    dia = gr.Dropdown(DIAS, value="Terça", label="Dia da semana")
                    botao = gr.Button("Calcular risco", variant="primary")
                with gr.Column(scale=1):
                    saida = gr.Markdown()
                    gr.Markdown("#### E se mudasse um fator? (mesmo cenário, trocando um item por vez)")
                    sim = gr.Dataframe(interactive=False)
            entradas = [distrito, luz, via, area, sup, clima, mes, dia]
            botao.click(consultar, entradas, [saida, sim])
            rural = "Aberdeenshire" if "Aberdeenshire" in pr.distritos else padrao
            gr.Examples([[rural, "Darkness - no lighting", "Single carriageway", "Rural",
                          "Wet or damp", "Raining no high winds", "Dezembro", "Sábado"],
                         [padrao, "Daylight", "Roundabout", "Urban", "Dry",
                          "Fine no high winds", "Julho", "Terça"]], entradas)
        with gr.Tab("2. Mapa e ranking de distritos"):
            with gr.Row():
                with gr.Column():
                    d_mapa = gr.Dropdown(pr.distritos, value=padrao, label="Destacar distrito no mapa")
                    fig = gr.Plot()
                with gr.Column():
                    minimo = gr.Slider(100, 5000, value=1000, step=100,
                                       label="Mínimo de acidentes no distrito")
                    quantos = gr.Slider(5, 50, value=15, step=5, label="Quantos mostrar")
                    tab = gr.Dataframe(interactive=False)
            d_mapa.change(mapa, d_mapa, fig)
            minimo.change(ranking, [minimo, quantos], tab)
            quantos.change(ranking, [minimo, quantos], tab)
            demo.load(mapa, d_mapa, fig)
            demo.load(ranking, [minimo, quantos], tab)
        with gr.Tab("3. Priorizar uma lista"):
            gr.Markdown("Envie um CSV com vários locais/situações (por exemplo, trechos que a "
                        "prefeitura quer avaliar). O painel devolve a lista **ordenada por risco**. "
                        "Colunas: `" + "`, `".join(COLUNAS_LISTA) + "`. "
                        "Veja `docs/exemplo_lista.csv`.")
            arq = gr.File(label="CSV de entrada", file_types=[".csv"])
            exemplo = Path(__file__).parent / "docs" / "exemplo_lista.csv"
            if exemplo.exists():
                gr.Examples([[str(exemplo)]], [arq])
            b2 = gr.Button("Priorizar", variant="primary")
            msg = gr.Markdown()
            tab2 = gr.Dataframe(interactive=False)
            baixar = gr.File(label="Baixar lista priorizada")
            b2.click(priorizar, arq, [tab2, baixar, msg])
        with gr.Tab("Sobre o modelo"):
            gr.Markdown(sobre_md())
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="gera link público temporário")
    ap.add_argument("--porta", type=int, default=7860)
    a = ap.parse_args()
    criar_app().launch(share=a.share, server_port=a.porta)
