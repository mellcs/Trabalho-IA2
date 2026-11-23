"""Fila Certa: triagem de reclamações de clientes (app de demonstração em Gradio).

Rodar:  python app.py          (abre em http://127.0.0.1:7860)
        python app.py --share  (gera um link público temporário)

Só lê os arquivos de models/; não precisa do dataset nem de retreinar.
"""
import argparse
import html
import inspect
import tempfile
from collections import Counter
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.inferencia import Analisador

AQUI = Path(__file__).parent
A = Analisador(AQUI / "models")

CORAL, TINTA, CORPO, MUDO, LINHA = "#E4572E", "#15171C", "#4A4F57", "#878C95", "#E0DCD3"
FAIXA = {  # cor forte, fundo, texto
    "Alta": ("#E4572E", "#FCE4DB", "#7A2410"),
    "Média": ("#C98500", "#FBEFD2", "#5E3E00"),
    "Baixa": ("#1F8A5B", "#DDF1E6", "#0C4A30"),
}
ACAO = {"Alta": "Responder hoje: cliente insatisfeito, risco de perder a venda e ganhar uma avaliação ruim.",
        "Média": "Vale uma leitura humana: pode ser reclamação.",
        "Baixa": "Sem urgência: elogio ou comentário neutro. Pode receber um agradecimento automático."}

EXEMPLOS = [
    ["Comprei faz duas semanas e até agora não chegou, ninguém responde meu e-mail"],
    ["Produto excelente, chegou antes do prazo e muito bem embalado. Recomendo!"],
    ["O produto é bonito, mas veio uma peça a menos e a cor é diferente da foto"],
    ["veio quebrado o vidro do porta retrato, quero a troca"],
    ["ok"],
]

CSS = f"""
.gradio-container {{max-width: 1180px !important; margin: 0 auto !important;}}
#topo {{background:{TINTA}; border-radius:18px; padding:26px 30px; color:#F6F4EF; display:flex;
       justify-content:space-between; align-items:center; gap:24px; flex-wrap:wrap}}
#topo h1 {{font-size:34px; margin:0; letter-spacing:-.02em; color:#F6F4EF}}
#topo h1 span {{color:{CORAL}}}
#topo p {{margin:6px 0 0; color:#C9CDD3; font-size:15px; max-width:620px}}
#topo .kpis {{display:flex; gap:10px; flex-wrap:wrap}}
#topo .kpi {{background:#1F2229; border:1px solid #2E323A; border-radius:12px; padding:10px 14px; min-width:120px}}
#topo .kpi b {{display:block; font-size:22px; color:#fff}}
#topo .kpi small {{color:#9097A1; font-size:12px}}
button.primary, .primary {{background:{CORAL} !important; border-color:{CORAL} !important; color:#fff !important}}
footer {{display:none !important}}
"""

TOPO = """
<div id="topo">
  <div>
    <h1>Fila <span>Certa</span></h1>
    <p>Cole a mensagem de um cliente (avaliação, WhatsApp, e-mail, rede social). A rede neural diz se é
    reclamação, com que prioridade responder e qual o motivo provável.</p>
  </div>
  <div class="kpis">
    <div class="kpi"><b>{rec:.0%}</b><small>das reclamações encontradas</small></div>
    <div class="kpi"><b>{alta_tx:.0%}</b><small>de acerto na prioridade alta</small></div>
    <div class="kpi"><b>{n_te}</b><small>comentários no teste</small></div>
  </div>
</div>"""


def topo_html():
    m, f = A.cfg["metricas_teste"], A.cfg["faixas_teste"]
    n_te = sum(v["comentarios"] for v in f.values())
    return TOPO.format(rec=m["Recall (reclamação)"], alta_tx=f["Alta"]["taxa_reclamacao"],
                       n_te=f"{n_te:,}".replace(",", "."))


def cartao(txt):
    if not txt or not txt.strip():
        return f"<div style='padding:28px;color:{MUDO};border:1px dashed {LINHA};border-radius:14px'>Digite ou cole um comentário e clique em <b>Analisar</b>.</div>"
    r = A.analisar(txt)
    p, pr = r["probabilidade"], r["prioridade"]
    forte, fundo, tinta = FAIXA[pr]
    titulo = "Reclamação" if r["reclamacao"] else "Não é reclamação"
    chips = "".join(
        f"<span style='display:inline-flex;align-items:center;gap:6px;margin:3px 6px 3px 0;padding:4px 10px;border-radius:12px;"
        f"background:{'#FCE4DB' if e > 0 else '#DDF1E6'};color:{'#7A2410' if e > 0 else '#0C4A30'};font-size:14px'>"
        f"{html.escape(w)} <b>{'+' if e > 0 else '−'}{abs(e) * 100:.0f}%</b></span>" for w, e in r["palavras"]) or \
        f"<span style='color:{MUDO}'>texto curto demais para explicar</span>"
    if r["reclamacao"]:
        termos = ", ".join(sorted({html.escape(t) for _, t in r["termos"]})) or "nenhum termo-chave"
        motivo = (f"<div style='margin-top:16px;padding:14px 16px;border-radius:12px;background:#F6F4EF'>"
                  f"<div style='font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:{MUDO}'>Motivo provável</div>"
                  f"<div style='font-size:20px;font-weight:700;color:{TINTA};margin-top:4px'>{html.escape(r['motivo'])}</div>"
                  f"<div style='font-size:13px;color:{MUDO};margin-top:4px'>termos encontrados: {termos}</div></div>")
    else:
        motivo = ""
    return f"""
<div style="border:1px solid {LINHA};border-radius:16px;padding:22px 24px;background:#fff;font-family:inherit">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:12px">
    <span style="font-size:26px;font-weight:800;color:{TINTA};letter-spacing:-.02em">{titulo}</span>
    <span style="background:{fundo};color:{tinta};padding:6px 14px;border-radius:14px;font-weight:700;display:inline-flex;align-items:center;gap:8px">
      <i style="width:9px;height:9px;border-radius:50%;background:{forte};display:inline-block"></i>Prioridade {pr}</span>
  </div>
  <div style="display:flex;align-items:baseline;gap:10px;margin-top:14px">
    <span style="font-size:44px;font-weight:800;color:{forte};letter-spacing:-.03em">{p:.0%}</span>
    <span style="color:{CORPO}">de chance de ser reclamação</span></div>
  <div style="background:#EDEAE3;border-radius:6px;height:10px;overflow:hidden;margin-top:6px">
    <div style="width:{max(p * 100, 1):.1f}%;height:10px;background:{forte}"></div></div>
  <p style="margin:12px 0 0;color:{CORPO}">{ACAO[pr]}</p>
  {motivo}
  <div style="margin-top:16px;font-weight:700;color:{TINTA}">Palavras que mais pesaram
    <span style="font-weight:400;color:{MUDO};font-size:13px">(+ empurra para reclamação, − para elogio)</span></div>
  <div style="margin-top:6px">{chips}</div>
</div>"""


def lote(arquivo):
    if arquivo is None:
        return "Envie um CSV ou clique no exemplo.", None, None, None
    df = pd.read_csv(arquivo if isinstance(arquivo, str) else arquivo.name)
    col = next((c for c in df.columns if c.lower() in ("comentario", "comentário", "texto", "mensagem",
                                                         "review_comment_message")), None)
    if col is None:
        col = df.select_dtypes("object").columns[-1]
    df = df.dropna(subset=[col]).reset_index(drop=True)
    res = pd.DataFrame(A.analisar_lote(df[col].tolist()))
    res = pd.concat([df.drop(columns=[col]), res], axis=1)
    ordem = {"Alta": 0, "Média": 1, "Baixa": 2}
    res = res.assign(_o=res["prioridade"].map(ordem)).sort_values(["_o", "prob_reclamacao"], ascending=[True, False]).drop(columns="_o")
    rec = res[res["reclamacao"] == "Sim"]
    cont = Counter(rec["motivo"])
    n_alta = int((res["prioridade"] == "Alta").sum())
    resumo = (f"### {len(res)} mensagens analisadas\n"
              f"**{len(rec)} parecem reclamação** ({len(rec) / max(len(res), 1):.0%}) · "
              f"**{n_alta} com prioridade alta** para responder primeiro")
    if cont:
        resumo += f" · problema mais frequente: **{cont.most_common(1)[0][0]}**"
    fig, ax = plt.subplots(figsize=(9, 2.9))
    if cont:
        itens = cont.most_common()[::-1]
        ax.barh([k for k, _ in itens], [v for _, v in itens], color=CORAL, height=0.6)
        for i, (_, v) in enumerate(itens):
            ax.text(v + 0.05, i, str(v), va="center", color=CORPO)
        ax.set_xticks([])
    ax.set_title("Motivos das reclamações no lote", loc="left", fontsize=12, color=TINTA)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    saida = Path(tempfile.mkdtemp()) / "fila_priorizada.csv"
    res.to_csv(saida, index=False, encoding="utf-8-sig")
    # tabela da tela: prioridade e motivo primeiro, comentário por último (mais legível)
    extras = [c for c in res.columns if c not in ("comentario", "prob_reclamacao", "reclamacao", "prioridade", "motivo")]
    tabela = res[["prioridade", "prob_reclamacao", "motivo"] + extras + ["comentario"]].rename(
        columns={"prob_reclamacao": "prob.", "comentario": "mensagem"})
    tabela["prob."] = (tabela["prob."] * 100).round(0).astype(int).astype(str) + "%"
    return resumo, tabela, fig, str(saida)


def sobre_md():
    m, f = A.cfg["metricas_teste"], A.cfg["faixas_teste"]
    voc = f"{A.cfg['vocab']:,}".replace(",", ".")
    arq = "BiLSTM" if A.cfg["arquitetura"] == "bilstm" else "CNN 1D"
    return f"""
### Como funciona
1. O comentário é normalizado (minúsculas, sem acento) e quebrado em palavras. O "não" é mantido.
2. Cada palavra vira um número do vocabulário ({voc} palavras aprendidas no treino) e depois um vetor (*embedding*).
3. Uma rede **{arq}** treinada do zero lê a frase e devolve a probabilidade de ser reclamação.
4. A probabilidade vira **prioridade**: alta (≥ {A.lim['alta']:.2f}), média e baixa (< {A.lim['baixa']:.2f}). Os limiares foram definidos na validação.
5. Nas reclamações, regras de palavras-chave indicam o **motivo** provável.
6. "Palavras que mais pesaram": tiramos uma palavra por vez e medimos quanto a decisão muda.

### Desempenho no conjunto de teste (comentários nunca vistos no treino)
| F1 (reclamação) | Recall | Precisão | Acurácia |
|---|---|---|---|
| {m['F1 (reclamação)']:.3f} | {m['Recall (reclamação)']:.3f} | {m['Precisão (reclamação)']:.3f} | {m['Acurácia']:.3f} |

| Prioridade | % das mensagens | % que são reclamação | % de todas as reclamações |
|---|---|---|---|
| Alta | {f['Alta']['fracao']:.0%} | {f['Alta']['taxa_reclamacao']:.0%} | {f['Alta']['fracao_das_reclamacoes']:.0%} |
| Média | {f['Média']['fracao']:.0%} | {f['Média']['taxa_reclamacao']:.0%} | {f['Média']['fracao_das_reclamacoes']:.0%} |
| Baixa | {f['Baixa']['fracao']:.0%} | {f['Baixa']['taxa_reclamacao']:.0%} | {f['Baixa']['fracao_das_reclamacoes']:.0%} |

**Limitação:** treinado com avaliações de e-commerce (Olist, 2016-2018), em que a nota serve de rótulo.
Antes de usar em SAC, WhatsApp ou redes sociais, é preciso validar com mensagens reais desses canais.
"""


def tema():
    return gr.themes.Soft(primary_hue="orange", neutral_hue="stone",
                          font=[gr.themes.Font("IBM Plex Sans"), gr.themes.Font("system-ui"), gr.themes.Font("sans-serif")])


def criar_app():
    # Gradio 6 recebe tema e CSS no launch(); versões 4 e 5, no Blocks()
    extra = {"css": CSS, "theme": tema()} if "css" in inspect.signature(gr.Blocks.__init__).parameters else {}
    with gr.Blocks(title="Fila Certa: triagem de reclamações", **extra) as demo:
        gr.HTML(topo_html())
        with gr.Tab("1. Analisar uma mensagem"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    entrada = gr.Textbox(label="Mensagem do cliente", lines=5,
                                         placeholder="Ex.: comprei faz 15 dias e não chegou...")
                    botao = gr.Button("Analisar", variant="primary")
                    gr.Examples(EXEMPLOS, [entrada], label="Exemplos (clique para preencher)")
                with gr.Column(scale=6):
                    saida = gr.HTML(cartao(""))
            botao.click(cartao, entrada, saida)
            entrada.submit(cartao, entrada, saida)
        with gr.Tab("2. Analisar um lote (CSV)"):
            gr.Markdown("Envie um CSV com uma coluna `comentario` (ou `texto`/`mensagem`). A lista volta "
                        "**ordenada por prioridade**, com o motivo de cada reclamação e o resumo dos problemas.")
            with gr.Row():
                arq = gr.File(label="CSV de mensagens", file_types=[".csv"], scale=3)
                with gr.Column(scale=2):
                    exemplo = AQUI / "docs" / "exemplo_comentarios.csv"
                    if exemplo.exists():
                        gr.Examples([[str(exemplo)]], [arq], label="Exemplo: mensagens de WhatsApp, e-mail e Reclame Aqui")
                    b2 = gr.Button("Analisar lote", variant="primary")
            resumo = gr.Markdown()
            grafico = gr.Plot(show_label=False)
            tabela = gr.Dataframe(interactive=False, wrap=True)
            baixar = gr.File(label="Baixar a fila priorizada (CSV)")
            b2.click(lote, arq, [resumo, tabela, grafico, baixar])
        with gr.Tab("Como funciona"):
            gr.Markdown(sobre_md())
    demo._fila_extra = extra
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--porta", type=int, default=7860)
    a = ap.parse_args()
    demo = criar_app()
    kw = {} if demo._fila_extra else {"css": CSS, "theme": tema()}
    demo.launch(share=a.share, server_port=a.porta, **kw)
