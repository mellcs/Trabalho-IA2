/* Fila Certa: interface (sem frameworks; fala com a API em app.py) */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const ORDEM = { "Alta": 0, "Média": 1, "Baixa": 2 };
  const COR = { "Alta": "var(--coral)", "Média": "var(--ambar)", "Baixa": "var(--verde)" };
  const CANAL_IC = { "whatsapp": "c-whatsapp", "e-mail": "c-email", "email": "c-email", "instagram": "c-instagram",
    "sac": "c-sac", "reclame aqui": "c-reclame", "avaliação": "c-avaliacao", "avaliacao": "c-avaliacao" };
  const fmt = n => n.toLocaleString("pt-BR");
  const pct = (x, d = 0) => (x * 100).toLocaleString("pt-BR", { maximumFractionDigits: d, minimumFractionDigits: d }) + "%";
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const icone = (id, cls = "ic") => `<svg class="${cls}"><use href="#${id}"/></svg>`;
  const canalIc = c => icone(CANAL_IC[(c || "").toLowerCase()] || "c-outro");

  // ---------- armazenamento local (só conveniência; o app funciona sem) ----------
  const LS = {
    ler(k, pad) { try { const v = localStorage.getItem("fc:" + k); return v ? JSON.parse(v) : pad; } catch { return pad; } },
    gravar(k, v) { try { localStorage.setItem("fc:" + k, JSON.stringify(v)); } catch { /* sem armazenamento */ } },
  };

  const S = {
    msgs: [], cfg: null, painel: null,
    resolvidas: new Set(LS.ler("resolvidas", [])),
    correcoes: LS.ler("correcoes", {}),          // id -> {texto, ia, equipe}
    importadas: LS.ler("importadas", []),
    filtro: { faixa: "pendentes", motivo: "", canal: "", busca: "" },
    sel: null, cache: new Map(),
  };
  const salvar = () => { LS.gravar("resolvidas", [...S.resolvidas]); LS.gravar("correcoes", S.correcoes); LS.gravar("importadas", S.importadas); };

  async function api(caminho, corpo) {
    const r = await fetch(caminho, corpo ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(corpo) } : {});
    if (!r.ok) throw new Error(`${caminho}: ${r.status}`);
    return r.json();
  }

  // ---------- toast ----------
  let toastT;
  function toast(msg, acao) {
    const t = $("#toast");
    t.innerHTML = `<span>${msg}</span>` + (acao ? `<button>${acao.rotulo}</button>` : "");
    if (acao) $("button", t).onclick = () => { acao.fn(); t.classList.remove("on"); };
    t.classList.add("on");
    clearTimeout(toastT);
    toastT = setTimeout(() => t.classList.remove("on"), acao ? 6000 : 3000);
  }

  // ---------- navegação entre telas ----------
  function irPara(view) {
    $$(".view").forEach(v => v.classList.toggle("ativa", v.id === "v-" + view));
    $$("[data-view]").forEach(b => b.classList.toggle("ativo", b.dataset.view === view));
    if (view === "painel") desenharPainel();
    try { history.replaceState(null, "", "#" + view); } catch { }
  }
  $$("[data-view]").forEach(b => b.addEventListener("click", () => irPara(b.dataset.view)));

  // =====================================================================
  // CAIXA DE ENTRADA
  // =====================================================================
  function pendentes() { return S.msgs.filter(m => !S.resolvidas.has(m.id)); }

  function filtradas() {
    const f = S.filtro, q = f.busca.trim().toLowerCase();
    return S.msgs.filter(m => {
      const res = S.resolvidas.has(m.id);
      if (f.faixa === "resolvidas" ? !res : res) return false;
      if (!["pendentes", "resolvidas"].includes(f.faixa) && m.prioridade !== f.faixa) return false;
      if (f.motivo && m.motivo !== f.motivo) return false;
      if (f.canal && m.canal !== f.canal) return false;
      if (q && !(m.texto.toLowerCase().includes(q) || (m.cliente || "").toLowerCase().includes(q))) return false;
      return true;
    }).sort((a, b) => ORDEM[a.prioridade] - ORDEM[b.prioridade] || b.probabilidade - a.probabilidade);
  }

  function desenharSeg() {
    const p = pendentes();
    const n = k => p.filter(m => m.prioridade === k).length;
    const opcoes = [["pendentes", "Pendentes", p.length], ["Alta", "Alta", n("Alta")], ["Média", "Média", n("Média")],
      ["Baixa", "Baixa", n("Baixa")], ["resolvidas", "Resolvidas", S.resolvidas.size]];
    $("#seg").innerHTML = opcoes.map(([k, r, c]) =>
      `<button role="tab" data-f="${k}" class="${S.filtro.faixa === k ? "ativo" : ""}" aria-selected="${S.filtro.faixa === k}">${r} <b>${c}</b></button>`).join("");
    $$("#seg button").forEach(b => b.onclick = () => { S.filtro.faixa = b.dataset.f; desenharCaixa(); });
  }

  function desenharResumo() {
    const p = pendentes(), rec = p.filter(m => m.reclamacao).length, alta = p.filter(m => m.prioridade === "Alta").length;
    $("#caixa-resumo").innerHTML = `${fmt(p.length)} pendentes · <b style="color:var(--coral-tx)">${fmt(alta)} urgentes</b> · ${fmt(rec)} reclamações`;
    const badge = alta ? String(alta) : "";
    $("#nav-pendentes").textContent = badge; $("#tab-pendentes").textContent = badge;
    const baixas = p.filter(m => m.prioridade === "Baixa");
    const ft = S.cfg && S.cfg.faixas_teste;
    $("#foco").hidden = !baixas.length;
    if (baixas.length && ft) {
      $("#foco-titulo").textContent = `${fmt(baixas.length)} mensagens podem esperar`;
      $("#foco-texto").textContent = `Na prioridade baixa, só ${pct(ft["Baixa"].taxa_reclamacao, 1)} são reclamações. Foque nas outras ${fmt(p.length - baixas.length)}.`;
      $("#b-agradecer-txt").textContent = `Agradecer as ${fmt(baixas.length)}`;
    }
  }

  function itemHTML(m) {
    const res = S.resolvidas.has(m.id);
    const corr = S.correcoes[m.id];
    return `<li class="item ${m.prioridade}${res ? " resolvida" : ""}" role="option" tabindex="-1" data-id="${esc(m.id)}" aria-selected="${S.sel === m.id}">
      <span class="canal-ic" title="${esc(m.canal)}">${canalIc(m.canal)}</span>
      <div style="min-width:0">
        <div class="item-topo"><b>${esc(m.cliente || "Cliente")}</b><small>${esc(m.canal)} · ${esc(m.data || "")}</small><span class="pill p-${m.prioridade}">${m.prioridade}</span></div>
        <p>${esc(m.texto)}</p>
        <div class="item-rodape">
          <span class="prob" title="Probabilidade de ser reclamação"><i style="--p:${(m.probabilidade * 100).toFixed(0)}%;--cor:${COR[m.prioridade]}"></i>${pct(m.probabilidade)}</span>
          ${m.reclamacao ? `<span class="chip">${esc(m.motivo)}</span>` : `<span class="chip">Elogio / neutro</span>`}
          ${corr ? `<span class="chip" title="Corrigida pela equipe">${icone("i-bandeira")} corrigida</span>` : ""}
          ${res ? `<span class="ok-ic">${icone("i-check")} resolvida</span>` : ""}
        </div>
      </div></li>`;
  }

  function desenharLista() {
    const l = filtradas();
    const ul = $("#lista");
    ul.innerHTML = l.length ? l.map(itemHTML).join("") :
      `<li class="lista-vazia">${S.filtro.faixa === "resolvidas" ? "Nenhuma mensagem resolvida ainda." : "Nada por aqui com esses filtros. 🎉"}</li>`;
    $$(".item", ul).forEach(li => li.onclick = () => selecionar(li.dataset.id));
  }

  function desenharFiltros() {
    const mot = [...new Set(S.msgs.filter(m => m.reclamacao).map(m => m.motivo))].sort();
    const can = [...new Set(S.msgs.map(m => m.canal))].sort();
    const fill = (sel, itens, rot) => {
      const atual = sel.value;
      sel.innerHTML = `<option value="">${rot}</option>` + itens.map(x => `<option>${esc(x)}</option>`).join("");
      sel.value = itens.includes(atual) ? atual : "";
    };
    fill($("#f-motivo"), mot, "Todos os motivos");
    fill($("#f-canal"), can, "Todos os canais");
  }

  function desenharCaixa() { desenharSeg(); desenharResumo(); desenharLista(); }

  $("#f-busca").addEventListener("input", e => { S.filtro.busca = e.target.value; desenharLista(); });
  $("#f-motivo").addEventListener("change", e => { S.filtro.motivo = e.target.value; desenharLista(); });
  $("#f-canal").addEventListener("change", e => { S.filtro.canal = e.target.value; desenharLista(); });

  // ---------- detalhe ----------
  async function detalhar(m) {
    if (S.cache.has(m.id)) return S.cache.get(m.id);
    const r = await api("/api/analisar", { texto: m.texto, canal: m.canal, cliente: m.cliente, data: m.data, nota: m.nota, id: m.id });
    S.cache.set(m.id, r);
    return r;
  }

  function destaquesHTML(seg) {
    // destaca só as palavras que realmente pesaram: até 6, e com pelo menos 25% do peso da maior
    const max = Math.max(0.0001, ...seg.map(s => Math.abs(s.w)));
    const top = new Set(seg.map((s, i) => [Math.abs(s.w), i]).sort((a, b) => b[0] - a[0]).slice(0, 6)
      .filter(([w]) => w >= Math.max(0.05, 0.25 * max)).map(([, i]) => i));
    return seg.map((s, i) => {
      if (!top.has(i)) return esc(s.t);
      const a = Math.min(1, Math.abs(s.w) / max) * 0.55 + 0.12;
      const cor = s.w > 0 ? `rgba(228,87,46,${a.toFixed(2)})` : `rgba(31,138,91,${(a * 0.8).toFixed(2)})`;
      return `<mark style="background:${cor}" title="${s.w > 0 ? "+" : "−"}${Math.round(Math.abs(s.w) * 100)}% da decisão">${esc(s.t)}</mark>`;
    }).join("");
  }

  function detalheHTML(m, r, opcoes = {}) {
    const p = r.probabilidade, pr = r.prioridade, res = S.resolvidas.has(m.id), corr = S.correcoes[m.id];
    const titulo = r.reclamacao ? "Reclamação" : "Não é reclamação";
    const nota = m.nota ? `<span class="nota-real">Nota que o cliente deu: <span class="estrelas">${"★".repeat(m.nota)}${"☆".repeat(5 - m.nota)}</span></span>` : "";
    const topo = opcoes.teste ? "" : `
      <div class="det-topo">
        <button class="btn btn-fantasma voltar" id="b-voltar" aria-label="Voltar para a lista">${icone("i-voltar")}</button>
        <span class="canal-ic">${canalIc(m.canal)}</span>
        <div><b>${esc(m.cliente || "Cliente")}</b><small>${esc(m.canal)} · ${esc(m.data || "")}</small></div>
        <div class="acoes">
          <button class="btn ${res ? "" : "btn-escuro"}" id="b-resolver">${icone(res ? "i-desfazer" : "i-check")}<span>${res ? "Reabrir" : "Marcar como resolvida"}</span></button>
        </div>
      </div>`;
    return `<div class="det">
      ${topo}
      <div class="texto-msg">${destaquesHTML(r.destaques)}</div>
      <div class="leg-destaque">
        <span><i style="background:rgba(228,87,46,.5)"></i>pesou para reclamação</span>
        <span><i style="background:rgba(31,138,91,.4)"></i>pesou para elogio</span>
        ${nota}
      </div>
      <div class="cartao veredito">
        <div class="medidor" style="--p:${(p * 100).toFixed(1)};--cor:${COR[pr]}"><div><span><b>${pct(p)}</b><small>chance</small></span></div></div>
        <div>
          <h3>${titulo}</h3>
          <div class="linha"><span class="pill p-${pr}">Prioridade ${pr}</span>${icone("i-relogio")}<span>${esc(r.prazo)}</span></div>
        </div>
      </div>
      ${r.reclamacao ? `<div class="motivo-box">
        <div><small>Motivo provável</small><b>${esc(r.motivo)}</b><span>${r.termos.length ? "Termos encontrados: " + r.termos.map(esc).join(", ") : "Nenhum termo-chave encontrado"}</span></div>
        <div><small>O que fazer</small><b>Ação para a equipe</b><span>${esc(r.acao)}</span></div>
      </div>` : ""}
      <div class="resposta">
        <header><h2>Resposta sugerida</h2><small>Revise antes de enviar</small></header>
        <textarea id="resp-texto" aria-label="Resposta sugerida">${esc(r.resposta)}</textarea>
        <div class="linha-btn">
          <button class="btn" id="b-copiar">${icone("i-copiar")}<span>Copiar resposta</span></button>
          ${opcoes.teste ? "" : `<button class="btn btn-primario" id="b-responder">${icone("i-check")}<span>Copiar e resolver</span></button>`}
        </div>
      </div>
      ${opcoes.teste ? "" : `<div class="feedback">${corr
        ? `<span class="corrigido">${icone("i-bandeira")} A equipe marcou como <b>${corr.equipe ? "reclamação" : "não reclamação"}</b>. Isso entra no próximo treino.</span> <button class="btn btn-fantasma" id="b-desfaz-corr">Desfazer</button>`
        : `<span>A IA errou?</span><button class="btn" id="b-corrigir">${icone("i-bandeira")}<span>${r.reclamacao ? "Não é reclamação" : "É reclamação"}</span></button>`}</div>`}
    </div>`;
  }

  async function copiar(txt) {
    try { await navigator.clipboard.writeText(txt); return true; }
    catch {
      const t = document.createElement("textarea"); t.value = txt; document.body.appendChild(t); t.select();
      let ok = false; try { ok = document.execCommand("copy"); } catch { } t.remove(); return ok;
    }
  }

  function alternarResolvida(id, forcar) {
    const virou = forcar ?? !S.resolvidas.has(id);
    if (virou) S.resolvidas.add(id); else S.resolvidas.delete(id);
    salvar();
    return virou;
  }

  async function selecionar(id, foco = true) {
    S.sel = id;
    const m = S.msgs.find(x => x.id === id);
    $$("#lista .item").forEach(li => li.setAttribute("aria-selected", li.dataset.id === id));
    const box = $("#detalhe");
    box.classList.add("aberto");
    if (!m) return;
    if (!S.cache.has(id)) box.innerHTML = `<div class="carregando"><div class="spin"></div></div>`;
    let r;
    try { r = await detalhar(m); } catch (e) { box.innerHTML = `<div class="vazio"><b>Não foi possível analisar</b><span>${esc(e.message)}</span></div>`; return; }
    if (S.sel !== id) return;
    box.innerHTML = detalheHTML(m, r);
    box.scrollTop = 0;
    $("#b-voltar").onclick = fecharDetalhe;
    $("#b-copiar").onclick = async () => toast(await copiar($("#resp-texto").value) ? "Resposta copiada." : "Não deu para copiar automaticamente.");
    $("#b-responder").onclick = async () => {
      await copiar($("#resp-texto").value);
      alternarResolvida(id, true);
      const prox = proximaDepois(id);
      desenharCaixa();
      toast("Resposta copiada e mensagem resolvida.", { rotulo: "Desfazer", fn: () => { alternarResolvida(id, false); desenharCaixa(); selecionar(id); } });
      if (prox) selecionar(prox); else fecharDetalhe();
    };
    $("#b-resolver").onclick = () => {
      const virou = alternarResolvida(id);
      const prox = virou ? proximaDepois(id) : null;
      desenharCaixa();
      if (virou) {
        toast("Mensagem resolvida.", { rotulo: "Desfazer", fn: () => { alternarResolvida(id, false); desenharCaixa(); selecionar(id); } });
        if (prox) selecionar(prox); else fecharDetalhe();
      } else selecionar(id);
    };
    const bc = $("#b-corrigir");
    if (bc) bc.onclick = () => {
      S.correcoes[id] = { texto: m.texto, ia: r.reclamacao, equipe: !r.reclamacao };
      salvar(); desenharLista(); selecionar(id);
      toast("Obrigado! A correção foi guardada para o próximo treino da rede.");
    };
    const bd = $("#b-desfaz-corr");
    if (bd) bd.onclick = () => { delete S.correcoes[id]; salvar(); desenharLista(); selecionar(id); };
    if (foco && matchMedia("(max-width:900px)").matches) box.focus?.();
  }

  function proximaDepois(id) {
    const l = filtradas().filter(m => m.id !== id && !S.resolvidas.has(m.id));
    return l.length ? l[0].id : null;
  }

  function fecharDetalhe() { $("#detalhe").classList.remove("aberto"); }

  // teclado: ↑ ↓ navegam, Esc fecha, / busca
  document.addEventListener("keydown", e => {
    const caixaAtiva = $("#v-caixa").classList.contains("ativa");
    const digitando = /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName);
    if (e.key === "/" && caixaAtiva && !digitando) { e.preventDefault(); $("#f-busca").focus(); return; }
    if (e.key === "Escape") { fecharDetalhe(); if (digitando) document.activeElement.blur(); return; }
    if (caixaAtiva && !digitando && S.sel && (e.key === "r" || e.key === "R")) { e.preventDefault(); $("#b-resolver")?.click(); return; }
    if (caixaAtiva && !digitando && S.sel && (e.key === "c" || e.key === "C")) { e.preventDefault(); $("#b-copiar")?.click(); return; }
    if (!caixaAtiva || digitando || !["ArrowDown", "ArrowUp"].includes(e.key)) return;
    e.preventDefault();
    const l = filtradas(); if (!l.length) return;
    let i = l.findIndex(m => m.id === S.sel);
    i = e.key === "ArrowDown" ? Math.min(l.length - 1, i + 1) : Math.max(0, i - 1);
    selecionar(l[i].id, false);
    $(`#lista .item[data-id="${CSS.escape(l[i].id)}"]`)?.scrollIntoView({ block: "nearest" });
  });


  // ---------- simular mensagens chegando (demonstração ao vivo) ----------
  const CHEGANDO = [
    { canal: "WhatsApp", cliente: "Mariana T.", texto: "Oi, meu pedido era pra ter chegado ontem e até agora nada, o rastreio parou" },
    { canal: "Instagram", cliente: "Rodrigo P.", texto: "Chegou certinho, produto lindo! Obrigado 😍" },
    { canal: "E-mail", cliente: "Fernanda L.", texto: "Recebi a cafeteira com a tampa quebrada, preciso da troca urgente" },
    { canal: "Avaliação", cliente: "Pedido #9b21f0", texto: "Ótimo, superou as expectativas" },
    { canal: "Reclame Aqui", cliente: "Gustavo M.", texto: "Cancelei a compra há 10 dias e o estorno não caiu. Ninguém responde." },
    { canal: "WhatsApp", cliente: "Aline S.", texto: "Pedi tamanho M e veio G, como faço pra trocar?" },
    { canal: "SAC", cliente: "Eduardo C.", texto: "Tudo certo com o pedido, entrega rápida" },
    { canal: "E-mail", cliente: "Patrícia R.", texto: "Comprei 3 unidades e só vieram 2 na caixa" },
  ];
  let vivoT = null, vivoI = 0;
  function pararVivo() {
    clearInterval(vivoT); vivoT = null;
    $("#b-vivo").setAttribute("aria-pressed", "false");
    $("#b-vivo-txt").textContent = vivoI >= CHEGANDO.length ? "Simular de novo" : "Simular mensagens chegando";
  }
  async function chegarUma() {
    if (vivoI >= CHEGANDO.length) { pararVivo(); vivoI = 0; return; }
    const m = { ...CHEGANDO[vivoI++], id: `sim-${Date.now()}`, data: "agora" };
    try {
      const [r] = await api("/api/lote", { mensagens: [m] });
      S.msgs.unshift(r); desenharFiltros(); desenharCaixa();
      const li = $(`#lista .item[data-id="${CSS.escape(r.id)}"]`);
      if (li) { li.classList.add("nova"); li.scrollIntoView({ block: "nearest", behavior: "smooth" }); }
      if (r.prioridade === "Alta") toast(`Nova mensagem urgente: ${esc(r.cliente)} (${esc(r.canal)}) · ${esc(r.motivo)}`);
    } catch (e) { pararVivo(); toast("Erro na simulação: " + e.message); }
  }
  $("#b-vivo").onclick = () => {
    if (vivoT) { pararVivo(); return; }
    $("#b-vivo").setAttribute("aria-pressed", "true"); $("#b-vivo-txt").textContent = "Recebendo… (clique para parar)";
    S.filtro.faixa = "pendentes"; chegarUma(); vivoT = setInterval(chegarUma, 3500);
  };

  // ---------- restaurar a demonstração ----------
  $("#b-restaurar").onclick = async () => {
    pararVivo(); vivoI = 0;
    S.resolvidas.clear(); S.correcoes = {}; S.importadas = []; S.cache.clear(); salvar();
    S.filtro = { faixa: "pendentes", motivo: "", canal: "", busca: "" };
    $("#f-busca").value = "";
    S.msgs = await api("/api/caixa");
    desenharFiltros(); desenharCaixa(); fecharDetalhe();
    const p = filtradas()[0]; if (p && matchMedia("(min-width:901px)").matches) selecionar(p.id, false);
    toast("Demonstração restaurada: caixa de entrada como no início.");
  };

  // agradecer automaticamente as de prioridade baixa
  $("#b-agradecer").onclick = () => {
    const ids = pendentes().filter(m => m.prioridade === "Baixa").map(m => m.id);
    ids.forEach(id => S.resolvidas.add(id)); salvar(); desenharCaixa();
    toast(`${fmt(ids.length)} clientes receberiam um agradecimento automático.`, {
      rotulo: "Desfazer", fn: () => { ids.forEach(id => S.resolvidas.delete(id)); salvar(); desenharCaixa(); }
    });
  };

  // ---------- importar / colar / exportar ----------
  function lerCSV(txt) {
    const linhas = []; let campo = "", linha = [], aspas = false;
    const sep = (txt.split("\n")[0].match(/;/g) || []).length > (txt.split("\n")[0].match(/,/g) || []).length ? ";" : ",";
    for (let i = 0; i < txt.length; i++) {
      const c = txt[i];
      if (aspas) { if (c === '"') { if (txt[i + 1] === '"') { campo += '"'; i++; } else aspas = false; } else campo += c; }
      else if (c === '"') aspas = true;
      else if (c === sep) { linha.push(campo); campo = ""; }
      else if (c === "\n" || c === "\r") { if (c === "\r" && txt[i + 1] === "\n") i++; linha.push(campo); linhas.push(linha); linha = []; campo = ""; }
      else campo += c;
    }
    if (campo || linha.length) { linha.push(campo); linhas.push(linha); }
    return linhas.filter(l => l.some(x => x.trim()));
  }

  async function adicionar(novas, origem) {
    if (!novas.length) { toast("Nenhuma mensagem encontrada."); return; }
    toast(`Analisando ${fmt(novas.length)} mensagens…`);
    try {
      const base = Date.now();
      const env = novas.map((m, i) => ({ ...m, id: `imp-${base}-${i}`, data: m.data || "hoje", cliente: m.cliente || `Cliente ${i + 1}` }));
      const r = await api("/api/lote", { mensagens: env });
      S.importadas = [...r, ...S.importadas].slice(0, 2000);
      S.msgs = [...r, ...S.msgs];
      salvar(); desenharFiltros(); S.filtro.faixa = "pendentes"; desenharCaixa();
      const alta = r.filter(m => m.prioridade === "Alta").length;
      toast(`${fmt(r.length)} mensagens ${origem}: ${fmt(alta)} urgentes foram para o topo.`);
    } catch (e) { toast("Erro ao analisar: " + e.message); }
  }

  $("#b-importar").onclick = () => $("#arq").click();
  $("#arq").onchange = async e => {
    const f = e.target.files[0]; if (!f) return;
    const linhas = lerCSV(await f.text());
    e.target.value = "";
    if (linhas.length < 2) { toast("O CSV precisa de um cabeçalho e pelo menos uma linha."); return; }
    const cab = linhas[0].map(c => c.trim().toLowerCase());
    let iTxt = cab.findIndex(c => ["comentario", "comentário", "texto", "mensagem", "review_comment_message"].includes(c));
    if (iTxt < 0) iTxt = cab.length - 1;
    const iCanal = cab.indexOf("canal"), iCli = cab.findIndex(c => ["cliente", "nome"].includes(c));
    adicionar(linhas.slice(1).filter(l => (l[iTxt] || "").trim()).map(l => ({
      texto: l[iTxt].trim(), canal: iCanal >= 0 ? l[iCanal] : "Importado", cliente: iCli >= 0 ? l[iCli] : null
    })), "importadas");
  };

  const dlg = $("#dlg-colar");
  $("#b-colar").onclick = () => { $("#colar-texto").value = ""; dlg.showModal(); $("#colar-texto").focus(); };
  dlg.addEventListener("close", () => {
    if (dlg.returnValue !== "ok") return;
    const canal = $("#colar-canal").value;
    adicionar($("#colar-texto").value.split("\n").map(s => s.trim()).filter(Boolean).map(t => ({ texto: t, canal })), "adicionadas");
  });

  function baixar(nome, conteudo, tipo = "text/csv;charset=utf-8") {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob(["﻿" + conteudo], { type: tipo }));
    a.download = nome; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
  const csvCampo = v => `"${String(v ?? "").replace(/"/g, '""')}"`;
  $("#b-exportar").onclick = () => {
    const l = filtradas();
    const cab = ["prioridade", "prob_reclamacao", "motivo", "prazo", "acao", "canal", "cliente", "data", "mensagem", "resposta_sugerida", "status"];
    const linhas = l.map(m => [m.prioridade, m.probabilidade.toFixed(3), m.motivo, m.prazo, m.acao, m.canal, m.cliente, m.data, m.texto, m.resposta,
      S.resolvidas.has(m.id) ? "resolvida" : "pendente"].map(csvCampo).join(","));
    baixar("fila_certa.csv", [cab.join(","), ...linhas].join("\n"));
    toast(`${fmt(l.length)} mensagens exportadas na ordem da fila.`);
  };

  // =====================================================================
  // PAINEL
  // =====================================================================
  const MES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"];
  const nomeMes = m => { const [a, n] = m.split("-"); return `${MES[+n - 1]}/${a.slice(2)}`; };
  const nomeMesLongo = m => { const [a, n] = m.split("-"); return `${MES[+n - 1]}/${a}`; };

  function desenharKPIs() {
    const P = S.painel, ft = P.faixas_teste;
    const lerAM = ft["Alta"].fracao + ft["Média"].fracao, achaAM = ft["Alta"].fracao_das_reclamacoes + ft["Média"].fracao_das_reclamacoes;
    $("#painel-sub").textContent = `${fmt(P.mensagens)} avaliações reais que a rede nunca viu no treino (${nomeMesLongo(P.periodo[0])} a ${nomeMesLongo(P.periodo[1])})`;
    $("#kpis").innerHTML = `
      <div class="kpi"><small>Mensagens analisadas</small><b>${fmt(P.mensagens)}</b><span>conjunto de teste da Olist</span></div>
      <div class="kpi"><small>Reclamações detectadas</small><b>${fmt(P.reclamacoes)}</b><span>${pct(P.reclamacoes / P.mensagens)} das mensagens</span></div>
      <div class="kpi"><small>Prioridade alta</small><b>${fmt(P.faixas["Alta"])}</b><span>${pct(ft["Alta"].taxa_reclamacao)} são reclamação de verdade</span></div>
      <div class="kpi destaque"><small>Lendo só alta + média</small><b>${pct(achaAM)}</b><span>das reclamações encontradas lendo ${pct(lerAM)} das mensagens</span></div>`;
  }

  function desenharSerie() {
    const P = S.painel, el = $("#g-serie");
    const W = Math.max(300, el.clientWidth), H = W < 520 ? 240 : 330, m = { t: 14, r: 8, b: 30, l: 34 };
    const dados = P.serie.map(s => ({ mes: s.mes, ent: s.por_motivo["Entrega / atraso"], out: s.reclamacoes - s.por_motivo["Entrega / atraso"], tot: s.reclamacoes, n: s.mensagens }));
    const max = Math.max(...dados.map(d => d.tot)) * 1.1;
    const pw = W - m.l - m.r, ph = H - m.t - m.b, bw = pw / dados.length, larg = Math.max(4, bw * 0.68);
    const y = v => m.t + ph - (v / max) * ph;
    const alertas = new Set(P.alertas.map(a => a.mes));
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Reclamações por mês, separando entrega das demais">`;
    const passo = max > 200 ? 50 : 25;
    for (let v = 0; v <= max; v += passo) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}" stroke="#ECE9E2"/><text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end" font-size="11" fill="#7E838C">${v}</text>`;
    dados.forEach((d, i) => {
      const x = m.l + i * bw + (bw - larg) / 2, r = Math.min(4, larg / 2);
      const yO = y(d.out), yE = y(d.tot);
      s += `<g class="barra-m" data-i="${i}">`;
      s += `<rect x="${m.l + i * bw}" y="${m.t}" width="${bw}" height="${ph}" fill="transparent"/>`;
      s += `<path d="M${x},${y(0)} V${yO + 1} H${x + larg} V${y(0)} Z" fill="#C9C3B7"/>`;
      s += `<path d="M${x},${yO - 1} V${yE + r} Q${x},${yE} ${x + r},${yE} H${x + larg - r} Q${x + larg},${yE} ${x + larg},${yE + r} V${yO - 1} Z" fill="${alertas.has(d.mes) ? "#E4572E" : "#F0A58C"}"/>`;
      if (alertas.has(d.mes)) s += `<circle cx="${x + larg / 2}" cy="${yE - 9}" r="4" fill="#E4572E"/>`;
      const mostrar = W < 520 ? i % 3 === 0 : (i % 2 === 0);
      if (mostrar) s += `<text x="${x + larg / 2}" y="${H - 10}" text-anchor="middle" font-size="11" fill="#7E838C">${nomeMes(d.mes)}</text>`;
      s += `</g>`;
    });
    s += `</svg>`;
    el.innerHTML = s;
    $("#leg-serie").innerHTML = `<span><i style="background:#E4572E"></i>Entrega / atraso (mês com alerta)</span><span><i style="background:#F0A58C"></i>Entrega / atraso</span><span><i style="background:#C9C3B7"></i>Demais motivos</span>`;
    const tip = $("#tooltip");
    $$(".barra-m", el).forEach(g => {
      const d = dados[+g.dataset.i];
      g.addEventListener("mousemove", e => {
        tip.innerHTML = `<b>${nomeMesLongo(d.mes)}</b><br>${fmt(d.tot)} reclamações de ${fmt(d.n)} mensagens<br>Entrega / atraso: <b>${fmt(d.ent)}</b><br>Demais motivos: ${fmt(d.out)}`;
        tip.style.left = Math.min(innerWidth - 250, e.clientX + 14) + "px"; tip.style.top = (e.clientY - 10) + "px"; tip.classList.add("on");
      });
      g.addEventListener("mouseleave", () => tip.classList.remove("on"));
    });
  }

  function desenharAlertas() {
    const ACAO = { "Entrega / atraso": "Cobrar a transportadora e avisar os clientes antes que reclamem." };
    $("#alertas").innerHTML = [...S.painel.alertas].reverse().map(a => `<li>${icone("i-alerta")}<div>
      <b>${nomeMesLongo(a.mes)}: ${esc(a.motivo)} subiu ${pct(a.alta)}</b>
      <span>${fmt(a.antes)} → ${fmt(a.depois)} reclamações em relação ao mês anterior. ${ACAO[a.motivo] || "Investigar a causa."}</span></div></li>`).join("")
      || `<li>${icone("i-check")}<div><b>Nenhum alerta</b></div></li>`;
  }

  function desenharMotivos() {
    const mt = Object.entries(S.painel.motivos).sort((a, b) => b[1] - a[1]);
    const tot = mt.reduce((s, [, v]) => s + v, 0), max = mt[0][1];
    $("#g-motivos").innerHTML = mt.map(([k, v]) => `<div class="barra${k.startsWith("Outros") ? " outros" : ""}">
      <span>${esc(k)}</span><div class="trilho"><i style="width:${(v / max * 100).toFixed(1)}%"></i></div><b>${pct(v / tot)}</b></div>`).join("");
  }

  function calcular() {
    const ft = S.painel.faixas_teste;
    const n = Math.max(0, +$("#c-msgs").value || 0), min = Math.max(0, +$("#c-min").value || 0), hora = Math.max(0, +$("#c-hora").value || 0);
    const lerAM = ft["Alta"].fracao + ft["Média"].fracao, achaAM = ft["Alta"].fracao_das_reclamacoes + ft["Média"].fracao_das_reclamacoes;
    const dias = 22, hoje = n * min * dias / 60, com = n * lerAM * min * dias / 60, eco = hoje - com;
    const h = v => v.toLocaleString("pt-BR", { maximumFractionDigits: v < 10 ? 1 : 0 }) + " h";
    $("#calc-res").innerHTML = `
      <div><small>Hoje (lendo tudo)</small><b>${h(hoje)}/mês</b></div>
      <div><small>Com a Fila Certa (alta + média)</small><b>${h(com)}/mês</b></div>
      <div class="ganho"><div><small>Economia</small><b>${h(eco)}/mês</b></div><div><small>em dinheiro</small><b>R$ ${Math.round(eco * hora).toLocaleString("pt-BR")}</b></div><div><small>reclamações encontradas</small><b>${pct(achaAM)}</b></div></div>`;
    $("#calc-nota").textContent = `Considera 22 dias úteis por mês. A equipe lê só as mensagens de prioridade alta e média (${pct(lerAM)} do total, medido no teste) e as de prioridade baixa recebem agradecimento automático.`;
  }
  ["#c-msgs", "#c-min", "#c-hora"].forEach(s => $(s).addEventListener("input", calcular));

  function desenharCorrecoes() {
    const c = Object.values(S.correcoes);
    $("#correcoes-txt").textContent = c.length
      ? `${c.length} mensagens foram corrigidas pela equipe nesta máquina. Elas viram exemplos novos para retreinar a rede (botão Baixar).`
      : "Quando a equipe marca que a IA errou, a correção fica guardada aqui e vira exemplo novo para o próximo treino da rede.";
    $("#b-exp-correcoes").disabled = !c.length;
  }
  $("#b-exp-correcoes").onclick = () => {
    const c = Object.values(S.correcoes);
    baixar("correcoes_para_treino.csv", ["texto,reclamacao_segundo_equipe,reclamacao_segundo_ia", ...c.map(x => [x.texto, x.equipe ? 1 : 0, x.ia ? 1 : 0].map(csvCampo).join(","))].join("\n"));
  };

  function desenharPainel() {
    if (!S.painel) return;
    desenharKPIs(); desenharAlertas(); desenharMotivos(); calcular(); desenharCorrecoes();
    requestAnimationFrame(desenharSerie);
  }
  let rz; addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => { if ($("#v-painel").classList.contains("ativa")) desenharSerie(); }, 150); });

  // =====================================================================
  // TESTAR
  // =====================================================================
  const EXEMPLOS = [
    "Comprei faz duas semanas e até agora não chegou, ninguém responde meu e-mail",
    "Produto excelente, chegou antes do prazo e muito bem embalado. Recomendo!",
    "O produto é bonito, mas veio uma peça a menos e a cor é diferente da foto",
    "veio quebrado o vidro do porta retrato, quero a troca",
    "Paguei e cancelaram meu pedido, quero meu dinheiro de volta",
  ];
  $("#t-exemplos").innerHTML = EXEMPLOS.map(t => `<button>${esc(t)}</button>`).join("");
  $$("#t-exemplos button").forEach((b, i) => b.onclick = () => { $("#t-texto").value = EXEMPLOS[i]; testar(); });
  async function testar() {
    const t = $("#t-texto").value.trim();
    if (!t) { $("#t-texto").focus(); return; }
    $("#t-resultado").innerHTML = `<div class="carregando"><div class="spin"></div></div>`;
    try {
      const r = await api("/api/analisar", { texto: t });
      $("#t-resultado").innerHTML = detalheHTML({ id: "teste", texto: t }, r, { teste: true });
      $("#b-copiar").onclick = async () => toast(await copiar($("#resp-texto").value) ? "Resposta copiada." : "Não deu para copiar.");
    } catch (e) { $("#t-resultado").innerHTML = `<div class="cartao">Erro: ${esc(e.message)}</div>`; }
  }
  $("#b-testar").onclick = testar;
  $("#t-texto").addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) testar(); });
  $("#api-exemplo").textContent =
`curl -X POST ${location.origin}/api/analisar \\
  -H "Content-Type: application/json" \\
  -d '{"texto": "não recebi meu pedido"}'

# resposta (resumida)
{ "probabilidade": 0.99, "prioridade": "Alta", "motivo": "Entrega / atraso",
  "prazo": "Responder em até 2 horas", "resposta": "Olá! Sentimos muito..." }`;

  // =====================================================================
  // COMO FUNCIONA
  // =====================================================================
  function desenharSobre() {
    const c = S.cfg, m = c.metricas_teste, ft = c.faixas_teste;
    const arq = c.arquitetura === "bilstm" ? "BiLSTM" : "CNN 1D";
    $("#nav-arq").textContent = arq;
    $("#sobre").innerHTML = `
      <header class="cab"><div><h1 id="t-sobre">Como funciona</h1><p class="sub">Uma rede neural própria, treinada do zero com avaliações reais de e-commerce brasileiro.</p></div></header>
      <div class="passos-ia">
        <div><small>1</small><b>Normaliza</b><span>minúsculas, sem acento; o "não" é mantido</span></div>
        <div><small>2</small><b>Vocabulário</b><span>${fmt(c.vocabulario)} palavras aprendidas no treino viram números</span></div>
        <div><small>3</small><b>Embedding</b><span>cada palavra vira um vetor de 100 dimensões</span></div>
        <div><small>4</small><b>Rede ${arq}</b><span>lê a frase nos dois sentidos e decide</span></div>
        <div><small>5</small><b>Ação</b><span>prioridade, motivo, prazo e resposta sugerida</span></div>
      </div>
      <div class="cartao">
        <header class="cartao-cab"><h2>Desempenho em mensagens nunca vistas no treino</h2></header>
        <table class="t"><tr><th>Métrica (classe reclamação)</th><th>Valor</th></tr>
          <tr><td>Reclamações encontradas (recall)</td><td>${pct(m["Recall (reclamação)"], 1)}</td></tr>
          <tr><td>Acerto quando diz que é reclamação (precisão)</td><td>${pct(m["Precisão (reclamação)"], 1)}</td></tr>
          <tr><td>F1</td><td>${m["F1 (reclamação)"].toFixed(3).replace(".", ",")}</td></tr>
          <tr><td>Acurácia geral</td><td>${pct(m["Acurácia"], 1)}</td></tr></table>
      </div>
      <div class="cartao">
        <header class="cartao-cab"><h2>Faixas de prioridade</h2></header>
        <table class="t"><tr><th>Prioridade</th><th>% das mensagens</th><th>% que são reclamação</th><th>% de todas as reclamações</th></tr>
          ${["Alta", "Média", "Baixa"].map(f => `<tr><td><span class="pill p-${f}">${f}</span></td><td>${pct(ft[f].fracao)}</td><td>${pct(ft[f].taxa_reclamacao)}</td><td>${pct(ft[f].fracao_das_reclamacoes)}</td></tr>`).join("")}
        </table>
        <p class="nota" style="margin-top:10px">Limiares definidos na validação: alta a partir de ${c.limiares.alta.toFixed(2).replace(".", ",")} (pelo menos 95% de reclamações), baixa abaixo de ${c.limiares.baixa.toFixed(2).replace(".", ",")} (no máximo 2%).</p>
      </div>
      <div class="cartao sobre-txt">
        <h2>Limitações</h2>
        <ul>
          <li>A rede aprendeu com avaliações de e-commerce (Olist, 2016-2018), em que a nota serve de rótulo. Antes de usar em WhatsApp ou SAC, é preciso validar com mensagens reais desses canais.</li>
          <li>O motivo é decidido por regras de palavras-chave: reclamações sem os termos caem em "Outros".</li>
          <li>Frases mistas ("chegou rápido, mas parou de funcionar") ainda confundem a rede.</li>
          <li>As respostas sugeridas são modelos: o atendente revisa antes de enviar.</li>
        </ul>
      </div>`;
  }

  // =====================================================================
  // INÍCIO
  // =====================================================================
  async function iniciar() {
    try {
      const [cfg, caixa, painel] = await Promise.all([api("/api/config"), api("/api/caixa"), api("/api/painel")]);
      S.cfg = cfg; S.painel = painel;
      S.msgs = [...S.importadas, ...caixa];
      desenharFiltros(); desenharCaixa(); desenharSobre();
      const v = (location.hash || "").slice(1);
      if (["painel", "testar", "sobre"].includes(v)) irPara(v);
      else if (matchMedia("(min-width:901px)").matches) { const p = filtradas()[0]; if (p) selecionar(p.id, false); }
    } catch (e) {
      $("#caixa-resumo").textContent = "Não foi possível falar com o servidor. Ele está rodando (python app.py)?";
    }
  }
  iniciar();
})();
