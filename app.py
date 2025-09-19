from flask import Flask, request, jsonify, render_template_string
import os
import json
import numpy as np

# Reuso do pipeline existente
from fact_checker import (
    load_vector_store,
    load_classifier,
    historical_consistency_for_text,
    bert_probability_true_for_text,
    fuse_scores,
    classify_text,
)
from external_sources import verify_with_external_sources

app = Flask(__name__)

# Carregar artefatos uma vez
INDEX, METADATA, VCFG = load_vector_store()
CLF, LE, SBERT, CCFG = load_classifier()

DEFAULTS = {
    "k": 20,
    "w_hist": 1/3,
    "w_bert": 1/3,
    "w_fontes": 1/3,
    # novos padrões de chunking
    "max_tokens": 512,
    "overlap_tokens": 64,
    "hist_agg": "max",   # max ou mean
    "bert_agg": "mean",  # mean ou max
}


def analyze_text_payload(text: str, k: int, w_hist: float, w_bert: float, w_fontes: float,
                          max_tokens: int, overlap_tokens: int, hist_agg: str, bert_agg: str):
    text = (text or "").strip()
    if not text:
        return {"error": "texto vazio"}, 400

    # Consistência histórica e BERT com suporte a chunking (512 tokens por padrão)
    hist_score, neighbors = historical_consistency_for_text(
        INDEX, SBERT, text, k=k, max_tokens=max_tokens, overlap_tokens=overlap_tokens, aggregate=hist_agg
    )

    bert_score_true = bert_probability_true_for_text(
        CLF, LE, SBERT, CCFG, text, max_tokens=max_tokens, overlap_tokens=overlap_tokens, aggregate=bert_agg
    )
    bert_label = "provavelmente verdadeiro" if bert_score_true >= 50.0 else "provavelmente falso"

    fonte_score, fonte_details = verify_with_external_sources(text, SBERT)

    final_score = fuse_scores(hist_score, bert_score_true, fonte_score, w_hist, w_bert, w_fontes)
    final_label, final_conf_alt = classify_text(final_score)

    payload = {
        "texto_analisado": text,
        "historico": {
            "consistencia": round(hist_score, 1),
            "k": k,
            "vizinhos": neighbors,
            "aggregate": hist_agg,
        },
        "bert": {
            "rotulo": bert_label,
            "prob_true": round(bert_score_true, 1),
            "aggregate": bert_agg,
        },
        "confirmacao_fontes": {
            "fonte_score": round(fonte_score, 1),
            "detalhes": fonte_details,
        },
        "final": {
            "rotulo": final_label,
            "score": round(final_score, 1),
        },
        "pesos": {"historico": w_hist, "bert": w_bert, "fontes": w_fontes},
        "chunking": {
            "max_tokens": max_tokens,
            "overlap_tokens": overlap_tokens,
        },
    }
    return payload, 200


@app.route("/health", methods=["GET"]) 
def health():
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"]) 
def analyze():
    data = {}
    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        # fallback para form-urlencoded
        data = request.form.to_dict()

    text = data.get("text") or data.get("texto") or ""

    def _get_num(key, typ, default):
        try:
            return typ(data.get(key, default))
        except Exception:
            return default

    k = _get_num("k", int, DEFAULTS["k"])
    w_hist = _get_num("w_hist", float, DEFAULTS["w_hist"])
    w_bert = _get_num("w_bert", float, DEFAULTS["w_bert"])
    w_fontes = _get_num("w_fontes", float, DEFAULTS["w_fontes"])

    max_tokens = _get_num("max_tokens", int, DEFAULTS["max_tokens"])
    overlap_tokens = _get_num("overlap_tokens", int, DEFAULTS["overlap_tokens"])

    hist_agg = str(data.get("hist_agg", DEFAULTS["hist_agg"]))
    if hist_agg not in ("max", "mean"):
        hist_agg = DEFAULTS["hist_agg"]
    bert_agg = str(data.get("bert_agg", DEFAULTS["bert_agg"]))
    if bert_agg not in ("max", "mean"):
        bert_agg = DEFAULTS["bert_agg"]

    result, status = analyze_text_payload(
        text=text,
        k=k,
        w_hist=w_hist,
        w_bert=w_bert,
        w_fontes=w_fontes,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        hist_agg=hist_agg,
        bert_agg=bert_agg,
    )
    return jsonify(result), status


@app.route("/analyze_friendly", methods=["POST"])
def analyze_friendly():
    data = {}
    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form.to_dict()

    text = data.get("text") or data.get("texto") or ""

    if not text.strip():
        return jsonify({"error": "Por favor, insira um texto para análise."}), 400

    k = int(data.get("k", DEFAULTS["k"]))
    w_hist = float(data.get("w_hist", DEFAULTS["w_hist"]))
    w_bert = float(data.get("w_bert", DEFAULTS["w_bert"]))
    w_fontes = float(data.get("w_fontes", DEFAULTS["w_fontes"]))
    max_tokens = int(data.get("max_tokens", DEFAULTS["max_tokens"]))
    overlap_tokens = int(data.get("overlap_tokens", DEFAULTS["overlap_tokens"]))
    hist_agg = data.get("hist_agg", DEFAULTS["hist_agg"]).lower()
    bert_agg = data.get("bert_agg", DEFAULTS["bert_agg"]).lower()

    result, status = analyze_text_payload(
        text=text,
        k=k,
        w_hist=w_hist,
        w_bert=w_bert,
        w_fontes=w_fontes,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        hist_agg=hist_agg,
        bert_agg=bert_agg,
    )

    if status != 200:
        return jsonify(result), status

    friendly_output = {
        "Texto Analisado": result["texto_analisado"],
        "Consistência Histórica": f"{result['historico']['consistencia']}%",
        "Análise BERT": {
            "Rótulo": result["bert"]["rotulo"],
            "Probabilidade de Verdadeiro": f"{result['bert']['prob_true']}%",
        },
        "Confirmação por Fontes Externas": f"{result['confirmacao_fontes']['fonte_score']}%",
        "Classificação Final": {
            "Rótulo": result["final"]["rotulo"],
            "Score": f"{result['final']['score']}%",
        },
    }

    return jsonify(friendly_output), 200


@app.route("/", methods=["GET"]) 
def index():
    html = """
    <!doctype html>
    <meta charset=\"utf-8\" />
    <title>GoldenFred - Verificador</title>
    <style>
      :root { --c-bg:#f5f7fa; --c-card:#ffffff; --c-border:#d0d7de; --c-accent:#2563eb; --c-text:#1f2328; --c-bad:#dc2626; --c-good:#059669; }
      body { font-family: system-ui, Arial, sans-serif; max-width: 960px; margin: 1.5rem auto; padding: 0 1rem; background: var(--c-bg); color: var(--c-text); }
      h1 { margin-top: .2rem; font-size: 1.6rem; }
      textarea { width: 100%; min-height: 180px; resize: vertical; padding:.75rem; border:1px solid var(--c-border); border-radius:6px; font-size: .95rem; }
      .panel, .card { background: var(--c-card); border:1px solid var(--c-border); border-radius:8px; padding:1rem 1.25rem; margin-bottom:1rem; box-shadow: 0 2px 4px rgba(0,0,0,.04); }
      .row { display:flex; flex-wrap:wrap; gap:.75rem; margin:.5rem 0; }
      label { font-size:.75rem; text-transform:uppercase; letter-spacing:.5px; display:flex; flex-direction:column; gap:.2rem; min-width:110px; }
      input, select { padding:.45rem .5rem; border:1px solid var(--c-border); border-radius:5px; background:#fff; font-size:.8rem; }
      button { cursor:pointer; background: var(--c-accent); color:#fff; border:none; padding:.7rem 1.1rem; border-radius:6px; font-weight:600; font-size:.85rem; display:inline-flex; align-items:center; gap:.4rem; }
      button:disabled { opacity:.5; cursor:not-allowed; }
      .results-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:.85rem; margin-top:.75rem; }
      .metric { border:1px solid var(--c-border); border-radius:8px; padding:.6rem .75rem .75rem; background:#fff; position:relative; }
      .metric h3 { margin:0 0 .35rem; font-size:.8rem; font-weight:600; letter-spacing:.5px; text-transform:uppercase; color:#555; }
      .pct { font-size:1.35rem; font-weight:700; }
      .bar { height:6px; border-radius:4px; background:linear-gradient(90deg,var(--c-accent),var(--c-good)); margin-top:.4rem; position:relative; overflow:hidden; }
      .bar span { position:absolute; top:0; left:0; bottom:0; width:0%; background: linear-gradient(90deg,var(--c-accent),var(--c-good)); transition: width .6s ease; }
      .final { border:2px solid var(--c-accent); }
      .status { font-size:.75rem; font-weight:600; margin-top:.3rem; }
      .status.good { color: var(--c-good); }
      .status.bad { color: var(--c-bad); }
      #detailsBox { display:none; margin-top:1rem; }
      pre { background:#0f172a; color:#f1f5f9; padding:1rem; border-radius:8px; font-size:.75rem; line-height:1.25rem; overflow:auto; max-height:420px; }
      .flex-between { display:flex; justify-content:space-between; align-items:center; gap:.75rem; }
      .small { font-size:.7rem; color:#555; }
      a.inline { font-size:.7rem; color:var(--c-accent); text-decoration:none; }
      a.inline:hover { text-decoration:underline; }
      .fade { animation: fade .35s ease; }
      @keyframes fade { from{opacity:0; transform:translateY(4px);} to{opacity:1; transform:translateY(0);} }
    </style>
    <h1>GoldenFred</h1>
    <p class=\"small\">Ferramenta experimental de apoio à verificação. Não substitui checagem humana. Forneça uma afirmação ou pequeno texto.</p>

    <div class=\"panel\">
      <label style=\"display:block; font-weight:600; margin-bottom:.5rem; font-size:.8rem; text-transform:uppercase; letter-spacing:.5px;\">Texto para análise</label>
      <textarea id=\"text\" placeholder=\"Ex: O Brasil é o maior exportador mundial de café.\"></textarea>
      <div class=\"row\" style=\"margin-top:.5rem;\">
        <label>k
          <input id=\"k\" type=\"number\" value=\"20\" min=\"1\" max=\"100\" />
        </label>
        <label>Peso histórico
          <input id=\"w_hist\" type=\"number\" step=\"0.05\" value=\"0.333\" />
        </label>
        <label>Peso BERT
          <input id=\"w_bert\" type=\"number\" step=\"0.05\" value=\"0.333\" />
        </label>
        <label>Peso fontes
          <input id=\"w_fontes\" type=\"number\" step=\"0.05\" value=\"0.333\" />
        </label>
        <label>Hist agg
          <select id=\"hist_agg\"><option>max</option><option>mean</option></select>
        </label>
        <label>BERT agg
          <select id=\"bert_agg\"><option>mean</option><option>max</option></select>
        </label>
      </div>
      <div class=\"row\">
        <label>max_tokens
          <input id=\"max_tokens\" type=\"number\" value=\"512\" />
        </label>
        <label>overlap_tokens
          <input id=\"overlap_tokens\" type=\"number\" value=\"64\" />
        </label>
      </div>
      <div class=\"flex-between\" style=\"margin-top:.75rem;\">
        <div>
          <button id=\"run\">Analisar</button>
          <button id=\"toggleRaw\" style=\"background:#64748b;\" disabled>Mostrar detalhes</button>
        </div>
        <div class=\"small\">Endpoint bruto: <code>POST /analyze</code></div>
      </div>
    </div>

    <div id=\"summaryArea\" style=\"display:none;\" class=\"fade\">
      <div class=\"card\">
        <h2 style=\"margin:0 0 .75rem; font-size:1.05rem;\">Resultados</h2>
        <div class=\"results-grid\">
          <div class=\"metric\" id=\"mHist\">
            <h3>Consistência Histórica</h3>
            <div class=\"pct\" id=\"pctHist\">--%</div>
            <div class=\"bar\"><span id=\"barHist\"></span></div>
            <div class=\"status\" id=\"stHist\"></div>
          </div>
          <div class=\"metric\" id=\"mBert\">
            <h3>BERT (Prob. Verdadeiro)</h3>
            <div class=\"pct\" id=\"pctBert\">--%</div>
            <div class=\"bar\"><span id=\"barBert\"></span></div>
            <div class=\"status\" id=\"stBert\"></div>
          </div>
          <div class=\"metric\" id=\"mFontes\">
            <h3>Fontes Externas</h3>
            <div class=\"pct\" id=\"pctFontes\">--%</div>
            <div class=\"bar\"><span id=\"barFontes\"></span></div>
            <div class=\"status\" id=\"stFontes\"></div>
          </div>
          <div class=\"metric final\" id=\"mFinal\" style=\"grid-column: span 1;\">
            <h3>Média / Score Final</h3>
            <div class=\"pct\" id=\"pctFinal\">--%</div>
            <div class=\"bar\"><span id=\"barFinal\"></span></div>
            <div class=\"status\" id=\"stFinal\"></div>
          </div>
        </div>
        <p class=\"small\" style=\"margin-top:1rem;\">As porcentagens resultam de três sinais combinados pelos pesos definidos. Use como indicação inicial, não veredicto definitivo.</p>
      </div>
    </div>

    <div id=\"detailsBox\" class=\"card fade\">
      <div class=\"flex-between\" style=\"margin-bottom:.5rem;\">
        <strong>Detalhes completos (JSON)</strong>
        <button id=\"closeRaw\" style=\"background:#e11d48;\">Fechar</button>
      </div>
      <pre id=\"out\">{}</pre>
    </div>

    <script>
      const el = id => document.getElementById(id);
      const fmt = v => (v===null||v===undefined||isNaN(v)? '--' : (Math.round(v*10)/10) + '%');
      function paint(idBar, idPct, idStatus, val){
        el(idPct).textContent = fmt(val);
        el(idBar).style.width = (val||0) + '%';
        if(val===null || isNaN(val)) { el(idStatus).textContent=''; return; }
        let cls=''; let txt='';
        if(val >= 66){ cls='good'; txt='alto'; }
        else if(val >= 40){ cls=''; txt='moderado'; }
        else { cls='bad'; txt='baixo'; }
        el(idStatus).className='status ' + cls;
        el(idStatus).textContent = 'Nível ' + txt;
      }
      function toggleDetails(show){
        el('detailsBox').style.display = show? 'block':'none';
        el('toggleRaw').textContent = show? 'Ocultar detalhes':'Mostrar detalhes';
      }
      el('toggleRaw').onclick = ()=>{
        if(el('detailsBox').style.display==='none'){ toggleDetails(true); } else { toggleDetails(false);} };
      el('closeRaw').onclick = ()=> toggleDetails(false);

      el('run').onclick = async () => {
        const text = el('text').value.trim();
        if(!text){ alert('Insira um texto.'); return; }
        el('run').disabled = true; el('run').textContent='Analisando...'; toggleDetails(false); el('toggleRaw').disabled = true;
        const body = {
          text,
          k: Number(el('k').value),
          w_hist: Number(el('w_hist').value),
          w_bert: Number(el('w_bert').value),
          w_fontes: Number(el('w_fontes').value),
          max_tokens: Number(el('max_tokens').value),
          overlap_tokens: Number(el('overlap_tokens').value),
          hist_agg: el('hist_agg').value,
          bert_agg: el('bert_agg').value,
        };
        try {
          const res = await fetch('/analyze', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
          const json = await res.json();
          if(res.status !== 200){ alert(json.error || 'Erro na análise'); } else {
            el('summaryArea').style.display='block';
            const h = json.historico?.consistencia;
            const b = json.bert?.prob_true;
            const f = json.confirmacao_fontes?.fonte_score;
            const fin = json.final?.score;
            paint('barHist','pctHist','stHist', h);
            paint('barBert','pctBert','stBert', b);
            paint('barFontes','pctFontes','stFontes', f);
            paint('barFinal','pctFinal','stFinal', fin);
            el('out').textContent = JSON.stringify(json, null, 2);
            el('toggleRaw').disabled = false;
          }
        } catch(e){
          alert('Falha na requisição. Veja console.'); console.error(e);
        } finally {
          el('run').disabled = false; el('run').textContent='Analisar';
        }
      };
    </script>
    """
    return render_template_string(html)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=port, debug=debug)
