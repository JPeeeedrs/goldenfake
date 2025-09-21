from flask import Flask, request, jsonify
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
    load_online_adaptor,
    combined_historical_consistency_for_text,
)
from external_sources import verify_with_external_sources

app = Flask(__name__)

# Habilitar CORS para desenvolvimento (inclui file:// como origem nula)
try:
    from flask_cors import CORS
    CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False, methods=["GET","POST","OPTIONS"], allow_headers=["Content-Type"], expose_headers=["Content-Type"], send_wildcard=True)
except Exception:
    pass

# Carregar artefatos uma vez
INDEX, METADATA, VCFG = load_vector_store()
CLF, LE, SBERT, CCFG = load_classifier()
ONLINE = load_online_adaptor(LE, SBERT, CCFG)

DEFAULTS = {
    "k": 20,
    "w_hist": 0.5,
    "w_bert": 0.35,
    "w_fontes": 0.33,
    # novos padrões de chunking
    "max_tokens": 512,
    "overlap_tokens": 64,
    "hist_agg": "max",   # max ou mean
    "bert_agg": "mean",  # mean ou max
    # Wikipedia dinâmico (opcional)
    "use_wiki": False,
    "wiki_titles": 3,
    "w_faiss": 0.5,
    "w_wiki": 0.5,
}


def _negative_label_from_le(le) -> str:
    classes = list(getattr(le, "classes_", []))
    if not classes:
        return "false"
    if "true" in classes and len(classes) == 2:
        return classes[0] if classes[1] == "true" else classes[1]
    # fallback: use first non-true
    for c in classes:
        if c != "true":
            return c
    return classes[0]


def _as_bool(val, default=False):
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "y", "on"): return True
    if s in ("0", "false", "no", "n", "off"): return False
    return default


def _flatten_external_evidence(details):
    items = []
    seen = set()
    for cl in details or []:
        for ev in (cl.get("evidencias") or []):
            url = ev.get("url") or ""
            key = url or f"{ev.get('title') or ''}|{ev.get('publisher') or ''}"
            if key in seen:
                continue
            seen.add(key)
            sc = ev.get("score")
            try:
                pct = round(float(sc) * 100.0, 1)
            except Exception:
                continue
            rating = ev.get("rating")
            fc_flag = bool(ev.get("fact_checker")) or (isinstance(rating, str) and "fact-check" in rating.lower())
            items.append({
                "title": ev.get("title"),
                "url": url,
                "publisher": ev.get("publisher"),
                "provider": ev.get("provider"),
                "percent": pct,
                "similaridade": ev.get("similaridade"),
                "confianca_fonte": ev.get("confianca_fonte"),
                "overlap_bucket": ev.get("overlap_bucket") or (
                    ">=50%" if ev.get("passes_50pct") else ("40-49%" if ev.get("passes_40pct") else "<40%")
                ),
                "is_social": ev.get("is_social", False),
                "fact_checker": fc_flag,
                "rating": rating,
                # include backend-generated source tags for UI badges
                "source_tags": ev.get("source_tags") or [],
            })
    items.sort(key=lambda x: x.get("percent", 0), reverse=True)
    return items


def analyze_text_payload(text: str, k: int, w_hist: float, w_bert: float, w_fontes: float,
                          max_tokens: int, overlap_tokens: int, hist_agg: str, bert_agg: str,
                          use_wiki: bool, wiki_titles: int, w_faiss: float, w_wiki: float):
    text = (text or "").strip()
    if not text:
        return {"error": "texto vazio"}, 400

    # Consistência histórica (FAISS) ou combinação com Wikipedia
    hist_info = {}
    if use_wiki:
        hist_score, hist_info = combined_historical_consistency_for_text(
            INDEX, SBERT, text,
            k=k,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            aggregate=hist_agg,
            wiki_titles=wiki_titles,
            w_faiss=w_faiss,
            w_wiki=w_wiki,
        )
        neighbors = hist_info.get("faiss", {}).get("vizinhos", [])
    else:
        hist_score, neighbors = historical_consistency_for_text(
            INDEX, SBERT, text, k=k, max_tokens=max_tokens, overlap_tokens=overlap_tokens, aggregate=hist_agg
        )

    bert_score_true = bert_probability_true_for_text(
        CLF, LE, SBERT, CCFG, text, max_tokens=max_tokens, overlap_tokens=overlap_tokens, aggregate=bert_agg
    )
    bert_label = "provavelmente verdadeiro" if bert_score_true >= 50.0 else "provavelmente falso"

    fonte_score, fonte_details = verify_with_external_sources(text, SBERT)
    fontes_individuais = _flatten_external_evidence(fonte_details)

    # Adaptador online (se já houver updates)
    online_prob = ONLINE.prob_true_for_text(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens, aggregate=bert_agg)
    alpha_online = ONLINE.alpha() if ONLINE.is_trained() else 0.0

    # Primeiro, média ponderada entre histórico, bert e fontes (reduzindo peso de BERT pelo (1-alpha))
    base_final = fuse_scores(hist_score, bert_score_true, fonte_score, w_hist, w_bert * (1 - alpha_online), w_fontes)
    # Em seguida, mesclar com o adaptador online conforme alpha
    final_score = fuse_scores(base_final, (online_prob if online_prob is not None else 0.0), None, 1 - alpha_online, alpha_online, 0.0)
    final_label, _ = classify_text(final_score)

    historico_block = {
        "consistencia": round(hist_score, 1),
        "k": k,
        "aggregate": hist_agg,
    }
    if use_wiki:
        historico_block.update({
            "faiss": hist_info.get("faiss"),
            "wikipedia": hist_info.get("wikipedia"),
            "pesos_hist": hist_info.get("pesos"),
        })
    else:
        historico_block["vizinhos"] = neighbors

    payload = {
        "texto_analisado": text,
        "historico": historico_block,
        "bert": {
            "rotulo": bert_label,
            "prob_true": round(bert_score_true, 1),
            "aggregate": bert_agg,
        },
        "confirmacao_fontes": {
            "fonte_score": round(fonte_score, 1),
            "detalhes": fonte_details,
            "fontes_individuais": fontes_individuais,
        },
        "online_adaptor": {
            "prob_true": (round(online_prob, 1) if online_prob is not None else None),
            "alpha": round(alpha_online, 2),
            "n_updates": getattr(ONLINE, "n_updates", 0),
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

    use_wiki = _as_bool(data.get("use_wiki", DEFAULTS["use_wiki"]))
    wiki_titles = _get_num("wiki_titles", int, DEFAULTS["wiki_titles"])
    w_faiss = _get_num("w_faiss", float, DEFAULTS["w_faiss"])
    w_wiki = _get_num("w_wiki", float, DEFAULTS["w_wiki"])

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
        use_wiki=use_wiki,
        wiki_titles=wiki_titles,
        w_faiss=w_faiss,
        w_wiki=w_wiki,
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

    use_wiki = _as_bool(data.get("use_wiki", DEFAULTS["use_wiki"]))
    wiki_titles = int(data.get("wiki_titles", DEFAULTS["wiki_titles"]))
    w_faiss = float(data.get("w_faiss", DEFAULTS["w_faiss"]))
    w_wiki = float(data.get("w_wiki", DEFAULTS["w_wiki"]))

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
        use_wiki=use_wiki,
        wiki_titles=wiki_titles,
        w_faiss=w_faiss,
        w_wiki=w_wiki,
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
        "Adaptador Online": {
            "Prob Verdadeiro": (f"{result['online_adaptor']['prob_true']}%" if result['online_adaptor']['prob_true'] is not None else None),
            "Alpha": result['online_adaptor']['alpha'],
            "Atualizações": result['online_adaptor']['n_updates'],
        },
        "Classificação Final": {
            "Rótulo": result["final"]["rotulo"],
            "Score": f"{result['final']['score']}%",
        },
    }

    return jsonify(friendly_output), 200


@app.route("/train_online", methods=["POST"]) 
def train_online():
    """
    Atualiza o adaptador online com textos enviados pelo front.
    Request JSON/body:
    {
      text: string | undefined,
      texts: [string] | undefined,
      label: "true"|"false"|"fake" | undefined,
      labels: [..] | undefined,
      auto_label: bool (default true, usa BERT atual),
      threshold: number (0-1 ou 0-100; default 0.7),
      max_tokens, overlap_tokens, bert_agg (opcionais para auto_label)
    }
    """
    data = request.get_json(silent=True) or request.form.to_dict(flat=True) or {}

    # Coletar textos
    texts = []
    if isinstance(data.get("texts"), list):
        texts = [str(t) for t in data.get("texts")]
    elif data.get("texts") and isinstance(data.get("texts"), str):
        # permitir CSV simples
        texts = [s for s in data.get("texts").split("\n") if s.strip()]
    if data.get("text"):
        texts.append(str(data.get("text")))
    # limpar
    texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]

    if not texts:
        return jsonify({"updated": 0, "error": "nenhum texto fornecido"}), 400

    # Labels (opcionais)
    labels_in = None
    if isinstance(data.get("labels"), list):
        labels_in = [str(x).lower().strip() for x in data.get("labels")]
    elif data.get("label"):
        labels_in = [str(data.get("label")).lower().strip()] * len(texts)

    auto_label = True if str(data.get("auto_label", "true")).lower() in ("1", "true", "yes", "y") else False
    # Interpretar threshold em 0-1 ou 0-100
    thr_raw = data.get("threshold", 0.7)
    try:
        thr = float(thr_raw)
    except Exception:
        thr = 0.7
    if (thr > 1.0):
        thr = thr / 100.0
    thr = float(np.clip(thr, 0.5, 0.99))

    max_tokens = int(data.get("max_tokens", DEFAULTS["max_tokens"])) if str(data.get("max_tokens", "")).strip() else DEFAULTS["max_tokens"]
    overlap_tokens = int(data.get("overlap_tokens", DEFAULTS["overlap_tokens"])) if str(data.get("overlap_tokens", "")).strip() else DEFAULTS["overlap_tokens"]
    bert_agg = str(data.get("bert_agg", DEFAULTS["bert_agg"]))
    if bert_agg not in ("mean", "max"):
        bert_agg = DEFAULTS["bert_agg"]

    neg_label = _negative_label_from_le(LE)

    # Se labels não foram fornecidas e auto_label ativo, gerar a partir do BERT atual
    generated_labels = None
    if (not labels_in) and auto_label:
        generated_labels = []
        for t in texts:
            p = bert_probability_true_for_text(CLF, LE, SBERT, CCFG, t, max_tokens=max_tokens, overlap_tokens=overlap_tokens, aggregate=bert_agg)
            lab = "true" if (p >= thr * 100.0) else neg_label
            generated_labels.append(lab)
        labels_in = generated_labels

    if not labels_in or len(labels_in) != len(texts):
        return jsonify({"updated": 0, "error": "labels ausentes ou tamanho incorreto; ative auto_label ou envie labels."}), 400

    # Normalizar labels para os rótulos do encoder
    allowed = set(list(getattr(LE, "classes_", [])))
    labels_final = []
    for lab in labels_in:
        if lab == "true":
            labels_final.append("true" if "true" in allowed else list(allowed)[0])
        else:
            # mapear qualquer não-true para o rótulo negativo presente
            labels_final.append(neg_label if neg_label in allowed else list(allowed)[0])

    try:
        ok = ONLINE.update(texts, labels_final)
    except Exception as e:
        return jsonify({"updated": 0, "error": f"falha no update: {type(e).__name__}: {e}"}), 500

    return jsonify({
        "updated": int(ok) * len(texts),
        "n_updates_total": getattr(ONLINE, "n_updates", 0),
        "alpha": ONLINE.alpha() if ONLINE.is_trained() else 0.0,
        "labels_used": labels_final,
        "auto_label": auto_label,
        "threshold": thr,
    }), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=port, debug=debug)
