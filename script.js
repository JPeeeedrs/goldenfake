const el = (id) => document.getElementById(id);
const fmt = (v) =>
	v === null || v === undefined || isNaN(v)
		? "--"
		: Math.round(v * 10) / 10 + "%";

function paint(idBar, idPct, idStatus, val) {
	el(idPct).textContent = fmt(val);
	el(idBar).style.width = (val || 0) + "%";
	if (val === null || isNaN(val)) {
		el(idStatus).textContent = "";
		return;
	}
	let cls = "";
	let txt = "";
	if (val >= 66) {
		cls = "good";
		txt = "alto";
	} else if (val >= 40) {
		cls = "";
		txt = "moderado";
	} else {
		cls = "bad";
		txt = "baixo";
	}
	el(idStatus).className = "status " + cls;
	el(idStatus).textContent = "Nível " + txt;
}

function toggleDetails(show) {
	el("detailsBox").style.display = show ? "block" : "none";
	el("toggleRaw").textContent = show ? "Ocultar detalhes" : "Mostrar detalhes";
}

function getApiBase() {
	const manual = el("apiBase").value.trim().replace(/\/$/, "");
	if (manual) return manual;
	if (location.protocol === "file:") return "http://localhost:5000";
	return "";
}

function renderSources(list) {
	const card = el("sourcesCard");
	const cont = el("sourcesList");
	cont.innerHTML = "";
	if (!Array.isArray(list) || list.length === 0) {
		card.style.display = "none";
		return;
	}
	list.forEach((item) => {
		const row = document.createElement("div");
		row.className = "source-item";
		const left = document.createElement("div");
		const title = document.createElement("div");
		title.textContent = item.title || item.url || "(sem título)";
		const meta = document.createElement("div");
		meta.className = "meta";
		meta.textContent =
			(item.publisher || item.provider || "fonte") +
			" • " +
			(item.overlap_bucket || "");
		// badges de classificação: usar source_tags do backend
		const tags = Array.isArray(item.source_tags) ? item.source_tags : [];
		tags.forEach((tg) => {
			const b = document.createElement("span");
			b.className =
				"badge tag-" + tg.replace(/[^a-z0-9\-]/gi, "").toLowerCase();
			b.textContent = tg;
			meta.appendChild(b);
		});
		left.appendChild(title);
		left.appendChild(meta);

		const right = document.createElement("div");
		right.className = "percent";
		right.textContent =
			(item.percent != null ? item.percent.toFixed(1) : "--") + "%";

		row.appendChild(left);
		row.appendChild(right);
		const tipTags = tags && tags.length ? ` | tags: ${tags.join(", ")}` : "";
		row.title = `similaridade: ${item.similaridade ?? "--"} | confiança: ${
			item.confianca_fonte ?? "--"
		}${tipTags}`;
		row.onclick = () => {
			if (item.url) {
				window.open(item.url, "_blank");
			}
		};
		cont.appendChild(row);
	});
	card.style.display = "block";
}

document.addEventListener("DOMContentLoaded", () => {
	if (location.protocol === "file:")
		el("apiBase").placeholder = "http://localhost:5000";

	el("toggleRaw").onclick = () => {
		if (el("detailsBox").style.display === "none") {
			toggleDetails(true);
		} else {
			toggleDetails(false);
		}
	};
	el("closeRaw").onclick = () => toggleDetails(false);

	el("run").onclick = async () => {
		const text = el("text").value.trim();
		if (!text) {
			alert("Insira um texto.");
			return;
		}
		el("run").disabled = true;
		el("run").textContent = "Analisando...";
		toggleDetails(false);
		el("toggleRaw").disabled = true;
		const body = {
			text,
			k: Number(el("k").value),
			w_hist: Number(el("w_hist").value),
			w_bert: Number(el("w_bert").value),
			w_fontes: Number(el("w_fontes").value),
			max_tokens: Number(el("max_tokens").value),
			overlap_tokens: Number(el("overlap_tokens").value),
			hist_agg: el("hist_agg").value,
			bert_agg: el("bert_agg").value,
			use_wiki: el("use_wiki").value === "true",
			wiki_titles: Number(el("wiki_titles").value),
			w_faiss: Number(el("w_faiss").value),
			w_wiki: Number(el("w_wiki").value),
		};
		const base = getApiBase();
		try {
			const res = await fetch(base + "/analyze", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(body),
			});
			const json = await res.json();
			if (res.status !== 200) {
				alert(json.error || "Erro na análise");
			} else {
				el("summaryArea").style.display = "block";
				const h = json.historico?.consistencia;
				const b = json.bert?.prob_true;
				const f = json.confirmacao_fontes?.fonte_score;
				const fin = json.final?.score;
				paint("barHist", "pctHist", "stHist", h);
				paint("barBert", "pctBert", "stBert", b);
				paint("barFontes", "pctFontes", "stFontes", f);
				paint("barFinal", "pctFinal", "stFinal", fin);
				el("out").textContent = JSON.stringify(json, null, 2);
				const fontes = json.confirmacao_fontes?.fontes_individuais || [];
				renderSources(fontes);
				el("toggleRaw").disabled = false;
			}
		} catch (e) {
			alert(
				"Falha na requisição. Verifique a URL da API e CORS. Veja console."
			);
			console.error(e);
		} finally {
			el("run").disabled = false;
			el("run").textContent = "Analisar";
		}
	};
});
