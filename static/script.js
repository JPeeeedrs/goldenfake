const el = (id) => document.getElementById(id);
const fmt = (v) =>
	v === null || v === undefined || isNaN(v)
		? "--"
		: Math.round(v * 10) / 10 + "%";
const fmtRatio = (ratio) =>
	ratio === null || ratio === undefined || isNaN(ratio)
		? "--"
		: `${(ratio * 100).toFixed(1)}%`;

const DEFAULT_PARAMS = {
	k: 20,
	max_tokens: 512,
	overlap_tokens: 128,
	w_hist: 0.4,
	w_bert: 0.3,
	w_fontes: 0.3,
	hist_agg: "max",
	bert_agg: "mean",
};

function applyDefaultValues() {
	Object.entries(DEFAULT_PARAMS).forEach(([key, val]) => {
		const field = el(key);
		if (!field) return;
		field.value = val;
	});
}

// --- Pie chart (Chart.js) support ---
let pieChartInstance = null;
let barChartInstance = null;
let sourcePieInstance = null;
let sourceBarInstance = null;
let wikiCategoryChart = null;
let hasSourceList = false;
let hasWikiList = false;
let hasEntityList = false;
let copyFeedbackTimer = null;
const pieColors = ["#0ea5e9", "#22c55e", "#f59e0b"]; // 3 cores (Histórico, BERT, Fontes)
const barColors = ["#38bdf8", "#34d399", "#fbbf24", "#f97316"];
const sourceColors = ["#fbbf24", "#38bdf8", "#a855f7", "#22d3ee", "#fb7185"];
const chartLabels = [
	"Registros passados",
	"Modelo semântico",
	"Fontes externas",
	"Resultado final",
];
const ENTITY_STATUS_CLASS = {
	strong: "good",
	weak: "warn",
	missing: "bad",
};

function ensurePieCtx() {
	const canvas = el("pieChart");
	const fallback = el("pieFallback");
	if (!canvas) return null;
	if (typeof window.Chart === "undefined") {
		if (fallback) fallback.style.display = "block";
		canvas.style.display = "none";
		return null;
	}
	if (fallback) fallback.style.display = "none";
	canvas.style.display = "block";
	const ctx = canvas.getContext("2d");
	return ctx;
}

function ensureBarCtx() {
	const canvas = el("barChart");
	const fallback = el("barFallback");
	if (!canvas) return null;
	if (typeof window.Chart === "undefined") {
		if (fallback) fallback.style.display = "block";
		canvas.style.display = "none";
		return null;
	}
	if (fallback) fallback.style.display = "none";
	canvas.style.display = "block";
	return canvas.getContext("2d");
}

function updatePieChart(h, b, f) {
	const values = [h, b, f].map((v) =>
		typeof v === "number" && isFinite(v) ? v : 0
	);
	const ctx = ensurePieCtx();
	if (!ctx) return;
	if (!pieChartInstance) {
		pieChartInstance = new Chart(ctx, {
			type: "doughnut",
			data: {
				labels: chartLabels.slice(0, 3),
				datasets: [
					{
						data: values,
						backgroundColor: pieColors,
						borderWidth: 0,
						borderRadius: 6,
					},
				],
			},
			options: {
				responsive: true,
				maintainAspectRatio: false,
				cutout: "58%",
				plugins: {
					legend: {
						position: "bottom",
						labels: { color: "rgba(255,255,255,0.85)", usePointStyle: true },
					},
					tooltip: {
						backgroundColor: "rgba(6,2,10,0.92)",
						borderColor: "rgba(247,201,72,0.5)",
						borderWidth: 1,
					},
				},
			},
		});
	} else {
		pieChartInstance.data.labels = chartLabels.slice(0, 3);
		pieChartInstance.data.datasets[0].data = values;
		pieChartInstance.data.datasets[0].backgroundColor = pieColors;
		pieChartInstance.update();
	}
}

function updateBarChart(h, b, f, finalScore) {
	const values = [h, b, f, finalScore].map((v) =>
		typeof v === "number" && isFinite(v) ? v : 0
	);
	const ctx = ensureBarCtx();
	if (!ctx) return;
	if (!barChartInstance) {
		barChartInstance = new Chart(ctx, {
			type: "bar",
			data: {
				labels: chartLabels,
				datasets: [
					{
						data: values,
						backgroundColor: barColors,
						borderRadius: 12,
						barThickness: 18,
					},
				],
			},
			options: {
				indexAxis: "y",
				maintainAspectRatio: false,
				responsive: true,
				plugins: {
					legend: { display: false },
					tooltip: {
						backgroundColor: "rgba(6,2,10,0.92)",
						borderColor: "rgba(247,201,72,0.5)",
						borderWidth: 1,
					},
				},
				scales: {
					x: {
						beginAtZero: true,
						max: 100,
						grid: { color: "rgba(255,255,255,0.08)" },
						ticks: {
							color: "rgba(255,255,255,0.8)",
							callback: (val) => `${val}%`,
						},
					},
					y: {
						grid: { display: false },
						ticks: { color: "rgba(255,255,255,0.9)" },
					},
				},
			},
		});
	} else {
		barChartInstance.data.labels = chartLabels;
		barChartInstance.data.datasets[0].data = values;
		barChartInstance.update();
	}
}

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

async function copyRawJson() {
	const btn = el("copyRaw");
	const target = el("out");
	if (!btn || !target) return;
	const payload = target.textContent || "";
	if (!payload.trim()) return;
	const original = btn.textContent || "Copiar JSON";
	const showFeedback = (msg) => {
		btn.textContent = msg;
		clearTimeout(copyFeedbackTimer);
		copyFeedbackTimer = setTimeout(() => {
			btn.textContent = original;
			btn.disabled = false;
		}, 1500);
	};
	btn.disabled = true;
	try {
		if (navigator.clipboard?.writeText) {
			await navigator.clipboard.writeText(payload);
		} else {
			const temp = document.createElement("textarea");
			temp.value = payload;
			temp.style.position = "fixed";
			temp.style.opacity = "0";
			document.body.appendChild(temp);
			temp.focus();
			temp.select();
			document.execCommand("copy");
			document.body.removeChild(temp);
		}
		showFeedback("Copiado!");
	} catch (err) {
		console.error("Falha ao copiar JSON", err);
		showFeedback("Erro ao copiar");
	}
}

function getApiBase() {
	return window.location.origin;
}

function resetSourceCharts() {
	[sourcePieInstance, sourceBarInstance].forEach((inst, idx) => {
		if (inst && typeof inst.destroy === "function") {
			inst.destroy();
		}
		if (idx === 0) sourcePieInstance = null;
		if (idx === 1) sourceBarInstance = null;
	});
}

function renderSources(list) {
	const cont = el("sourcesList");
	if (!cont) return;
	cont.innerHTML = "";
	hasSourceList = Array.isArray(list) && list.length > 0;
	if (!hasSourceList) {
		resetSourceCharts();
		updateSourcesCardVisibility();
		return;
	}
	list.slice(0, 10).forEach((item) => {
		const row = document.createElement("div");
		row.className = "source-item";
		const left = document.createElement("div");
		const title = document.createElement("div");
		title.textContent = item.title || item.url || "(sem título)";
		const meta = document.createElement("div");
		meta.className = "meta";
		const baseMeta =
			(item.publisher || item.provider || "fonte") +
			(item.overlap_bucket ? " • " + item.overlap_bucket : "");
		const simLabel =
			typeof item.percent === "number"
				? `${item.percent.toFixed(1)}% similar`
				: null;
		const geminiLabel =
			typeof item.gemini_percent_true === "number"
				? `Gemini ${item.gemini_percent_true.toFixed(1)}%`
				: null;
		const percentLabel = [simLabel, geminiLabel].filter(Boolean).join(" • ");
		meta.textContent = percentLabel
			? `${baseMeta} • ${percentLabel}`
			: baseMeta;
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
		const rightParts = [];
		if (typeof item.percent === "number" && !Number.isNaN(item.percent)) {
			rightParts.push(`${item.percent.toFixed(1)}% similaridade`);
		}
		if (
			typeof item.gemini_percent_true === "number" &&
			!Number.isNaN(item.gemini_percent_true)
		) {
			rightParts.push(`Gemini ${item.gemini_percent_true.toFixed(1)}%`);
		}
		right.textContent = rightParts.length ? rightParts.join(" | ") : "--";

		row.appendChild(left);
		row.appendChild(right);
		const tipTags = tags && tags.length ? ` | tags: ${tags.join(", ")}` : "";
		row.title = `similaridade: ${
			typeof item.percent === "number" ? item.percent.toFixed(1) + "%" : "--"
		} | confiança: ${item.confianca_fonte ?? "--"}${tipTags}`;
		row.onclick = () => {
			if (item.url) {
				window.open(item.url, "_blank");
			}
		};
		cont.appendChild(row);
	});
	updateSourceCharts(list);
	updateSourcesCardVisibility();
}

function renderCorroborationNote(rawHist, adjHist, multiplier, ratio, corrobBlock) {
	const note = el("histCorrNote");
	if (!note) return;
	const hasMultiplier = typeof multiplier === "number" && isFinite(multiplier);
	const hasRatio = typeof ratio === "number" && isFinite(ratio);
	const rawTxt =
		typeof rawHist === "number" && isFinite(rawHist) ? fmt(rawHist) : "--";
	const adjTxt =
		typeof adjHist === "number" && isFinite(adjHist) ? fmt(adjHist) : "--";
	const queryEntities = Array.isArray(corrobBlock?.query_entities)
		? corrobBlock.query_entities
		: [];
	const bestArticle = Array.isArray(corrobBlock?.articles)
		? corrobBlock.articles.find(
				(art) =>
					Array.isArray(art?.matched_entities) && art.matched_entities.length
		  ) || corrobBlock.articles[0]
		: null;
	const matchedEntities = Array.isArray(bestArticle?.matched_entities)
		? bestArticle.matched_entities
		: [];
	const missingEntities = Array.isArray(bestArticle?.missing_entities)
		? bestArticle.missing_entities
		: [];
	if (!hasMultiplier) {
		if (!corrobBlock || queryEntities.length === 0) {
			note.textContent =
				"Corroboração FAISS: não aplicada (nenhuma entidade nomeada encontrada).";
			note.title = "";
			return;
		}
		note.textContent =
			"Corroboração FAISS indisponível (falha ao extrair texto do artigo).";
		note.title = "";
		return;
	}
	const multTxt = fmtRatio(multiplier);
	const ratioTxt = hasRatio ? fmtRatio(ratio) : null;
	const descriptor = ratioTxt
		? `sobreposição ${ratioTxt} • multiplicador ${multTxt}`
		: `multiplicador ${multTxt}`;
	const baseText = `Corroboração FAISS: ${descriptor} (bruto ${rawTxt} → ajustado ${adjTxt}).`;
	const tooltipParts = [];
	if (matchedEntities.length) {
		tooltipParts.push(`Entidades encontradas: ${matchedEntities.join(", ")}`);
	}
	if (missingEntities.length) {
		const miss = missingEntities.slice(0, 5).join(", ");
		tooltipParts.push(`Ausentes: ${miss}`);
	}
	note.textContent = baseText;
	note.title = tooltipParts.join(" | ");
}

function renderEntityVerification(block) {
	const section = el("entityBlock");
	const list = el("entityList");
	const summary = el("entitySummary");
	const entities = Array.isArray(block?.entidades) ? block.entidades : [];
	hasEntityList = entities.length > 0;
	if (!section || !list) return;
	if (!hasEntityList) {
		section.style.display = "none";
		list.innerHTML = "";
		if (summary) summary.textContent = "";
		updateSourcesCardVisibility();
		return;
	}
	section.style.display = "block";
	list.innerHTML = "";
	if (summary) {
		let avgPct = null;
		if (typeof block?.media_percent === "number") {
			avgPct = block.media_percent.toFixed(1);
		} else if (typeof block?.media_score === "number") {
			avgPct = (block.media_score * 100).toFixed(1);
		}
		const avgText = avgPct != null ? `${avgPct}%` : "--";
		const fortes = block?.fortes ?? 0;
		const fracas = block?.fracas ?? 0;
		const ausentes = block?.ausentes ?? 0;
		summary.textContent = `Média: ${avgText} • fortes: ${fortes} | fracas: ${fracas} | ausentes: ${ausentes}`;
	}
	entities.forEach((item) => {
		const row = document.createElement("div");
		row.className = "entity-item";
		const info = document.createElement("div");
		info.className = "info";
		const name = document.createElement("div");
		name.className = "name";
		name.textContent = item.entidade || "(sem nome)";
		const score = document.createElement("div");
		score.className = "small";
		let scoreVal = "--";
		if (typeof item.percent === "number") {
			scoreVal = `${item.percent.toFixed(1)}%`;
		} else if (typeof item.score === "number") {
			scoreVal = `${(item.score * 100).toFixed(1)}%`;
		}
		score.textContent = `Score: ${scoreVal}`;
		info.appendChild(name);
		info.appendChild(score);
		const status = document.createElement("div");
		const cls = ENTITY_STATUS_CLASS[item.status] || "warn";
		status.className = `status-chip ${cls}`;
		status.textContent = item.rotulo || item.status || "";
		row.appendChild(info);
		row.appendChild(status);
		if (Array.isArray(item.resultados) && item.resultados.length) {
			const tooltip = item.resultados
				.slice(0, 3)
				.map((res) => res.publisher || res.provider || res.url || "fonte")
				.join(" • ");
			const tipParts = [];
			if (scoreVal !== "--") tipParts.push(`Score ${scoreVal}`);
			if (tooltip) tipParts.push(tooltip);
			row.title = tipParts.join(" | ");
		}
		list.appendChild(row);
	});
	updateSourcesCardVisibility();
}

function buildSourceStats(list) {
	const counts = {};
	list.forEach((item) => {
		const tags = Array.isArray(item.source_tags) ? item.source_tags : [];
		const mainTag = tags[0] || "outros";
		counts[mainTag] = (counts[mainTag] || 0) + 1;
	});
	const entries = Object.entries(counts)
		.sort((a, b) => b[1] - a[1])
		.slice(0, 5);
	return entries.length ? entries : [["outros", list.length]];
}

function ensureSourceCtx(id, fallbackId) {
	const canvas = el(id);
	const fallback = el(fallbackId);
	if (!canvas) return null;
	if (typeof window.Chart === "undefined") {
		if (fallback) fallback.style.display = "block";
		canvas.style.display = "none";
		return null;
	}
	if (fallback) fallback.style.display = "none";
	canvas.style.display = "block";
	return canvas.getContext("2d");
}

function updateSourceCharts(list) {
	const stats = buildSourceStats(list);
	const labels = stats.map(([name]) => name);
	const values = stats.map(([, count]) => count);
	const ctxPie = ensureSourceCtx("sourcePie", "sourcePieFallback");
	const ctxBar = ensureSourceCtx("sourceBar", "sourceBarFallback");
	if (ctxPie) {
		if (!sourcePieInstance) {
			sourcePieInstance = new Chart(ctxPie, {
				type: "doughnut",
				data: {
					labels,
					datasets: [
						{
							data: values,
							backgroundColor: sourceColors,
							borderWidth: 0,
						},
					],
				},
				options: {
					maintainAspectRatio: false,
					plugins: {
						legend: {
							position: "bottom",
							labels: { color: "rgba(255,255,255,0.85)", usePointStyle: true },
						},
					},
					cutout: "55%",
				},
			});
		} else {
			sourcePieInstance.data.labels = labels;
			sourcePieInstance.data.datasets[0].data = values;
			sourcePieInstance.update();
		}
	}
	if (ctxBar) {
		if (!sourceBarInstance) {
			sourceBarInstance = new Chart(ctxBar, {
				type: "bar",
				data: {
					labels,
					datasets: [
						{
							data: values,
							backgroundColor: sourceColors,
							borderRadius: 10,
							barThickness: 16,
						},
					],
				},
				options: {
					maintainAspectRatio: false,
					plugins: { legend: { display: false } },
					scales: {
						y: {
							grid: { display: false },
							ticks: { color: "rgba(255,255,255,0.9)" },
						},
						x: {
							grid: { color: "rgba(255,255,255,0.08)" },
							ticks: { color: "rgba(255,255,255,0.8)" },
						},
					},
				},
			});
		} else {
			sourceBarInstance.data.labels = labels;
			sourceBarInstance.data.datasets[0].data = values;
			sourceBarInstance.update();
		}
	}
}

function ensureWikiCategoryCtx() {
	const canvas = el("wikiCategoryChart");
	const fallback = el("wikiCategoryFallback");
	if (!canvas) return null;
	if (typeof window.Chart === "undefined") {
		if (fallback) fallback.style.display = "block";
		canvas.style.display = "none";
		return null;
	}
	if (fallback) fallback.style.display = "none";
	canvas.style.display = "block";
	return canvas.getContext("2d");
}

function resetWikiCategoryChart() {
	if (wikiCategoryChart && typeof wikiCategoryChart.destroy === "function") {
		wikiCategoryChart.destroy();
	}
	wikiCategoryChart = null;
}

function buildWikiCategoryStats(items) {
	const stats = new Map();
	items.forEach((item) => {
		let categories = Array.isArray(item.categorias)
			? item.categorias
			: item.categoria
			? [item.categoria]
			: [];
		if (!categories.length) {
			categories = ["Sem categoria"];
		}
		categories.forEach((cat) => {
			const current = stats.get(cat) || { count: 0, titles: [] };
			current.count += 1;
			if (current.titles.length < 5) {
				current.titles.push(item.titulo || `Artigo ${item.id}`);
			}
			stats.set(cat, current);
		});
	});
	return Array.from(stats.entries())
		.map(([category, info]) => ({ category, ...info }))
		.sort((a, b) => b.count - a.count);
}

function updateWikiCategoryChart(stats) {
	const ctx = ensureWikiCategoryCtx();
	if (!ctx) {
		resetWikiCategoryChart();
		return;
	}
	const labels = stats.map((s) => s.category);
	const values = stats.map((s) => s.count);
	const palette = labels.map(
		(_, idx) => sourceColors[idx % sourceColors.length] || "#38bdf8"
	);
	if (!wikiCategoryChart) {
		wikiCategoryChart = new Chart(ctx, {
			type: "doughnut",
			data: {
				labels,
				datasets: [
					{
						data: values,
						backgroundColor: palette,
					},
				],
			},
			options: {
				maintainAspectRatio: false,
				plugins: {
					legend: {
						position: "bottom",
						labels: { color: "rgba(255,255,255,0.85)", usePointStyle: true },
					},
				},
				cutout: "58%",
			},
		});
	} else {
		wikiCategoryChart.data.labels = labels;
		wikiCategoryChart.data.datasets[0].data = values;
		wikiCategoryChart.data.datasets[0].backgroundColor = palette;
		wikiCategoryChart.update();
	}
}

function renderWikiCategoryList(stats) {
	const container = el("wikiCategoryList");
	if (!container) return;
	container.innerHTML = "";
	const topStats = stats.slice(0, 4);
	if (!topStats.length) {
		container.innerHTML = '<p class="small">Sem categorias suficientes.</p>';
		return;
	}
	topStats.forEach((item) => {
		const block = document.createElement("div");
		block.className = "wiki-cat-block";
		const title = document.createElement("div");
		title.className = "wiki-cat-title";
		title.innerHTML = `<span>${item.category}</span><span>${item.count} artigo(s)</span>`;
		const list = document.createElement("div");
		list.className = "wiki-cat-titles";
		item.titles.slice(0, 5).forEach((name) => {
			const tag = document.createElement("span");
			tag.className = "tag";
			tag.textContent = name;
			list.appendChild(tag);
		});
		block.appendChild(title);
		block.appendChild(list);
		container.appendChild(block);
	});
}

function renderWikiSources(section) {
	const block = el("wikiSourcesBlock");
	const list = el("wikiSourcesList");
	const metaInfo = el("wikiSourcesMeta");
	const catList = el("wikiCategoryList");
	if (!block || !list) return;
	const items = Array.isArray(section?.artigos_wikipedia_similares)
		? section.artigos_wikipedia_similares
		: [];
	list.innerHTML = "";
	hasWikiList = items.length > 0;

	if (!hasWikiList) {
		block.style.display = "none";
		if (metaInfo) metaInfo.textContent = "";
		if (catList) catList.innerHTML = "";
		resetWikiCategoryChart();
		updateSourcesCardVisibility();
		return;
	}
	block.style.display = "block";
	if (metaInfo) {
		const total = section?.total_encontrado ?? items.length;
		const shown = section?.limite_exibido ?? items.length;
		const minScore = section?.limiar_similaridade ?? 0.7;
		const configured = section?.limite_configurado ?? shown;
		const relaxado = section?.relaxado_por_falta ? " (limiar relaxado)" : "";
		metaInfo.textContent = `Exibindo ${shown} de ${total} (limite: ${configured} | ≥ ${Math.round(
			minScore * 100
		)}% de similaridade)${relaxado}`;
	}
	const stats = buildWikiCategoryStats(items);
	updateWikiCategoryChart(stats);
	renderWikiCategoryList(stats);
	items.forEach((item) => {
		const row = document.createElement("div");
		row.className = "source-item wiki";
		const left = document.createElement("div");
		const title = document.createElement("div");
		title.textContent = item.titulo || `Artigo ${item.id}`;
		const meta = document.createElement("div");
		meta.className = "meta";
		const cat =
			item.categoria ||
			(Array.isArray(item.categorias) ? item.categorias[0] : null) ||
			"Sem categoria";
		meta.textContent = cat;
		left.appendChild(title);
		left.appendChild(meta);
		const right = document.createElement("div");
		right.className = "percent";
		const simVal =
			typeof item.similaridade === "number"
				? Math.round(item.similaridade * 1000) / 10
				: null;
		right.textContent = simVal != null ? `${simVal.toFixed(1)}%` : "--";
		row.appendChild(left);
		row.appendChild(right);
		row.title = `Similaridade: ${
			item.similaridade != null ? item.similaridade : "--"
		} | Categoria: ${cat}`;
		if (item.url) {
			row.style.cursor = "pointer";
			row.onclick = () => window.open(item.url, "_blank");
		}
		list.appendChild(row);
	});
	updateSourcesCardVisibility();
}

function updateSourcesCardVisibility() {
	const card = el("sourcesCard");
	if (!card) return;
	card.style.display =
		hasSourceList || hasWikiList || hasEntityList ? "block" : "none";
}

document.addEventListener("DOMContentLoaded", () => {
	applyDefaultValues();
	const advBtn = el("toggleAdvanced");
	const advBox = el("advancedControls");
	if (advBtn && advBox) {
		advBtn.addEventListener("click", () => {
			const isOpen = advBox.classList.toggle("open");
			advBox.setAttribute("aria-hidden", isOpen ? "false" : "true");
			advBtn.textContent = isOpen
				? "Ocultar parâmetros avançados"
				: "Mostrar parâmetros avançados";
			advBtn.setAttribute("aria-expanded", isOpen ? "true" : "false");
		});
	}

	el("toggleRaw").onclick = () => {
		if (el("detailsBox").style.display === "none") {
			toggleDetails(true);
		} else {
			toggleDetails(false);
		}
	};
	el("closeRaw").onclick = () => toggleDetails(false);
	const copyBtn = el("copyRaw");
	if (copyBtn) {
		copyBtn.disabled = true;
		copyBtn.addEventListener("click", () => {
			copyRawJson();
		});
	}

	el("run").onclick = async () => {
		const text = el("text").value ? el("text").value.trim() : "";
		if (!text) {
			alert("Insira um texto.");
			return;
		}
		el("run").disabled = true;
		el("run").textContent = "Analisando...";
		toggleDetails(false);
		el("toggleRaw").disabled = true;
		if (copyBtn) {
			copyBtn.disabled = true;
			copyBtn.textContent = "Copiar JSON";
		}
		const body = {
			text,
			k: Number(el("k")?.value || DEFAULT_PARAMS.k),
			w_hist: Number(el("w_hist")?.value || DEFAULT_PARAMS.w_hist),
			w_bert: Number(el("w_bert")?.value || DEFAULT_PARAMS.w_bert),
			w_fontes: Number(el("w_fontes")?.value || DEFAULT_PARAMS.w_fontes),
			max_tokens: Number(el("max_tokens")?.value || DEFAULT_PARAMS.max_tokens),
			overlap_tokens: Number(
				el("overlap_tokens")?.value || DEFAULT_PARAMS.overlap_tokens
			),
			hist_agg: el("hist_agg")?.value || DEFAULT_PARAMS.hist_agg,
			bert_agg: el("bert_agg")?.value || DEFAULT_PARAMS.bert_agg,
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
				const hRaw = json.historico?.consistencia_raw;
				const b = json.bert?.prob_true;
				const f = json.confirmacao_fontes?.fonte_score;
				const fin = json.final?.score;
				paint("barHist", "pctHist", "stHist", h);
				paint("barBert", "pctBert", "stBert", b);
				paint("barFontes", "pctFontes", "stFontes", f);
				paint("barFinal", "pctFinal", "stFinal", fin);
				// Atualiza gráfico de pizza (apenas 3 fatias)
				updatePieChart(h, b, f);
				updateBarChart(h, b, f, fin);
				renderCorroborationNote(
					hRaw,
					h,
					json.historico?.corroboracao_multiplicador,
					json.historico?.corroboracao_score,
					json.historico?.corroboracao
				);
				const displayPayload = json.frontend_view || json;
				el("out").textContent = JSON.stringify(displayPayload, null, 2);
				const fontes = json.confirmacao_fontes?.fontes_individuais || [];
				const wikiSection = json.fontes_externas || {
					artigos_wikipedia_similares: json.wikipedia?.matches || [],
					total_encontrado: json.wikipedia?.matches?.length || 0,
					limite_exibido: json.wikipedia?.matches?.length || 0,
					limiar_similaridade: 0.7,
				};
				renderSources(fontes);
				renderEntityVerification(
					json.confirmacao_fontes?.entidades_verificadas || null
				);
				renderWikiSources(wikiSection);
				el("toggleRaw").disabled = false;
				if (copyBtn) {
					copyBtn.disabled = false;
					copyBtn.textContent = "Copiar JSON";
				}
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
