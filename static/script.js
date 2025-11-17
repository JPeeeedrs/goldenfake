const el = (id) => document.getElementById(id);
const fmt = (v) =>
	v === null || v === undefined || isNaN(v)
		? "--"
		: Math.round(v * 10) / 10 + "%";

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
		meta.textContent =
			(item.publisher || item.provider || "fonte") +
			" • " +
			(item.overlap_bucket || "");
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
	updateSourceCharts(list);
	updateSourcesCardVisibility();
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
		const avg =
			typeof block?.media_score === "number"
				? block.media_score.toFixed(2)
				: "--";
		const fortes = block?.fortes ?? 0;
		const fracas = block?.fracas ?? 0;
		const ausentes = block?.ausentes ?? 0;
		summary.textContent = `Média: ${avg} • fortes: ${fortes} | fracas: ${fracas} | ausentes: ${ausentes}`;
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
		const scoreVal =
			typeof item.score === "number" ? item.score.toFixed(2) : "--";
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
			row.title = tooltip;
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
		const categories = Array.isArray(item.categorias)
			? item.categorias
			: item.categoria
			? [item.categoria]
			: [];
		const primary = categories.length > 0 ? categories[0] : "Sem categoria";
		const current = stats.get(primary) || { count: 0, titles: [] };
		current.count += 1;
		current.titles.push(item.titulo || `Artigo ${item.id}`);
		stats.set(primary, current);
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
	const maxColors = sourceColors.length;
	const topStats = stats.slice(0, maxColors);
	const labels = topStats.map((s) => s.category);
	const values = topStats.map((s) => s.count);
	if (!wikiCategoryChart) {
		wikiCategoryChart = new Chart(ctx, {
			type: "doughnut",
			data: {
				labels,
				datasets: [
					{
						data: values,
						backgroundColor: sourceColors,
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
		metaInfo.textContent = `Exibindo ${shown} de ${total} (limite: ${configured} | ≥ ${Math.round(
			minScore * 100
		)}% de similaridade)`;
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
				el("out").textContent = JSON.stringify(json, null, 2);
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
