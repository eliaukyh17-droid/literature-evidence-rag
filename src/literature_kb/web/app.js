import { evidenceParts, readEvents } from "./stream-client.js";

const $ = (id) => document.getElementById(id);
let activeRequest = null;
let evidenceRequest = null;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function setBusy(busy) {
  $("send").disabled = busy;
  $("stop").hidden = !busy;
  document.querySelectorAll("select, .example").forEach((node) => { node.disabled = busy; });
  $("question-form").setAttribute("aria-busy", String(busy));
}

function sourceButton(source, compact = false) {
  const button = element("button", compact ? "source-chip" : "source-card");
  button.type = "button";
  if (compact) {
    button.textContent = `[${source.source_id}] ${source.title}`;
  } else {
    button.append(element("span", "source-id", source.source_id), element("strong", "", source.title));
    const footer = element("small");
    footer.append(element("span", "", `页标记 ${source.page_start}`), element("span", "", "查看原文 ↗"));
    button.append(footer);
  }
  button.addEventListener("click", () => showEvidence(source));
  return button;
}

function renderSources(sources, provisional = false) {
  $("sources").replaceChildren(...sources.map((source) => sourceButton(source)));
  $("source-count").textContent = String(sources.length);
  $("source-hint").textContent = provisional
    ? "正在依据这些片段生成，完成后核对实际引用。"
    : sources.length ? "回答使用了以下来源，可打开原文核对。" : "本次尚无可列出的回答来源。";
}

async function showEvidence(source) {
  evidenceRequest?.abort();
  const controller = new AbortController();
  evidenceRequest = controller;
  $("evidence-title").textContent = source.title;
  $("evidence-meta").textContent = `[${source.source_id}] · 页标记 ${source.page_start}`;
  $("evidence-body").textContent = "正在读取原文…";
  if (!$("evidence-dialog").open) $("evidence-dialog").showModal();
  try {
    const response = await fetch(`/v1/evidence/${encodeURIComponent(source.chunk_id)}`, { signal: controller.signal });
    if (!response.ok) throw new Error(response.status === 410 ? "文献已被替换或删除，请重新检索。" : "原文暂时无法读取。");
    const evidence = await response.json();
    if (evidence.version_id !== source.version_id) throw new Error("文献版本已变化，请重新检索。");
    const [before, selected, after] = evidenceParts(evidence);
    if (controller !== evidenceRequest) return;
    const mark = element("mark", "", selected);
    $("evidence-body").replaceChildren(document.createTextNode(before), mark, document.createTextNode(after));
    mark.scrollIntoView({ block: "nearest" });
  } catch (error) {
    if (error.name !== "AbortError" && controller === evidenceRequest) $("evidence-body").textContent = error.message;
  }
}

function makeTurn(question) {
  $("welcome").hidden = true;
  const turn = element("article", "turn");
  const heading = element("div", "answer-heading");
  const status = element("span", "answer-status", "正在检索");
  heading.append(element("span", "answer-symbol", "文"), element("strong", "", "文献助手"), status);
  const answer = element("div", "answer-text");
  const metrics = element("div", "answer-metrics");
  const citations = element("div", "turn-sources");
  turn.append(element("p", "question-bubble", question), heading, answer, citations, metrics);
  $("messages").append(turn);
  turn.scrollIntoView({ block: "nearest" });
  return { turn, status, answer, metrics, citations };
}

function stageLabel(payload) {
  const labels = { retrieving: "正在检索文献", preparing: "正在整理证据", generating: "正在读取资料与生成" };
  if (payload.stage === "prefill") {
    const total = Number(payload.total);
    const processed = Number(payload.processed);
    return total > 0 && Number.isFinite(processed) ? `正在读取资料 ${Math.min(100, Math.round(processed / total * 100))}%` : "正在读取资料";
  }
  return labels[payload.stage] || "处理中";
}

$("question-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = $("question").value.trim();
  if (!query || activeRequest) return;
  const controller = new AbortController();
  activeRequest = controller;
  const view = makeTurn(query);
  const request = { query, mode: $("mode").value, top_k: Number($("top-k").value), max_new_tokens: Number($("max-tokens").value) };
  if ($("scope").value) request.document_ids = [$("scope").value];
  $("question").value = "";
  renderSources([], true);
  setBusy(true);
  let terminal = false;
  try {
    const response = await fetch("/v1/answer/stream", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request), signal: controller.signal,
    });
    await readEvents(response, (type, payload) => {
      if (type === "started") view.turn.dataset.requestId = payload.request_id;
      if (type === "stage") view.status.textContent = stageLabel(payload);
      if (type === "sources") renderSources(payload.sources, true);
      if (type === "answer_delta") {
        view.answer.classList.add("generating");
        view.answer.append(document.createTextNode(payload.text));
        view.status.textContent = "生成中 · 待校验";
      }
      if (type === "error") throw new Error(errorLabel(payload));
      if (type === "done") {
        terminal = true;
        view.answer.classList.remove("generating");
        view.answer.textContent = payload.answer;
        view.status.textContent = payload.status === "insufficient_evidence" ? "资料不足" : "引用已核对";
        view.citations.replaceChildren(...payload.citations.map((source) => sourceButton(source, true)));
        renderSources(payload.citations);
        const meta = payload.metadata;
        const details = [];
        if (Number.isFinite(meta.retrieval_ms)) details.push(`检索 ${(meta.retrieval_ms / 1000).toFixed(1)}s`);
        if (Number.isFinite(meta.first_answer_delta_ms)) details.push(`首字 ${(meta.first_answer_delta_ms / 1000).toFixed(1)}s`);
        details.push(`总耗时 ${(payload.elapsed_ms / 1000).toFixed(1)}s`);
        view.metrics.textContent = details.join(" · ");
      }
    });
    if (!terminal) throw new Error("本次回答尚未完成。");
  } catch (error) {
    view.answer.classList.remove("generating");
    view.answer.textContent = "";
    view.answer.classList.add("answer-error");
    view.citations.replaceChildren();
    renderSources([]);
    const stopped = error.name === "AbortError";
    view.status.textContent = stopped ? "已停止" : "本次未完成";
    view.answer.textContent = stopped ? "本次生成已停止，可以继续提问。" : error.message;
  } finally {
    activeRequest = null;
    setBusy(false);
    $("question").focus();
  }
});

function errorLabel(payload) {
  const labels = {
    answer_busy: "已有问答正在处理，请稍后再试。", generation_busy: "模型正在处理其他请求，请稍后再试。",
    generation_not_configured: "请先启动并配置本地模型服务。", generation_unavailable: "本地模型连接中断，请检查模型服务。",
    generation_timeout: "本地模型响应超时，可减少候选片段后重新提问。", answer_timeout: "本次处理超时，可减少候选片段后重新提问。",
    answer_context_changed: "文献在生成期间发生变化，请重新提问。", generation_incomplete: "回答未完整生成，可增加回答长度后重新提问。",
    invalid_citations: "本次来源引用未通过校验，请重新提问。", invalid_model_answer: "模型返回格式未通过校验，请重新提问。",
  };
  return labels[payload.code] || `本次问答未能完成（${payload.code || "服务异常"}）。`;
}

$("stop").addEventListener("click", () => activeRequest?.abort());
$("question").addEventListener("keydown", (event) => {
  if (event.ctrlKey && event.key === "Enter" && !event.isComposing) $("question-form").requestSubmit();
});
document.querySelectorAll(".example").forEach((button) => button.addEventListener("click", () => {
  $("question").value = button.dataset.question;
  $("mode").value = button.dataset.mode;
  $("question").focus();
}));
$("close-evidence").addEventListener("click", () => $("evidence-dialog").close());
$("evidence-dialog").addEventListener("close", () => evidenceRequest?.abort());

async function loadLibrary() {
  try {
    const [healthResponse, documentsResponse] = await Promise.all([fetch("/v1/health"), fetch("/v1/documents")]);
    if (!healthResponse.ok || !documentsResponse.ok) throw new Error("连接失败");
    const health = await healthResponse.json();
    const { documents } = await documentsResponse.json();
    $("library-summary").textContent = `${health.documents} 份文献 · ${health.chunks.toLocaleString()} 个片段`;
    $("model-status").textContent = health.demo_mode ? "交互演示 · 示例回答" : health.answer_generation_enabled ? "本地模型已配置" : "模型服务待配置";
    $("model-status").classList.toggle("ready", health.answer_generation_enabled);
    for (const doc of documents) {
      const option = element("option", "", doc.title);
      option.value = doc.document_id;
      $("scope").append(option);
    }
    if (!health.dense_index.ready) $("mode").value = "bm25";
  } catch {
    $("model-status").textContent = "服务连接失败";
    $("library-summary").textContent = "暂时无法读取";
  }
}

loadLibrary();
