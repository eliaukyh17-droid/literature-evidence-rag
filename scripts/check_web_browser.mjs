// Browser acceptance with local Chromium CDP and public fixtures. No npm packages.
import assert from "node:assert/strict";
import { spawn, execFileSync } from "node:child_process";
import { openSync, closeSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { randomUUID } from "node:crypto";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const folder = path.join(root, ".local", `web-browser-${randomUUID().slice(0, 12)}`);
mkdirSync(folder, { recursive: true });
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function port() {
  const server = net.createServer();
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const result = server.address().port;
  await new Promise((resolve) => server.close(resolve));
  return result;
}
async function until(check, timeout = 15000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) {
    try { const result = await check(); if (result) return result; } catch {}
    await sleep(100);
  }
  throw new Error("Browser check timed out");
}
const children = [];
const descriptors = [];
function launch(command, args, name, env = process.env) {
  const log = openSync(path.join(folder, `${name}.log`), "w");
  descriptors.push(log);
  const child = spawn(command, args, { cwd: root, env, windowsHide: true, stdio: ["ignore", log, log] });
  children.push(child);
  return child;
}
let websocket;
const browserErrors = [];
const result = { fixture: true, real_model_used: false, checks: [], passed: false };
const apiPort = await port();
const debugPort = await port();
try {
  launch(path.join(root, ".venv/Scripts/python.exe"), ["-m", "uvicorn", "scripts.web_fixture:create_app", "--factory", "--host", "127.0.0.1", "--port", String(apiPort), "--no-access-log", "--log-config", "logging.json"], "api", { ...process.env, KB_WEB_TEST_DIR: folder });
  await until(async () => (await fetch(`http://127.0.0.1:${apiPort}/v1/health`)).ok);
  launch("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe", ["--headless=new", "--disable-gpu", "--disable-background-networking", "--disable-component-update", "--no-first-run", "--no-default-browser-check", `--user-data-dir=${path.join(folder, "profile")}`, `--remote-debugging-port=${debugPort}`, "--remote-debugging-address=127.0.0.1", "--window-size=1440,1000", "about:blank"], "browser");
  const target = await until(async () => (await (await fetch(`http://127.0.0.1:${debugPort}/json/list`)).json()).find((item) => item.type === "page"));
  websocket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { websocket.onopen = resolve; websocket.onerror = reject; });
  let id = 0;
  const pending = new Map();
  websocket.onmessage = ({ data }) => {
    const packet = JSON.parse(data);
    if (packet.method === "Runtime.exceptionThrown") browserErrors.push(packet.params.exceptionDetails.text);
    if (packet.id && pending.has(packet.id)) {
      const { resolve, reject } = pending.get(packet.id);
      pending.delete(packet.id);
      if (packet.error) reject(new Error(JSON.stringify(packet.error))); else resolve(packet.result);
    }
  };
  function command(method, params = {}) {
    return new Promise((resolve, reject) => { const next = ++id; pending.set(next, { resolve, reject }); websocket.send(JSON.stringify({ id: next, method, params })); });
  }
  async function evaluate(expression) {
    const value = await command("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
    if (value.exceptionDetails) throw new Error(value.exceptionDetails.text);
    return value.result.value;
  }
  const waitFor = (expression) => until(() => evaluate(expression));
  async function screenshot(name) {
    const image = await command("Page.captureScreenshot", { format: "png" });
    writeFileSync(path.join(folder, name), Buffer.from(image.data, "base64"));
  }
  async function ask(question) {
    await evaluate(`document.getElementById('question').value=${JSON.stringify(question)}; document.getElementById('question-form').requestSubmit(); true`);
  }
  await command("Runtime.enable");
  await command("Page.enable");
  await command("Page.navigate", { url: `http://127.0.0.1:${apiPort}/` });
  await waitFor("document.getElementById('library-summary')?.textContent.includes('2 份文献')");
  await screenshot("desktop-welcome.png");
  await ask("证据定位 安全测试");
  await waitFor("document.querySelector('.turn:last-child .answer-status')?.textContent.includes('待校验') && document.querySelector('.turn:last-child .answer-text')?.textContent.length > 0");
  assert(await evaluate("document.getElementById('send').disabled && !document.getElementById('stop').hidden"));
  result.checks.push("incremental_text_before_done");
  await waitFor("document.querySelector('.turn:last-child .answer-status')?.textContent === '引用已核对'");
  assert(await evaluate("document.querySelector('.answer-text').textContent.includes('<img') && !document.querySelector('.answer-text img') && window.injected !== true"));
  result.checks.push("untrusted_model_text_is_plain_text");
  await evaluate("document.querySelector('.source-chip').click(); true");
  await waitFor("document.getElementById('evidence-dialog').open && document.querySelector('#evidence-body mark')?.textContent.length > 0");
  await screenshot("evidence-dialog.png");
  await evaluate("document.getElementById('close-evidence').click(); true");
  result.checks.push("citation_opens_highlighted_source");
  await ask("证据定位 错误测试");
  await waitFor("document.querySelector('.turn:last-child .answer-status')?.textContent === '本次未完成'");
  assert(await evaluate("!document.querySelector('.turn:last-child .source-chip') && !document.querySelector('.turn:last-child .answer-text').textContent.includes('浏览器夹具')"));
  result.checks.push("invalid_citation_preview_discarded");
  await ask("证据定位 停止测试");
  await waitFor("document.querySelector('.turn:last-child .answer-status')?.textContent.includes('待校验')");
  await evaluate("document.getElementById('stop').click(); true");
  await waitFor("document.querySelector('.turn:last-child .answer-status')?.textContent === '已停止'");
  await until(() => { const state = JSON.parse(readFileSync(path.join(folder, "fixture_state.json"))); return state.cancelled >= 1 && state.active === 0; });
  result.checks.push("browser_abort_cancels_backend");
  await ask("证据定位");
  await waitFor("document.querySelector('.turn:last-child .answer-status')?.textContent === '引用已核对'");
  const ids = await evaluate("[...document.querySelectorAll('.turn')].map(node => node.dataset.requestId)");
  assert.equal(new Set(ids).size, 4);
  result.checks.push("next_question_after_abort_succeeds");
  await evaluate("[...document.querySelectorAll('.turn')].slice(0,-1).forEach(node => node.remove()); window.scrollTo(0,0); true");
  await screenshot("desktop-answer.png");
  await evaluate("document.querySelector('.source-chip').click(); true");
  await waitFor("document.getElementById('evidence-dialog').open && document.querySelector('#evidence-body mark')?.textContent.length > 0");
  await screenshot("evidence-dialog.png");
  await evaluate("document.getElementById('close-evidence').click(); true");
  await command("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await command("Page.navigate", { url: `http://127.0.0.1:${apiPort}/` });
  await waitFor("document.getElementById('library-summary')?.textContent.includes('2 份文献')");
  assert(await evaluate("document.documentElement.scrollWidth <= window.innerWidth"));
  await screenshot("mobile-welcome.png");
  result.checks.push("mobile_layout_without_horizontal_overflow");
  assert.deepEqual(browserErrors, []);
  result.passed = true;
} finally {
  websocket?.close();
  for (const child of children.reverse()) {
    if (child.exitCode === null) {
      try { execFileSync("taskkill", ["/PID", String(child.pid), "/T", "/F"], { stdio: "ignore", windowsHide: true }); } catch {}
    }
  }
  for (const descriptor of descriptors) closeSync(descriptor);
  result.browser_errors = browserErrors;
  result.api_port = apiPort;
  result.debug_port = debugPort;
  result.ports_closed = await until(async () => {
    const states = await Promise.all([apiPort, debugPort].map(async (value) => { try { await fetch(`http://127.0.0.1:${value}/`); return false; } catch { return true; } }));
    return states.every(Boolean);
  });
  writeFileSync(path.join(folder, "result.json"), JSON.stringify(result, null, 2));
  console.log(JSON.stringify({ ...result, output: folder }));
}
