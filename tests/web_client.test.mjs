import assert from "node:assert/strict";
import test from "node:test";
import { evidenceParts, readEvents } from "../src/literature_kb/web/stream-client.js";

function frame(event, sequence, data = {}) {
  return `id: test:${sequence}\r\nevent: ${event}\r\ndata: ${JSON.stringify({ request_id: "test", sequence, ...data })}\r\n\r\n`;
}

function response(text) {
  const bytes = new TextEncoder().encode(text);
  return new Response(new ReadableStream({ start(controller) {
    for (let i = 0; i < bytes.length; i += 2) controller.enqueue(bytes.slice(i, i + 2));
    controller.close();
  } }), { headers: { "content-type": "text/event-stream" } });
}

test("SSE browser decoder handles split UTF8 and CRLF", async () => {
  const events = [];
  await readEvents(response(frame("started", 1) + frame("answer_delta", 2, { text: "中俄😀" }) + frame("done", 3)), (event, data) => events.push([event, data]));
  assert.equal(events[1][1].text, "中俄😀");
  assert.equal(events.at(-1)[0], "done");
});

test("EOF and duplicate sequence cannot become a completed answer", async () => {
  await assert.rejects(readEvents(response(frame("started", 1)), () => {}), /提前结束/);
  await assert.rejects(readEvents(response(frame("started", 1) + frame("done", 1)), () => {}), /顺序异常/);
});

test("evidence highlight uses Unicode code points and preserves source text", () => {
  assert.deepEqual(evidenceParts({ page_text: "前😀原文后", page_char_start: 2, page_char_end: 4, text: "原文" }), ["前😀", "原文", "后"]);
  assert.throws(() => evidenceParts({ page_text: "前😀原文后", page_char_start: 3, page_char_end: 5, text: "原文" }), /定位异常/);
});
