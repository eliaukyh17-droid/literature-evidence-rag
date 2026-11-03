export async function readEvents(response, onEvent) {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error?.message || `请求失败（${response.status}）`);
  }
  if (!response.headers.get("content-type")?.includes("text/event-stream") || !response.body) {
    throw new Error("服务没有返回问答流。");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let buffer = "";
  let requestId = null;
  let sequence = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      buffer = buffer.replace(/\r\n/g, "\n");
      if (buffer.length > 256 * 1024) throw new Error("问答流超出接收上限。");
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        let event = "message";
        let id = "";
        const data = [];
        for (const line of frame.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          if (line.startsWith("id:")) id = line.slice(3).trim();
          if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
        }
        if (!data.length) continue;
        const payload = JSON.parse(data.join("\n"));
        if (!requestId && event === "started") requestId = payload.request_id;
        if (!requestId || payload.request_id !== requestId || payload.sequence !== sequence + 1 || id !== `${requestId}:${payload.sequence}`) {
          throw new Error("问答流标识或顺序异常，请重新提问。");
        }
        sequence = payload.sequence;
        await onEvent(event, payload);
        if (event === "done" || event === "error") return payload;
      }
      if (done) throw new Error("连接提前结束，本次回答尚未完成。");
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

export function evidenceParts(evidence) {
  const characters = Array.from(evidence.page_text);
  const start = evidence.page_char_start;
  const end = evidence.page_char_end;
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < start || end > characters.length || characters.slice(start, end).join("") !== evidence.text) {
    throw new Error("原文片段定位异常，请重新检索。");
  }
  return [characters.slice(0, start).join(""), evidence.text, characters.slice(end).join("")];
}
