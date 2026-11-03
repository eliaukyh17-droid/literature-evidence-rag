# 运行与配置

## 交互演示

```bash
uv sync --locked --extra vector
uv run --locked python scripts/run_demo.py
```

浏览器打开http://127.0.0.1:8000/。顶部标注“交互演示 · 示例回答”。这个模式从自造文档检索，按固定规则组装回答并分段发送，便于体验来源、错误和停止流程。

## 关键词检索

```bash
uv run --locked literature-kb ingest guide.md catalog.md
uv run --locked literature-kb search "证据定位"
uv run --locked uvicorn literature_kb.api:create_app --factory --host 127.0.0.1 --port 8000 --no-access-log --log-config logging.json
```

默认文档根目录为examples，库在.local/demo。API文档在http://127.0.0.1:8000/docs。文档导入仅接受已配置根目录内的相对Markdown路径。

## 真实向量检索

下载固定的公开E5快照。

```bash
uv run --locked python scripts/download_embedding.py --revision 614241f622f53c4eeff9890bdc4f31cfecc418b3 --variant int8
```

下载器先解析固定revision，核对文件大小、Git blob或LFS SHA，写入本地清单。运行时再次核对图与tokenizer。INT8 ONNX图约118MB，向量维度384，运行使用CPU两线程。

Unix shell配置如下。KB_SOURCE_ROOT可替换为自己的Markdown文献目录。

```bash
export KB_SOURCE_ROOT="$PWD/examples"
export KB_DATA_DIR="$PWD/.local/library"
export KB_EMBEDDING_MODEL_DIR="$PWD/.local/models/multilingual-e5-small-614241f622f5-int8"
export KB_CHUNK_MAX_TOKENS=512
uv run --locked literature-kb ingest guide.md catalog.md
uv run --locked literature-kb index-vectors --batch-size 8
uv run --locked literature-kb search "文档更新" --mode hybrid_terms
```

PowerShell使用对应环境变量语法。

```powershell
$env:KB_SOURCE_ROOT = Join-Path $PWD 'examples'
$env:KB_DATA_DIR = Join-Path $PWD '.local/library'
$env:KB_EMBEDDING_MODEL_DIR = Join-Path $PWD '.local/models/multilingual-e5-small-614241f622f5-int8'
$env:KB_CHUNK_MAX_TOKENS = '512'
uv run --locked literature-kb ingest guide.md catalog.md
uv run --locked literature-kb index-vectors --batch-size 8
```

## 接入本地生成模型

准备支持JSON Schema和聊天模板的Qwen3 GGUF，以及[llama.cpp运行程序](https://github.com/ggml-org/llama.cpp)。本项目真实模型验证使用b10837。将下面的模型路径替换成自己的文件。

```bash
llama-server -m /path/to/model.gguf --alias literature-qwen3-local --host 127.0.0.1 --port 18080 -c 3072 -np 1 --reasoning off --offline
```

在已完成向量配置和建库的另一个终端设置模型服务地址，然后启动API。

```bash
export KB_LOCAL_LLM_URL=http://127.0.0.1:18080
export KB_LOCAL_LLM_MODEL=literature-qwen3-local
uv run --locked uvicorn literature_kb.api:create_app --factory --host 127.0.0.1 --port 8000 --no-access-log --log-config logging.json
```

```powershell
$env:KB_LOCAL_LLM_URL = 'http://127.0.0.1:18080'
$env:KB_LOCAL_LLM_MODEL = 'literature-qwen3-local'
uv run --locked uvicorn literature_kb.api:create_app --factory --host 127.0.0.1 --port 8000 --no-access-log --log-config logging.json
```

浏览器打开首页。CPU可先选1段候选、标准回答。增加候选数和回答长度会增加推理时间。每问独立，页面刷新后清空本页记录。

## API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/v1/health` | 库状态、模型配置与处理锁 |
| GET | `/v1/documents` | 活动文档 |
| POST | `/v1/documents/import` | 相对MD路径导入 |
| DELETE | `/v1/documents/{id}` | 逻辑删除 |
| POST | `/v1/search` | 检索候选 |
| GET | `/v1/evidence/{chunk_id}` | 活动版本的原文与位置 |
| POST | `/v1/answer` | 完整结构化问答 |
| POST | `/v1/answer/stream` | SSE阶段、增量与最终校验 |

```bash
curl -N http://127.0.0.1:8000/v1/answer/stream \
  -H 'Content-Type: application/json' \
  -d '{"query":"文献怎样保留来源？","mode":"bm25","top_k":1,"max_new_tokens":256}'
```

SSE事件依次包含started、stage、sources、answer_delta和done，错误以error终止。长等待可能收到ping。每条带request_id、sequence和elapsed_ms。answer_delta的provisional为true，done提供最终答案。业务错误可能发生在HTTP200头发送之后，应检查终止事件。重新提问由客户端发起，Last-Event-ID恢复请求返回400。

## 常见状态

- 409表示准备或生成期间文档改变，需要重新检索。
- 410表示原引用对应文档已删除或替换。
- 503表示索引过期、模型未配置或服务繁忙。
- 504表示本次问答或模型请求超时。
- 502表示模型输出格式、引用或完成状态不合格。

Qdrant local同一目录仅由一个进程占用，API运行时先停止API再建库。每个库与文档根目录绑定，换来源目录应创建新库。.env.example提供配置参考，应用读取shell环境变量。
