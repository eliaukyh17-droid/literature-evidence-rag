# 参与开发

```bash
uv sync --locked --extra vector
uv run --locked pytest -q
uv run --locked ruff check src tests scripts
uv run --locked ruff format --check src tests scripts
node --test tests/web_client.test.mjs
```

测试使用自造文档和模拟模型。修改检索或分块时应保留原文位置、文档版本、旧调用行为和语料指纹。修改流式协议时应验证完整生成、非法输出、断线和取消，确保最后只有一个终止事件。

反馈问题时提供Python/系统版本、复现步骤和已脱敏的错误码。原始资料、模型文件与.local运行数据由使用者本地管理。
