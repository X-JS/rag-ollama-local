# rag-ollama-local

从零搭建本地私有化 RAG 知识库｜Ollama + LangChain + Chroma 完整落地教程。

数据完全本地存储，不上传云端；可离线检索自有文档，实现私有文档问答并自动标注资料来源。

`***注：纯学习，不具备上运营环境的条件***`

> 双模式：问题命中知识库 → 依据资料回答并标注来源；资料缺失或无关 → 正常对话（不会编造来源）。

## 使用方法

> 一句话流程：`放文件到 data/ → uv run rag-ollama-build → uv run rag-ollama-qa`

**前置条件**

- Ollama 已启动并拉好本地模型（默认 `qwen2:7b`，可用 `ollama list` 查看；换模型改两个文件里的 `MODEL_NAME`/`EMBED_MODEL_NAME`）。
- Tesseract-OCR 已手工安装并加入系统 PATH（用于图片/铭牌 OCR）。

**第一步：放资料**

把 PDF / Word / Excel / 图片放进 `data/` 文件夹，文件名建议 `[标准/厂家/手册]文档名称.后缀`，方便溯源。

**第二步：入库**（首次运行或更新资料时执行，构建本地向量库）

```bash
uv run rag-ollama-build
```

**第三步：启动网页问答**

```bash
uv run rag-ollama-qa
```

浏览器自动打开 `http://127.0.0.1:7860`，输入问题即可查询，回答末尾自动标注参考来源。提问资料库不存在的内容时，会提示「当前知识库内无匹配内容」。

**重新全量入库**：更新资料后想重建，先删除 `db/` 文件夹，再跑第二步。

## 环境要求

- Python 3.11（由 uv 管理，无需手动 pip）
- Ollama 已启动并拉好本地模型（默认 `qwen2:7b`）
- Tesseract-OCR 已手工安装并加入系统 PATH

## 项目结构

```
rag-ollama-local/
├─ data/                     # 存放所有私有资料：PDF/Word/Excel/JPG/PNG
├─ db/                       # 向量数据库文件，脚本运行后自动生成
├─ src/rag_ollama_local/
│  ├─ load_data_to_db.py     # 批量解析文档、入库向量库脚本
│  └─ knowledge_qa.py        # Gradio 网页问答主程序
├─ pyproject.toml            # uv 依赖管理 + 控制台入口
└─ README.md
```

## 安装依赖（使用 uv）

```bash
uv add langchain langchain-community langchain-text-splitters chromadb gradio pytesseract pymupdf python-docx openpyxl unstructured
```

> 默认锁定在 langchain 0.3.x，与文中 API（`RetrievalQA`、`llms.Ollama`、`OllamaEmbeddings`）兼容。

## 可自定义配置

两个脚本共用的配置集中在 `src/rag_ollama_local/config.py`，改这里即可，无需改脚本：

| 配置 | 说明 |
| --- | --- |
| `OLLAMA_BASE_URL` | Ollama 服务地址，默认 `http://192.168.211.163:11434` |
| `MODEL_NAME` | 生成（对话）模型，默认 `qwen2:7b` |
| `EMBED_MODEL` | 向量化（embedding）模型，默认 `nomic-embed-text` |
| `DATA_DIR` | 私有资料目录，默认 `<仓库根>/data` |
| `DB_DIR` | 向量库目录，默认 `<仓库根>/db` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 文本切片长度（默认 1600 / 350） |

> 向量化模型需要先在 Ollama 拉取：`ollama pull nomic-embed-text`。生成模型和向量化模型可分开指定。另外 `search_kwargs={"k": 3}`（每次检索片段数量）在 `knowledge_qa.py` 的 `build_qa_chain()` 内。

## 常见问题

- 模型幻觉：配置了 `temperature=0` 与强约束 Prompt，若仍出现，检查 `k` 值是否过大。
- OCR 空白：确认 Tesseract-OCR 已加入系统 PATH。
- Ollama 拒绝连接：确认 Ollama 服务已启动（`ollama list`），防火墙放行 11434 端口。
- 向量库重复膨胀：更新资料前删除 `db/` 文件夹重新全量入库。
