import hashlib
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain_core.documents import Document

from rag_ollama_local.config import (
    DATA_DIR,
    DB_DIR,
    OLLAMA_BASE_URL,
    EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# 国标专用文本分割器（适配规范条款、表格、分段内容）
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n## ", "\n\n### ", "\n\n",
        "\n", "；", "。", "：",
        "！", "？", "，", " "
    ]
)

# 初始化向量化模型
embeddings = OllamaEmbeddings(
    model=EMBED_MODEL,
    base_url=OLLAMA_BASE_URL
)


def get_doc_tag(file_name: str) -> str:
    """仅给文档打分类标签，用于前端展示溯源，不做检索过滤"""
    fn = file_name.lower()
    if "氢气" in fn or "gb50177" in fn or "gb16993" in fn:
        return "hydrogen"
    elif "gb150" in fn or "hg/t20570" in fn or "安全阀" in fn or "压力容器" in fn or "压力管道" in fn:
        return "general_chem"
    elif "gb50493" in fn or "报警" in fn or "检测报警" in fn:
        return "alarm"
    else:
        return "other"


def load_all_files() -> List[Document]:
    docs = []
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)
        print(f"📂 自动创建文件夹 {DATA_DIR}，请放入所有PDF/Word/Excel/图片规范文件")
        return docs

    file_list = [p for p in DATA_DIR.iterdir() if p.is_file()]
    total_file = len(file_list)
    print(f"📁 文件夹内共 {total_file} 个文件，开始解析")
    for idx, file_path in enumerate(file_list, 1):
        filename = file_path.name
        tag = get_doc_tag(filename)
        try:
            file_docs = []
            if filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(str(file_path))
                file_docs = loader.load()
                print(f"[{idx}/{total_file}] ✅ PDF | {filename} | 分类标签：{tag}")
            elif filename.lower().endswith((".docx", ".doc")):
                loader = UnstructuredWordDocumentLoader(str(file_path), mode="elements")
                file_docs = loader.load()
                print(f"[{idx}/{total_file}] ✅ Word | {filename} | 分类标签：{tag}")
            elif filename.lower().endswith(".xlsx"):
                loader = UnstructuredExcelLoader(str(file_path), mode="elements")
                file_docs = loader.load()
                print(f"[{idx}/{total_file}] ✅ Excel | {filename} | 分类标签：{tag}")
            elif filename.lower().endswith((".jpg", ".png", ".jpeg")):
                loader = UnstructuredImageLoader(str(file_path))
                file_docs = loader.load()
                print(f"[{idx}/{total_file}] ✅ 图片OCR | {filename} | 分类标签：{tag}")
            else:
                print(f"[{idx}/{total_file}] ⏭️ 跳过不支持格式：{filename}")
                continue
            # 完善溯源元数据，前端可展示、定位原文
            for chunk_idx, doc in enumerate(file_docs):
                doc.metadata["source_file"] = filename
                doc.metadata["full_path"] = str(file_path)
                doc.metadata["doc_tag"] = tag
                doc.metadata["chunk_no"] = chunk_idx
            docs.extend(file_docs)
        except Exception as e:
            print(f"[{idx}/{total_file}] ❌ {filename} 解析失败：{str(e)}")
    return docs


def deduplicate_chunks(chunks: List[Document]) -> List[Document]:
    """去除完全重复文本切片，减少向量库冗余、加速检索"""
    seen = set()
    clean = []
    for c in chunks:
        text = c.page_content.strip()
        if text not in seen:
            seen.add(text)
            clean.append(c)
    return clean


def _chunk_id(doc: Document) -> str:
    """以『来源文件 + 内容哈希』为稳定 ID，重复执行脚本可幂等更新"""
    digest = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
    return f"{doc.metadata.get('source_file', 'unknown')}::{digest}"


def update_single_vector_db(split_docs: List[Document]):
    """单库统一增量更新，按内容哈希幂等写入，避免重复切片"""
    ids = [_chunk_id(d) for d in split_docs]
    if DB_DIR.exists() and any(DB_DIR.iterdir()):
        db = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)
        existing = set(db.get(ids=ids)["ids"])
        new_docs, new_ids = [], []
        for d, i in zip(split_docs, ids):
            if i not in existing:
                new_docs.append(d)
                new_ids.append(i)
        if new_docs:
            db.add_documents(new_docs, ids=new_ids)
            print(f"\n🔄 增量追加 {len(new_docs)} 条切片，跳过重复 {len(split_docs) - len(new_docs)} 条")
        else:
            print(f"\n🔄 无新增内容，跳过写入（共 {len(split_docs)} 条均已存在）")
    else:
        print(f"\n🆕 首次创建统一向量库，写入 {len(split_docs)} 条切片")
        db = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=str(DB_DIR),
            ids=ids
        )
    db.persist()
    return db


def main() -> None:
    print("=" * 65)
    print("标准RAG统一向量库构建工具（全文档单库，无检索过滤）")
    print("=" * 65)

    raw_docs = load_all_files()
    if not raw_docs:
        print("\n⚠️ 未读取到任何可解析文档，程序退出")
        return
    print(f"\n📄 原始页面总片段：{len(raw_docs)}")
    print("✂️ 执行国标文档切片拆分...")
    split_chunks = TEXT_SPLITTER.split_documents(raw_docs)
    unique_chunks = deduplicate_chunks(split_chunks)
    print(f"🧹 去重后有效入库切片：{len(unique_chunks)}")

    update_single_vector_db(unique_chunks)
    print("\n🎉 统一向量库更新完成！")
    print("💡 使用说明：")
    print("  1. 所有规范全部放入 ./data，统一存入 ./db 单个向量库")
    print("  2. doc_tag仅做分类标记，检索时不做任何拦截，全量召回所有匹配切片")
    print("  3. LLM会自行筛选检索内容中与问题相关的段落，自动忽略无关文档片段")


if __name__ == "__main__":
    main()
