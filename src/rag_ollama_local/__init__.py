import argparse
import sys
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from rag_ollama_local.config import (
    DB_DIR,
    DUAL_MODE,
    OLLAMA_BASE_URL,
    MODEL_NAME,
    EMBED_MODEL,
)

__version__ = "0.1.0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", action="store_true", help="show version")
    parser.add_argument("-c", "--count", action="store_true", help="show count")
    args = parser.parse_args()

    if args.version:
        print(__version__)
        sys.exit(0)
    if args.count:
        print(db_count())
        sys.exit(0)

    print("Hello from rag-ollama-local!")


def db_count():
    """
    获取向量库数据量
    """
    # 加载本地库
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    vectorstore = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)

    # 获取底层原生 chromadb Collection 对象
    collection = vectorstore._collection

    # 获取向量总 chunk 条数
    return collection.count()