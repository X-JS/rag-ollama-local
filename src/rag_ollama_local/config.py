import sys
from pathlib import Path

# 保证 Windows 控制台（GBK）也能正常输出 emoji
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 项目根目录（data/db 统一放在仓库根下，与运行目录无关）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 数据与向量库目录
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = PROJECT_ROOT / "db"  # 全部文档统一存入这一个向量库

# ========== Ollama / 大模型配置（两个脚本共用）==========
OLLAMA_BASE_URL = "http://192.168.211.163:11434"
# 生成（对话）用的模型
MODEL_NAME = "qwen2.5:3b"
# 向量化（embedding）用的模型，建议用专门的 embedding 模型
EMBED_MODEL = "nomic-embed-text"

# ========== 文本切片配置（load_data_to_db 使用）==========
CHUNK_SIZE = 1600
CHUNK_OVERLAP = 350

# ========== 问答模式开关 ==========
# True  = 双模模式：检索到相关信息走 RAG 并标注来源；未检索到则用大模型正常聊天
# False = 严格模式：未检索到相关信息时，直接回复「未找到相关信息」，不进入聊天
DUAL_MODE = False
