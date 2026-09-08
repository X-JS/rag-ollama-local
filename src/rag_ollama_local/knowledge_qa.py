import json
import time

from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
import gradio as gr

from rag_ollama_local.config import (
    DB_DIR,
    DUAL_MODE,
    OLLAMA_BASE_URL,
    MODEL_NAME,
    EMBED_MODEL,
)

NOT_FOUND_MSG = "未找到相关信息，请换个问法，或确认知识库中已有对应资料。"

DB_DIR = str(DB_DIR)

# ========== 运行日志（右侧面板实时展示）==========
_LOG_LINES: list[str] = []


def _log(msg: str) -> str:
    """追加一条带时间戳的日志，返回当前完整日志文本"""
    _LOG_LINES.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    return "\n".join(_LOG_LINES)


def _current_log() -> str:
    return "\n".join(_LOG_LINES)

# 双模式提示词：资料相关→RAG增强并标来源；资料缺失/无关→正常对话
PROMPT_TEMPLATE = """你是电子产品行业专业知识库助手，请严格遵守以下规则：
1. 下面给出的是知识库中检索到的相关片段，每个片段开头都标注了「来源文件」。如果片段与你的问题相关，请优先依据这些片段作答，分点说明，每一条结论尽量标注对应的来源文件名；
2. 如果检索不到相关资料，或这些片段与问题无关：请用你自身的知识正常回答（可以正常闲聊、对话），但不要谎称引用了资料，也不要在回答中编造来源；
3. 严禁编造检索片段中不存在的术语、工况、标准编号；
4. 回答严谨、专业、结构清晰。不要在回答结尾自行罗列「参考来源」清单，系统会在末尾自动追加去重后的来源。

### 知识库检索片段：
{context}

用户提问：{question}
"""


_CACHED_COMPONENTS = None


def build_components():
    """初始化大模型与向量库，并缓存复用（懒加载：首次提问才联网，后续复用实例）"""
    global _CACHED_COMPONENTS
    if _CACHED_COMPONENTS is not None:
        return _CACHED_COMPONENTS
    # temperature=0关闭创造性，杜绝幻觉；num_ctx 调小以减少预填充耗时
    llm = Ollama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_ctx=4096,
        num_predict=1024,  # 限制最大生成长度，避免一直生成不停/失控
    )
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    _CACHED_COMPONENTS = (llm, vectorstore)
    return _CACHED_COMPONENTS


def _extract_sources(docs):
    """去重提取来源文件名（兼容 source / source_file 元数据）"""
    names, seen = [], set()
    for d in docs:
        src = d.metadata.get("source_file") or d.metadata.get("source") or "未知文件"
        if src not in seen:
            seen.add(src)
            names.append(src)
    return names


# 问答逻辑（生成器）：边处理边推送日志到右侧面板，回答则流式输出到左侧
def chat_answer(question):
    _LOG_LINES.clear()
    _log(f"收到问题：{question}")

    # Chroma 的 similarity_search_with_score 返回的是 L² 距离（越小越相似，非 0-1 相似度）
    SIM_THRESHOLD = 500.0  # 阈值越大越宽松，越小越严格；请按你的 embedding 模型微调

    _log("加载大模型与向量库…")
    llm, vector_store = build_components()
    _log("模型与向量库就绪")

    _log("正在向量化并检索相似片段…")
    docs_with_score = vector_store.similarity_search_with_score(question, k=3)
    _log(f"相似度检索完成，返回 {len(docs_with_score)} 个片段")
    _log("检索详情：\n" + json.dumps(
        [
            {
                "score": round(score, 1),
                "source": doc.metadata.get("source_file") or doc.metadata.get("source"),
                "preview": doc.page_content[:50],
            }
            for doc, score in docs_with_score
        ],
        ensure_ascii=False,
        indent=2,
        default=str,
    ))

    valid_docs = [doc for doc, score in docs_with_score if score <= SIM_THRESHOLD]
    _log(f"按距离阈值 <= {SIM_THRESHOLD} 过滤，有效片段 {len(valid_docs)} 个")

    answer = ""
    if valid_docs:
        # ==========检索到有效文档，走RAG流程==========
        _log("命中知识库：走 RAG 增强问答")
        # 每个片段开头标注来源文件名，供 LLM 行内引用（同一文件的多个片段文件名相同）
        context = "\n\n".join(
            f"[片段{i + 1}｜来源文件：{d.metadata.get('source_file') or d.metadata.get('source') or '未知文件'}]\n{d.page_content}"
            for i, d in enumerate(valid_docs)
        )
        prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
        prompt_text = prompt.format(context=context, question=question)
        _log("开始流式生成回答…")
        for chunk in llm.stream(prompt_text):
            answer += chunk
            yield answer, _current_log()
    else:
        if not DUAL_MODE:
            # ==========严格模式：未检索到相关信息，直接告知，不进入聊天==========
            _log("当前为严格模式：未检索到相关信息，直接回复未找到")
            yield NOT_FOUND_MSG, _current_log()
            return
        # ==========无有效匹配，直接大模型闲聊，不拼接知识库context==========
        _log("无匹配资料：使用大模型直接闲聊")
        for chunk in llm.stream(question):
            answer += chunk
            yield answer, _current_log()

    if valid_docs:
        # 参考来源由程序按文件名去重后统一追加，避免 LLM 把正文里的公司名重复罗列
        sources = _extract_sources(valid_docs)
        if sources:
            answer = answer.rstrip() + "\n\n参考来源：\n" + "\n".join(
                f"{i + 1}. {name}" for i, name in enumerate(sources)
            )
            _log(f"追加去重后的参考来源 {len(sources)} 条")

    _log("回答生成完成")
    yield answer.strip(), _current_log()


# 注入到页面 page-load 的脚本：
#   Enter（无修饰键）→ 发送：拦截默认换行，再向文本框补发一次 Enter，让 Gradio 的 submit 事件触发
#   Ctrl/Cmd+Enter → 换行（显式插入）
# 只处理真实按键事件（isTrusted），避免补发事件导致死循环
ENTER_TO_SEND_JS = """
(() => {
    document.addEventListener('keydown', function (e) {
        if (e.key !== 'Enter' || !e.isTrusted) return;
        const el = document.activeElement;
        if (!el || el.tagName !== 'TEXTAREA') return;

        if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            e.stopImmediatePropagation();
            document.execCommand('insertText', false, '\\n');
        } else if (!e.shiftKey && !e.isComposing) {
            e.preventDefault();
            e.stopImmediatePropagation();
            const btn = document.querySelector('#submit_btn');
            if (btn) btn.click();
        }
    }, true);
})();
"""


# 日志框自动滚动到底部（每 300ms 检查一次，日志多时也能跟上）
AUTO_SCROLL_JS = """
setInterval(() => {
    const t = document.querySelector('#log_box textarea');
    if (t) t.scrollTop = t.scrollHeight;
}, 300);
"""

# 让回答框/日志框的文本域撑满所在行的剩余高度。
# 用「行底部 - 文本域顶部」计算，二者同为视口坐标，滚动偏移互相抵消 → 滚动时高度稳定不变
FIT_HEIGHTS_JS = """
function fitHeights() {
    document.querySelectorAll('#answer_box, #log_box').forEach(function (box) {
        const ta = box.querySelector('textarea');
        const row = box.closest('.row');
        if (!ta || !row) return;
        const taTop = ta.getBoundingClientRect().top;
        const rowBottom = row.getBoundingClientRect().bottom;
        const avail = rowBottom - taTop - 12;  // 12px 底部留白
        ta.style.height = Math.max(120, avail) + 'px';
    });
}
window.addEventListener('load', fitHeights);
window.addEventListener('resize', fitHeights);
setInterval(fitHeights, 600);
"""

# 让两列等高、且行本身撑满剩余高度；回答框/日志框（各自列最后一个 form）占满剩余高度。
# body 禁止整页滚动：两栏各自内部滚动，避免「越滚越高」的回环
CUSTOM_CSS = """
html, body { height: 100%; overflow: hidden !important; }
#submit_btn { display: none !important; }
/* 让两列等高、且行本身撑满剩余高度 */
.row.grow-children { flex: 1 1 auto !important; align-items: stretch !important; }
.row.grow-children > .column { display: flex !important; flex-direction: column !important; min-height: 0 !important; }
/* 每列最后一个 .form（回答框 / 日志框）占满剩余高度 */
.row.grow-children > .column .form:last-child { flex: 1 1 auto !important; min-height: 0 !important; }
#answer_box, #log_box { flex: 1 1 auto !important; min-height: 0 !important; }
"""

# 一并注入到 <head>：回车发送脚本、日志自动滚动脚本、高度自适应脚本、布局样式
HEAD_HTML = (
    f"<script>{ENTER_TO_SEND_JS}</script>"
    f"<script>{AUTO_SCROLL_JS}</script>"
    f"<script>{FIT_HEIGHTS_JS}</script>"
    f"<style>{CUSTOM_CSS}</style>"
)


def main() -> None:
    # 搭建网页界面：左右分栏（左1/3 右2/3）；fill_height + scale 让两栏纵向填满页面
    with gr.Blocks(title="本地私有RAG知识库问答系统", theme=gr.themes.Soft(), fill_height=True) as demo:
        gr.Markdown("# 🔒 离线私有知识库AI问答工具")
        gr.Markdown("数据完全本地存储，不上传云端；有资料自动溯源，无资料可正常对话")
        with gr.Row(scale=1):
            with gr.Column(scale=1):
                question_input = gr.Textbox(
                    label="输入你的问题",
                    placeholder="Enter 发送 / Ctrl+Enter 换行",
                    lines=3,
                )
                # 隐藏的提交按钮：不可见，仅供 JS 触发提交（Gradio 多行文本框回车不会触发 submit 事件）
                submit_btn = gr.Button("提交查询", variant="primary", elem_id="submit_btn")
                answer_out = gr.Textbox(
                    label="AI回答（附带资料溯源）",
                    lines=6,
                    scale=1,
                    interactive=False,
                    elem_id="answer_box",
                )
            with gr.Column(scale=2):
                log_out = gr.Textbox(
                    label="运行日志（实时）",
                    lines=6,
                    scale=1,
                    interactive=False,
                    max_lines=500,
                    elem_id="log_box",
                )
        submit_btn.click(fn=chat_answer, inputs=question_input, outputs=[answer_out, log_out])

    # 仅本地访问，自动弹出浏览器页面；注入「回车发送 + 日志自动滚动」脚本与样式
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        head=HEAD_HTML,
    )


if __name__ == "__main__":
    main()
