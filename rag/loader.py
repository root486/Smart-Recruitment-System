import hashlib
import re
from pathlib import Path

import httpx
from loguru import logger
from settings import settings

# ── Contextual Retrieval 配置 ──
CONTEXT_LLM_MODEL = "qwen-plus"  # 低成本模型，¥0.0015/千token，专用于离线上下文生成
CONTEXT_LLM_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
CONTEXT_PROMPT_TEMPLATE = """<document>
{full_document}
</document>

Here is the chunk we want to situate within the whole document:

<chunk>
{chunk_content}
</chunk>

Please give a short, succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. Use Chinese."""


async def _generate_chunk_context(
    full_document: str,
    chunk_content: str,
) -> str:
    """
    调用 LLM 为单个 chunk 生成上下文描述。
    使用 qwen-plus，一次调用生成一个 chunk 的上下文。
    """
    prompt = CONTEXT_PROMPT_TEMPLATE.format(
        full_document=full_document,
        chunk_content=chunk_content,
    )
    headers = {
        "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CONTEXT_LLM_MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 200,
        "temperature": 0.0,
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                CONTEXT_LLM_URL,
                json=payload,
                headers=headers,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            context = data["choices"][0]["message"]["content"].strip()
            return context
    except Exception as e:
        logger.warning(f"生成 chunk 上下文失败，使用空上下文: {e}")
        return ""


async def contextualize_chunks(
    chunks: list[dict],
    doc_full_texts: dict[str, str],
) -> list[dict]:
    """
    Contextual Retrieval：为每个 chunk 注入 LLM 生成的上下文前缀。
    
    这是 Anthropic 2024-2025 年提出的核心优化：
    - 入库前用 LLM 为每个 chunk 生成 50-100 字的"这段内容在哪、讲什么"的描述
    - 将描述拼到 chunk 前面，再做 Embedding 和 BM25 索引
    - 同时改善稠密检索和稀疏检索两条通路
    
    参数:
        chunks: load_and_chunk() 的输出
        doc_full_texts: {"source_path": "完整文档内容", ...}
    
    返回:
        注入上下文后的 chunks（content 字段前增加了上下文描述）
    """
    # 按 source 分组，批量处理
    by_source: dict[str, list[int]] = {}
    for i, ch in enumerate(chunks):
        by_source.setdefault(ch["source"], []).append(i)

    total = len(chunks)
    processed = 0

    for source, indices in by_source.items():
        full_text = doc_full_texts.get(source, "")
        if not full_text:
            continue

        for idx in indices:
            chunk = chunks[idx]
            context = await _generate_chunk_context(full_text, chunk["content"])
            if context:
                # 格式：[上下文: ...]\n\n<原始内容>
                chunk["content"] = f"[上下文: {context}]\n\n{chunk['content']}"
                chunk["contextualized"] = True
                chunk["context_description"] = context
            processed += 1
            if processed % 10 == 0:
                logger.debug(f"Contextual Retrieval 进度: {processed}/{total}")

    logger.info(f"Contextual Retrieval 完成: {processed}/{total} 个 chunk 已注入上下文")
    return chunks


def load_markdown_files(data_dir: str) -> list[dict]:
    """
    递归扫描 data_dir，读取所有 .md / .txt 文件。
    返回: [{"content": str, "source": str, "title": str}, ...]
    """
    docs = []
    data_path = Path(data_dir)
    if not data_path.exists():
        return docs

    for file_path in sorted(data_path.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in {".md", ".txt"}:
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception:
            continue
        if not text.strip():
            continue

        docs.append({
            "content": text,
            "source": str(file_path.relative_to(data_path)),
            "title": file_path.stem,
        })
    return docs


def _parse_heading_path(text: str) -> list[tuple[int, str, str]]:
    """
    解析 Markdown 标题层级，返回每个段落所属的标题路径。
    返回: [(段落起始位置, 段落文本, 标题路径如 "文档 > 技术栈要求 > 后端"), ...]
    """
    lines = text.split("\n")
    # 当前各层级的标题
    heading_stack = []  # [(level, title), ...]
    sections = []  # [(start_line, end_line, heading_path)]

    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+(.+)", line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            # 弹出 >= 当前层级的标题
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))

        path = " > ".join([h[1] for h in heading_stack]) if heading_stack else ""
        sections.append((i, line, path))

    return sections


def _build_sections_with_context(text: str) -> list[dict]:
    """
    将文本按 ## 标题切分为逻辑段，每段携带标题上下文。
    返回: [{"content": str, "context": str}, ...]
    """
    lines = text.split("\n")
    heading_info = _parse_heading_path(text)

    # 找到所有 ##+ 标题行索引
    heading_indices = []
    for i, line in enumerate(lines):
        if re.match(r"^#{1,6}\s+", line):
            heading_indices.append(i)

    if not heading_indices:
        return [{"content": text.strip(), "context": ""}]

    sections = []
    for idx, h_idx in enumerate(heading_indices):
        start = h_idx
        end = heading_indices[idx + 1] if idx + 1 < len(heading_indices) else len(lines)
        section_lines = lines[start:end]
        section_text = "\n".join(section_lines).strip()
        if not section_text:
            continue
        # 获取该段落的标题路径（取第一个非空路径）
        path = heading_info[h_idx][2] if h_idx < len(heading_info) else ""
        sections.append({"content": section_text, "context": path})

    # 处理第一个标题之前的内容（如果有）
    if heading_indices and heading_indices[0] > 0:
        preamble = "\n".join(lines[:heading_indices[0]]).strip()
        if preamble:
            sections.insert(0, {"content": preamble, "context": ""})

    return sections


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[dict]:
    """
    标题感知分块：
    1. 先按 ## 标题切分成逻辑段
    2. 每段内部按 \n\n 断段
    3. 每个 chunk 附带标题上下文
    4. chunk 之间有 overlap 防止语义割裂

    返回: [{"content": str, "context": str}, ...]
    """
    sections = _build_sections_with_context(text)
    all_chunks = []

    for section in sections:
        ctx = section["context"]
        body = section["content"]

        # 标题行本身也作为内容保留
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        buf = ""
        section_chunks = []

        for para in paragraphs:
            tentative = f"{buf}\n\n{para}" if buf else para
            if len(tentative) <= chunk_size:
                buf = tentative
            else:
                if buf:
                    section_chunks.append(buf)
                # 如果单段超过 chunk_size，强制截断
                if len(para) > chunk_size:
                    # 按句子截断，保留重叠
                    para_chunks = _split_long_paragraph(para, chunk_size, chunk_overlap)
                    section_chunks.extend(para_chunks)
                    buf = ""
                else:
                    buf = para

        if buf:
            section_chunks.append(buf)

        # 为每个 chunk 添加标题上下文
        for ch in section_chunks:
            if ctx:
                enriched = f"[上下文: {ctx}]\n\n{ch}"
            else:
                enriched = ch
            all_chunks.append({"content": enriched, "context": ctx})

    return all_chunks


def _split_long_paragraph(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    将超长段落按句子边界截断，保留 overlap。
    """
    sentences = re.split(r"(?<=[。！？.!?])\s*", text)
    chunks = []
    buf = ""
    for sent in sentences:
        if not sent.strip():
            continue
        tentative = buf + sent if buf else sent
        if len(tentative) <= chunk_size:
            buf = tentative
        else:
            if buf:
                chunks.append(buf)
            buf = sent
    if buf:
        chunks.append(buf)

    # 添加重叠：每个 chunk 末尾与下一个开头有 overlap
    if len(chunks) > 1 and overlap > 0:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(prev_tail + "\n" + chunks[i])
        return overlapped

    return chunks


def _make_chunk_id(source: str, chunk_index: int, content_preview: str) -> str:
    """
    生成稳定的 chunk ID（SHA256 前 12 位），不受文档顺序影响。
    """
    raw = f"{source}:{chunk_index}:{content_preview[:200]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _load_doc_full_texts(data_dir: str) -> dict[str, str]:
    """
    加载所有文档的完整文本，key 为相对路径（source）。
    供 Contextual Retrieval 使用。
    """
    docs = load_markdown_files(data_dir)
    return {d["source"]: d["content"] for d in docs}


async def load_and_chunk_with_context(
    data_dir: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    enable_contextual: bool = True,
) -> list[dict]:
    """
    加载 + 标题感知分块 + Contextual Retrieval 上下文注入。
    

    1. 标题感知切分
    2. LLM 为每个 chunk 生成上下文前缀
    3. 上下文前缀拼入 chunk → Embedding + BM25 双通路受益
    

    
    参数:
        data_dir: 知识库目录
        chunk_size: 分块大小（字符）
        chunk_overlap: 重叠大小（字符）
        enable_contextual: 是否启用上下文注入（关闭则退化为 load_and_chunk）
    
    返回: [{"content": str, "source": str, "title": str, 
            "chunk_index": int, "chunk_id": str, 
            "contextualized": bool, "context_description": str}, ...]
    """
    # 阶段1：加载文档完整文本（供上下文生成用）
    doc_full_texts = _load_doc_full_texts(data_dir)

    # 阶段2：标题感知切分（同步，保留现有逻辑）
    chunks = load_and_chunk(data_dir, chunk_size, chunk_overlap)
    if not chunks:
        return []

    # 阶段3：Contextual Retrieval 上下文注入（异步，LLM 调用）
    if enable_contextual:
        chunks = await contextualize_chunks(chunks, doc_full_texts)

    return chunks


def load_and_chunk(
    data_dir: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[dict]:
    """
    加载 + 标题感知分块，一步完成（同步版，向后兼容）。
    如需 Contextual Retrieval 增强，请使用 load_and_chunk_with_context()。
    返回: [{"content": str, "source": str, "title": str, "chunk_index": int, "chunk_id": str}, ...]
    """
    docs = load_markdown_files(data_dir)
    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc["content"], chunk_size, chunk_overlap)
        for i, chunk in enumerate(chunks):
            chunk_id = _make_chunk_id(doc["source"], i, chunk["content"])
            all_chunks.append({
                "content": chunk["content"],
                "source": doc["source"],
                "title": doc["title"],
                "chunk_index": i,
                "chunk_id": chunk_id,
            })
    return all_chunks
