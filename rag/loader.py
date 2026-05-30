import hashlib
import re
from pathlib import Path


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


def load_and_chunk(
    data_dir: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[dict]:
    """
    加载 + 标题感知分块，一步完成。
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
