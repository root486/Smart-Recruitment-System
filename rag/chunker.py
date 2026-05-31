"""
文本切分工具模块。
- 标题感知的 Markdown 段落分割
- 段落聚合 / 句子级截断
- 稳定 chunk ID 生成
"""
import hashlib
import re


# ── 两级分块常量 ──

PARENT_SIZE = 1000
PARENT_OVERLAP = 200
CHILD_SIZE = 400
CHILD_OVERLAP = 100


# ── Markdown 标题解析 ──

def parse_heading_path(text: str) -> list[tuple[int, str, str]]:
    """
    解析 Markdown 标题层级，返回每个行所属的标题路径。
    返回: [(行号, 行文本, 标题路径), ...]
    """
    lines = text.split("\n")
    heading_stack: list[tuple[int, str]] = []
    sections = []

    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+(.+)", line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))

        path = " > ".join([h[1] for h in heading_stack]) if heading_stack else ""
        sections.append((i, line, path))

    return sections


def build_sections_with_context(text: str) -> list[dict]:
    """
    将文本按标题切分为逻辑段，每段携带标题上下文。
    返回: [{"content": str, "context": str}, ...]
    """
    lines = text.split("\n")
    heading_info = parse_heading_path(text)

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

        # 跳过只有标题、没有正文的段（如孤立的 # 文档标题）
        body_lines = [l for l in section_lines[1:] if l.strip() and not re.match(r"^#{1,6}\s+", l)]
        if not body_lines:
            continue

        path = heading_info[h_idx][2] if h_idx < len(heading_info) else ""
        sections.append({"content": section_text, "context": path})

    if heading_indices and heading_indices[0] > 0:
        preamble = "\n".join(lines[:heading_indices[0]]).strip()
        if preamble:
            sections.insert(0, {"content": preamble, "context": ""})

    return sections


# ── 段落聚合 / 句子切分 ──

def aggregate_paragraphs(paragraphs: list[str], chunk_size: int) -> list[str]:
    """将段落列表按 chunk_size 聚合，不切断段落边界。"""
    chunks = []
    buf = ""
    for para in paragraphs:
        tentative = f"{buf}\n\n{para}" if buf else para
        if len(tentative) <= chunk_size:
            buf = tentative
        else:
            if buf:
                chunks.append(buf)
            if len(para) > chunk_size:
                chunks.append(para)
                buf = ""
            else:
                buf = para
    if buf:
        chunks.append(buf)
    return chunks


def split_long_paragraph(text: str, chunk_size: int, overlap: int) -> list[str]:
    """将超长段落按句子边界截断，保留 overlap。"""
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

    if len(chunks) > 1 and overlap > 0:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(prev_tail + "\n" + chunks[i])
        return overlapped

    return chunks


# ── 稳定 ID ──

def make_chunk_id(source: str, chunk_index: int, content_preview: str, level: str = "") -> str:
    """生成稳定的 chunk ID（SHA256 前 12 位）。"""
    raw = f"{source}:{level}:{chunk_index}:{content_preview[:200]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
