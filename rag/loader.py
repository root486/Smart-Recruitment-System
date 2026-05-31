"""文档加载 + 父子块切分入口。"""
from pathlib import Path

from rag.chunker import (
    build_sections_with_context,
    make_chunk_id,
    split_long_paragraph,
    PARENT_SIZE,
    PARENT_OVERLAP,
    CHILD_SIZE,
    CHILD_OVERLAP,
)


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


def load_and_chunk(
    data_dir: str,
    parent_size: int = PARENT_SIZE,
    parent_overlap: int = PARENT_OVERLAP,
    child_size: int = CHILD_SIZE,
    child_overlap: int = CHILD_OVERLAP,
    files: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    两级父子块切分：
    1. 父块：标题感知，~1000 字，保留完整段落语义
    2. 子块：从父块中拆分，~400 字，更精细的检索粒度

    检索时用子块匹配，返回父块内容。

    参数:
        files: 可选，只处理指定相对路径的文件列表（增量更新时使用）。

    返回: (parent_chunks, child_chunks)
    """
    docs = load_markdown_files(data_dir)
    if files is not None:
        docs = [d for d in docs if d["source"] in files]
    parent_chunks = []
    child_chunks = []

    for doc in docs:
        sections = build_sections_with_context(doc["content"])
        parent_idx = 0

        for section in sections:
            ctx = section["context"]
            body = section["content"]

            paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]

            # 聚合为父块
            parent_texts = []
            buf = ""
            for para in paragraphs:
                tentative = f"{buf}\n\n{para}" if buf else para
                if len(tentative) <= parent_size:
                    buf = tentative
                else:
                    if buf:
                        parent_texts.append(buf)
                    if len(para) > parent_size:
                        for sub in split_long_paragraph(para, parent_size, parent_overlap):
                            parent_texts.append(sub)
                        buf = ""
                    else:
                        buf = para
            if buf:
                parent_texts.append(buf)

            # 为每条父块文本生成父块 + 子块
            for p_text in parent_texts:
                parent_id = make_chunk_id(doc["source"], parent_idx, p_text, level="parent")

                if ctx:
                    parent_content = f"[{ctx}]\n\n{p_text}"
                else:
                    parent_content = p_text

                parent_chunks.append({
                    "content": parent_content,
                    "source": doc["source"],
                    "title": doc["title"],
                    "chunk_id": parent_id,
                    "chunk_index": parent_idx,
                })

                # 从父块中拆分子块
                child_paragraphs = [p.strip() for p in p_text.split("\n\n") if p.strip()]
                child_texts = []
                buf = ""
                for cp in child_paragraphs:
                    tentative = f"{buf}\n\n{cp}" if buf else cp
                    if len(tentative) <= child_size:
                        buf = tentative
                    else:
                        if buf:
                            child_texts.append(buf)
                        if len(cp) > child_size:
                            for sub in split_long_paragraph(cp, child_size, child_overlap):
                                child_texts.append(sub)
                            buf = ""
                        else:
                            buf = cp
                if buf:
                    child_texts.append(buf)

                for c_idx, c_text in enumerate(child_texts):
                    # 跳过过短的子块（仅标题无正文，< 60 字）
                    if len(c_text) < 60:
                        continue

                    child_id = make_chunk_id(doc["source"], parent_idx * 1000 + c_idx, c_text, level="child")

                    if ctx and len(child_texts) == 1:
                        child_content = f"[{ctx}]\n\n{c_text}"
                    else:
                        child_content = c_text

                    child_chunks.append({
                        "content": child_content,
                        "source": doc["source"],
                        "title": doc["title"],
                        "chunk_id": child_id,
                        "chunk_index": parent_idx * 1000 + c_idx,
                        "parent_chunk_id": parent_id,
                    })

                parent_idx += 1

    return parent_chunks, child_chunks
