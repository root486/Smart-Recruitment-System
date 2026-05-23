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


def chunk_text(text: str, chunk_size: int = 800) -> list[str]:
    """
    按段落边界切分文本，每个 chunk 尽量不超过 chunk_size 字符。
    优先在 \\n\\n 处断开，保持语义完整。
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buf = ""
    for para in paragraphs:
        tentative = f"{buf}\n\n{para}" if buf else para
        if len(tentative) <= chunk_size:
            buf = tentative
        else:
            if buf:
                chunks.append(buf)
            buf = para
    if buf:
        chunks.append(buf)
    return chunks


def load_and_chunk(data_dir: str, chunk_size: int = 800) -> list[dict]:
    """
    加载 + 分块，一步完成。
    返回: [{"content": str, "source": str, "title": str, "chunk_index": int}, ...]
    """
    docs = load_markdown_files(data_dir)
    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc["content"], chunk_size)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "content": chunk,
                "source": doc["source"],
                "title": doc["title"],
                "chunk_index": i,
            })
    return all_chunks
