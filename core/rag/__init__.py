from core.rag.store import ingest, retrieve

__all__ = ["ensure_ingested", "retrieve_knowledge"]


async def ensure_ingested() -> None:
    """
    确保知识库已入库（懒加载，已入库则跳过）。
    建议在首次评分前调用。
    """
    await ingest(force=False)


async def retrieve_knowledge(query: str, top_k: int = 5) -> str:
    """
    检索相关知识片段，拼接为统一字符串返回。
    """
    docs = await retrieve(query, top_k=top_k)
    if not docs:
        return ""
    return "\n\n---\n\n".join(docs)
