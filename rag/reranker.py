"""
DashScope Rerank 重排序模块
使用 gte-rerank-v2 模型对召回结果进行精细排序。
"""
import httpx
from loguru import logger
from settings import settings

RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
RERANK_MODEL = "gte-rerank-v2"


async def dashscope_rerank(
    query: str,
    documents: list[str],
    top_n: int = 5,
) -> list[tuple[int, float]] | None:
    """
    调用 DashScope Rerank API 对文档列表重排序。

    参数:
        query:     用户查询文本
        documents: 候选文档列表（来自混合检索的粗排结果）
        top_n:     返回前 N 个结果

    返回:
        [(原始索引, 相关性分数), ...]  按分数降序排列
        None 表示调用失败，调用方应降级为原始排序
    """
    if not documents:
        return []

    headers = {
        "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": RERANK_MODEL,
        "input": {
            "query": query,
            "documents": documents,
        },
        "parameters": {
            "top_n": min(top_n, len(documents)),
            "return_documents": False,
        },
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                RERANK_URL,
                json=payload,
                headers=headers,
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("output", {}).get("results", [])
            # API 返回格式: [{"index": 0, "relevance_score": 0.95}, ...]
            sorted_results = sorted(results, key=lambda r: r["relevance_score"], reverse=True)
            return [(r["index"], r["relevance_score"]) for r in sorted_results]
    except Exception as e:
        logger.warning(f"DashScope Rerank 调用失败，降级为原始排序: {e}")
        return None
