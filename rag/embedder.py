import httpx
from settings import settings

EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"


async def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    批量生成文本向量（异步）。
    直接调 DashScope compatible API。
    """
    headers = {
        "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            EMBEDDING_URL,
            json={
                "model": EMBEDDING_MODEL,
                "input": texts,
            },
            headers=headers,
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        # compatible 模式返回格式同 OpenAI：{"data": [{"embedding": [...], "index": 0}, ...]}
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]


async def embed_query(text: str) -> list[float]:
    """
    单条查询文本生成向量（异步）。
    """
    vectors = await embed_documents([text])
    return vectors[0]


def embed_documents_sync(texts: list[str]) -> list[list[float]]:
    """
    批量生成文本向量（同步封装，用于入库等非异步上下文）。
    """
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(embed_documents(texts))
    raise RuntimeError("embed_documents_sync 不能在已有事件循环中调用，请使用 await embed_documents()")
