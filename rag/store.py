import hashlib
import os
import pickle
from pathlib import Path

import chromadb
import jieba
from loguru import logger
from rank_bm25 import BM25Okapi
from settings import settings
from rag.embedder import embed_documents, embed_query
from rag.loader import load_and_chunk, load_and_chunk_with_context
from rag.reranker import dashscope_rerank

COLLECTION_NAME = "hr_knowledge_base"
FINGERPRINT_KEY = "data_fingerprint"
BM25_STATE_FILE = "bm25_state.pkl"

# 混合检索配置
RRF_K = 60  # RRF 平滑常数
RECALL_MULTIPLIER = 4  # 粗排召回倍数（召回 top_k * n 个结果用于精排）


def _compute_files_fingerprint(data_dir: str) -> str:
    """
    计算 data/ 目录下所有文件的 SHA256 指纹。
    文件相对路径 + 文件内容一起参与哈希——
    增/删/改/重命名文件都会导致指纹变化。
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return ""

    hasher = hashlib.sha256()
    for file_path in sorted(data_path.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in {".md", ".txt"}:
            continue
        hasher.update(str(file_path.relative_to(data_path)).encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(file_path.read_bytes())
    return hasher.hexdigest()


def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)


def _get_collection() -> chromadb.Collection:
    client = _get_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


# ── BM25 索引管理 ──

_bm25_index: BM25Okapi | None = None
_bm25_texts: list[str] = []


def _bm25_state_path() -> str:
    return os.path.join(settings.CHROMA_DB_PATH, BM25_STATE_FILE)


def _save_bm25_to_disk():
    """持久化 BM25 索引到磁盘，避免重启后重建。"""
    global _bm25_index, _bm25_texts
    if _bm25_index is None:
        return
    state = {"index": _bm25_index, "texts": _bm25_texts}
    try:
        with open(_bm25_state_path(), "wb") as f:
            pickle.dump(state, f)
        logger.debug(f"BM25 索引已保存到 {_bm25_state_path()}")
    except Exception as e:
        logger.warning(f"BM25 索引保存失败: {e}")


def _load_bm25_from_disk() -> bool:
    """从磁盘加载 BM25 索引，成功返回 True。"""
    global _bm25_index, _bm25_texts
    path = _bm25_state_path()
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:
            state = pickle.load(f)
        _bm25_index = state["index"]
        _bm25_texts = state["texts"]
        logger.info(f"RAG: BM25 索引已从磁盘加载，共 {len(_bm25_texts)} 篇")
        return True
    except Exception as e:
        logger.warning(f"BM25 索引加载失败，将从 ChromaDB 重建: {e}")
        return False


def _remove_bm25_state():
    """删除磁盘上的 BM25 状态文件。"""
    path = _bm25_state_path()
    if os.path.exists(path):
        os.remove(path)


def _build_bm25_index(texts: list[str]) -> BM25Okapi:
    """用 jieba 分词构建 BM25 索引。"""
    tokenized = [list(jieba.cut(t)) for t in texts]
    return BM25Okapi(tokenized)


# ── 入库 ──

async def ingest(force: bool = False) -> int:
    """
    将 data/ 目录下的文档向量化后存入 ChromaDB，并构建 BM25 索引。
    - 对比文件指纹：指纹不变则跳过，指纹变化则自动重建
    - force=True 强制清空重建，不对比指纹
    返回本次入库的 chunk 数量（0 表示跳过）。
    """
    global _bm25_index, _bm25_texts

    current_fingerprint = _compute_files_fingerprint(settings.DATA_DIR)
    if not current_fingerprint:
        logger.info("RAG: data/ 目录为空，跳过入库")
        return 0

    collection = _get_collection()

    # 指纹未变 → 跳过
    if not force:
        stored = collection.metadata
        if stored and stored.get(FINGERPRINT_KEY) == current_fingerprint:
            logger.info(f"RAG: 知识库未变更，跳过入库 (指纹: {current_fingerprint[:8]}...)")
            # 懒加载 BM25 索引（优先读磁盘，失败则从 ChromaDB 重建）
            if _bm25_index is None and collection.count() > 0:
                if not _load_bm25_from_disk():
                    all_docs = collection.get()
                    _bm25_texts = all_docs.get("documents", [])
                    if _bm25_texts:
                        _bm25_index = _build_bm25_index(_bm25_texts)
                        _save_bm25_to_disk()
                        logger.info(f"RAG: BM25 索引已重建，共 {len(_bm25_texts)} 篇")
            return 0

    # 指纹变化 → 清空重建
    if collection.count() > 0:
        logger.info("RAG: 检测到知识库变更，清空重建...")
        client = _get_client()
        client.delete_collection(COLLECTION_NAME)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        _bm25_index = None
        _bm25_texts = []
        _remove_bm25_state()

    # 分块 + Contextual Retrieval 上下文注入 + 向量化 + 写入
    chunks = await load_and_chunk_with_context(settings.DATA_DIR)
    if not chunks:
        logger.info("RAG: data/ 目录为空，跳过入库")
        return 0

    texts = [c["content"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {"source": c["source"], "title": c["title"], "chunk_index": c["chunk_index"]}
        for c in chunks
    ]

    batch_size = 10  # DashScope API 单次最多 10 条
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        batch_metas = metadatas[i : i + batch_size]

        vectors = await embed_documents(batch_texts)
        collection.add(
            ids=batch_ids,
            embeddings=vectors,
            documents=batch_texts,
            metadatas=batch_metas,
        )
        total += len(batch_texts)

    # 构建 BM25 索引并持久化
    _bm25_texts = texts
    _bm25_index = _build_bm25_index(texts)
    _save_bm25_to_disk()
    logger.info(f"RAG: BM25 索引构建完成，共 {len(texts)} 篇")

    # 存入指纹
    client = _get_client()
    collection.modify(metadata={FINGERPRINT_KEY: current_fingerprint})

    logger.info(f"RAG: 知识库入库完成，共 {total} 个 chunk (指纹: {current_fingerprint[:8]}...)")
    return total


# ── 检索 ──

def _rrf_fusion(
    dense_ranked: list[tuple[int, str]],
    sparse_ranked: list[tuple[int, str]],
    top_k: int,
) -> list[str]:
    """
    RRF (Reciprocal Rank Fusion) 融合稠密和稀疏检索结果。
    score = Σ 1/(k + rank)
    """
    scores: dict[str, float] = {}

    for rank, (_, doc) in enumerate(dense_ranked):
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (RRF_K + rank + 1)

    for rank, (_, doc) in enumerate(sparse_ranked):
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (RRF_K + rank + 1)

    seen = set()
    merged = []
    for doc, _ in sorted(scores.items(), key=lambda x: -x[1]):
        if doc not in seen:
            seen.add(doc)
            merged.append(doc)
            if len(merged) >= top_k:
                break
    return merged


async def retrieve(query: str, top_k: int = 5) -> list[str]:
    """
    混合检索 = 稠密向量 + BM25 稀疏 → RRF 融合 → DashScope Rerank 精排。
    双向降级：
    - BM25 不可用 → 纯稠密检索
    - Rerank 失败   → 返回 RRF 融合结果
    """
    global _bm25_index, _bm25_texts

    collection = _get_collection()
    if collection.count() == 0:
        return []

    # 懒加载 BM25（若尚未加载）
    if _bm25_index is None and collection.count() > 0:
        if not _load_bm25_from_disk():
            all_docs = collection.get()
            _bm25_texts = all_docs.get("documents", [])
            if _bm25_texts:
                _bm25_index = _build_bm25_index(_bm25_texts)
                _save_bm25_to_disk()

    recall_k = top_k * RECALL_MULTIPLIER

    # ── 第一路：稠密向量检索 ──
    query_vec = await embed_query(query)
    dense_results = collection.query(
        query_embeddings=[query_vec],
        n_results=recall_k,
    )
    dense_docs = dense_results.get("documents", [[]])[0]
    dense_ranked = [(i, d) for i, d in enumerate(dense_docs) if d]

    # ── 第二路：BM25 稀疏检索 ──
    sparse_ranked: list[tuple[int, str]] = []
    if _bm25_index is not None and _bm25_texts:
        try:
            tokenized_query = list(jieba.cut(query))
            bm25_scores = _bm25_index.get_scores(tokenized_query)
            indexed_scores = list(enumerate(bm25_scores))
            indexed_scores.sort(key=lambda x: -x[1])
            for idx, score in indexed_scores[:recall_k]:
                if score > 0 and idx < len(_bm25_texts):
                    sparse_ranked.append((idx, _bm25_texts[idx]))
        except Exception as e:
            logger.warning(f"BM25 检索异常，降级为纯稠密检索: {e}")

    # ── RRF 融合 ──
    if sparse_ranked:
        merged = _rrf_fusion(dense_ranked, sparse_ranked, recall_k)
        logger.debug(f"RRF 融合: 稠密 {len(dense_ranked)} + 稀疏 {len(sparse_ranked)} -> {len(merged)}")
    else:
        merged = [d for _, d in dense_ranked]
        logger.debug("BM25 不可用，使用纯稠密检索结果")

    # ── DashScope Rerank 精排 ──
    if merged:
        rerank_results = await dashscope_rerank(query, merged, top_n=top_k)
        if rerank_results:
            reranked = [merged[idx] for idx, _ in rerank_results if idx < len(merged)]
            logger.debug(f"Rerank 完成: {len(merged)} -> {len(reranked)}")
            return reranked[:top_k]
        else:
            logger.debug("Rerank 失败，降级为 RRF 融合结果")

    return merged[:top_k]
