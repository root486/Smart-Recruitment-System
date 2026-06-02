"""RAG 入库与检索编排。"""
import jieba
from loguru import logger
from settings import settings
from rag.embedder import embed_documents_sync, embed_query
from rag.loader import load_and_chunk
from rag.reranker import dashscope_rerank
from rag.state import (
    COLLECTION_NAME,
    RRF_K,
    RECALL_MULTIPLIER,
    AUTO_MERGE_THRESHOLD,
    compute_file_hashes,
    save_file_hashes,
    load_file_hashes,
    get_collection,
    delete_collection,
    get_bm25_index,
    get_bm25_texts,
    set_bm25_index,
    set_bm25_texts,
    build_bm25_index,
    save_bm25,
    load_bm25,
    get_parent_store,
    set_parent_store,
    save_parent_store,
    load_parent_store,
    get_child_to_parent,
    set_child_to_parent,
    save_child_parent,
    load_child_parent,
    remove_chunks_by_source,
    reset_state,
)


# ── 入库 ──

async def ingest(force: bool = False) -> int:
    """
    将 data/ 目录下的文档向量化后存入 ChromaDB，并构建 BM25 索引。
    - 文件级增量更新：只重建变更文件，删除已移除文件
    - force=True → 强制全量重建
    返回子块数量（0 = 跳过）。
    """
    current_hashes = compute_file_hashes(settings.DATA_DIR)
    if not current_hashes:
        logger.info("RAG: data/ 目录为空，跳过入库")
        return 0

    collection = get_collection()
    prev_hashes = load_file_hashes()

    # 全量重建模式
    if force or not prev_hashes:
        if collection.count() > 0:
            logger.info("RAG: 全量重建...")
            delete_collection()
            collection = get_collection()
            reset_state()

        parent_chunks, child_chunks = load_and_chunk(settings.DATA_DIR)
        if not child_chunks:
            logger.info("RAG: data/ 目录为空，跳过入库")
            return 0

        total = await _embed_and_store(collection, child_chunks)
        _build_mappings(parent_chunks, child_chunks)
        _rebuild_bm25([c["content"] for c in child_chunks])
        save_file_hashes(current_hashes)
        logger.info(f"RAG: 全量入库完成 — 父块 {len(parent_chunks)} + 子块 {total}")
        return total

    # 文件级增量模式
    added = {f: current_hashes[f] for f in current_hashes if f not in prev_hashes}
    changed = {f: current_hashes[f] for f in current_hashes if f in prev_hashes and current_hashes[f] != prev_hashes[f]}
    removed = [f for f in prev_hashes if f not in current_hashes]

    if not added and not changed and not removed:
        logger.info("RAG: 知识库未变更，跳过入库")
        return 0

    # 删除已移除文件的块
    for rm_source in removed:
        n = remove_chunks_by_source(rm_source)
        if n > 0:
            _cleanup_mappings_for_source(rm_source)

    # 变更文件：先删旧块，再重新索引
    for ch_source in changed:
        remove_chunks_by_source(ch_source)
        _cleanup_mappings_for_source(ch_source)

    # 只加载变更/新增的文件
    to_process = list(added.keys()) + list(changed.keys())
    if to_process:
        logger.info(f"RAG: 增量更新 — 新增 {len(added)} + 变更 {len(changed)} + 删除 {len(removed)} 个文件")
        parent_chunks, child_chunks = load_and_chunk(settings.DATA_DIR, files=to_process)
        if child_chunks:
            await _embed_and_store(collection, child_chunks)
            _build_mappings(parent_chunks, child_chunks, merge=True)

        # 重建 BM25（从当前全部子块）
        all_docs = collection.get()
        all_texts = all_docs.get("documents", [])
        if all_texts:
            _rebuild_bm25(all_texts)

    save_file_hashes(current_hashes)
    total = collection.count()
    logger.info(f"RAG: 增量入库完成 — 当前共 {total} 个子块")
    return total


# ── 入库辅助 ──

async def _embed_and_store(collection, child_chunks: list[dict]) -> int:
    """子块入库 ChromaDB。"""
    from rag.embedder import embed_documents

    texts = [c["content"] for c in child_chunks]
    ids = [c["chunk_id"] for c in child_chunks]
    metadatas = [
        {"source": c["source"], "title": c["title"],
         "chunk_index": c["chunk_index"], "parent_chunk_id": c["parent_chunk_id"]}
        for c in child_chunks
    ]

    batch_size = 10
    total = 0
    for i in range(0, len(child_chunks), batch_size):
        vectors = await embed_documents(texts[i : i + batch_size])
        collection.add(
            ids=ids[i : i + batch_size],
            embeddings=vectors,
            documents=texts[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )
        total += len(texts[i : i + batch_size])
    return total


def _build_mappings(parent_chunks: list[dict], child_chunks: list[dict], merge: bool = False) -> None:
    """构建父块和子→父映射，merge=True 时合并到已有映射。"""
    new_parents = {pc["chunk_id"]: pc["content"] for pc in parent_chunks}
    new_child_parent = {cc["content"]: cc["parent_chunk_id"] for cc in child_chunks}

    if merge:
        parent_store = get_parent_store()
        parent_store.update(new_parents)
        set_parent_store(parent_store)
        child_parent = get_child_to_parent()
        child_parent.update(new_child_parent)
        set_child_to_parent(child_parent)
    else:
        set_parent_store(new_parents)
        set_child_to_parent(new_child_parent)

    save_parent_store()
    save_child_parent()
    logger.info(f"RAG: 父块映射 {len(new_parents)} 条, 子→父映射 {len(new_child_parent)} 条")


def _cleanup_mappings_for_source(source: str) -> None:
    """从 parent_store 和 child_parent 中移除指定 source 的条目。"""
    child_parent = get_child_to_parent()
    parent_store = get_parent_store()

    referenced_parents = set(child_parent.values())
    leftover = {k: v for k, v in parent_store.items() if k in referenced_parents}
    set_parent_store(leftover)
    save_parent_store()

    valid_parent_ids = set(leftover.keys())
    cleaned = {k: v for k, v in child_parent.items() if v in valid_parent_ids}
    set_child_to_parent(cleaned)
    save_child_parent()


def _rebuild_bm25(texts: list[str]) -> None:
    """重建 BM25 索引（全量）。"""
    set_bm25_texts(texts)
    set_bm25_index(build_bm25_index(texts))
    save_bm25()
    logger.info(f"RAG: BM25 索引重建完成，共 {len(texts)} 篇")


# ── 检索 ──

def _rrf_fusion(
    dense_ranked: list[tuple[int, str]],
    sparse_ranked: list[tuple[int, str]],
    top_k: int,
) -> list[str]:
    """RRF 融合稠密和稀疏结果。score = Σ 1/(k + rank)"""
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
    混合检索：子块匹配 → 父块展开 → RRF 融合 → DashScope Rerank 精排。
    双向降级：BM25 不可用 → 纯稠密; Rerank 失败 → RRF 结果。
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    # 懒加载
    if get_bm25_index() is None and collection.count() > 0:
        if not load_bm25():
            all_docs = collection.get()
            texts = all_docs.get("documents", [])
            if texts:
                set_bm25_texts(texts)
                set_bm25_index(build_bm25_index(texts))
                save_bm25()

    if not get_parent_store() and collection.count() > 0:
        load_parent_store()
    if not get_child_to_parent() and collection.count() > 0:
        load_child_parent()

    recall_k = top_k * RECALL_MULTIPLIER

    # 稠密检索
    query_vec = await embed_query(query)
    dense_results = collection.query(query_embeddings=[query_vec], n_results=recall_k)
    dense_docs = dense_results.get("documents", [[]])[0]
    dense_ranked = [(i, d) for i, d in enumerate(dense_docs) if d]

    # BM25 稀疏检索
    sparse_ranked: list[tuple[int, str]] = []
    bm25_index = get_bm25_index()
    bm25_texts = get_bm25_texts()
    if bm25_index is not None and bm25_texts:
        try:
            tokenized_query = list(jieba.cut(query))
            bm25_scores = bm25_index.get_scores(tokenized_query)
            indexed = sorted(enumerate(bm25_scores), key=lambda x: -x[1])
            for idx, score in indexed[:recall_k]:
                if score > 0 and idx < len(bm25_texts):
                    sparse_ranked.append((idx, bm25_texts[idx]))
        except Exception as e:
            logger.warning(f"BM25 检索异常，降级为纯稠密检索: {e}")

    # RRF 融合
    if sparse_ranked:
        merged = _rrf_fusion(dense_ranked, sparse_ranked, recall_k)
        logger.debug(f"RRF 融合: 稠密 {len(dense_ranked)} + 稀疏 {len(sparse_ranked)} → {len(merged)}")
    else:
        merged = [d for _, d in dense_ranked]

    # Rerank 精排
    if merged:
        rerank_results = await dashscope_rerank(query, merged, top_n=top_k)
        if rerank_results:
            reranked = [merged[idx] for idx, _ in rerank_results if idx < len(merged)]
            logger.debug(f"Rerank: {len(merged)} → {len(reranked)}")
            merged = reranked[:top_k]
        else:
            logger.debug("Rerank 失败，降级为 RRF 融合结果")
    merged = merged[:top_k]

    # 子块 → 父块展开（阈值条件：同一父块下 ≥ AUTO_MERGE_THRESHOLD 个子块命中才展开）
    parent_store = get_parent_store()
    child_to_parent = get_child_to_parent()
    expanded = []
    seen = set()
    if parent_store and child_to_parent:
        # 按父块分组统计命中数
        parent_hits: dict[str, list[str]] = {}
        unmatched = []
        for text in merged:
            parent_id = child_to_parent.get(text)
            if parent_id and parent_id in parent_store:
                parent_hits.setdefault(parent_id, []).append(text)
            else:
                unmatched.append(text)

        # 阈值判断：≥阈值 → 展开父块；<阈值 → 保留子块
        for parent_id, children in parent_hits.items():
            if len(children) >= AUTO_MERGE_THRESHOLD:
                parent_text = parent_store[parent_id]
                if parent_text not in seen:
                    seen.add(parent_text)
                    expanded.append(parent_text)
                logger.debug(f"Auto-merge: 父块 {parent_id} 命中 {len(children)} 子块 → 展开")
            else:
                for child_text in children:
                    if child_text not in seen:
                        seen.add(child_text)
                        expanded.append(child_text)

        for text in unmatched:
            if text not in seen:
                seen.add(text)
                expanded.append(text)
    else:
        for t in merged:
            if t not in seen:
                seen.add(t)
                expanded.append(t)

    return expanded
