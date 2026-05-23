import hashlib
from pathlib import Path

import chromadb
from loguru import logger
from settings import settings
from core.rag.embedder import embed_documents, embed_query
from core.rag.loader import load_and_chunk

COLLECTION_NAME = "hr_knowledge_base"
FINGERPRINT_KEY = "data_fingerprint"


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
        # 相对路径参与哈希，删除/重命名也能检测到
        hasher.update(str(file_path.relative_to(data_path)).encode("utf-8"))
        hasher.update(b"\x00")  # 分隔符
        hasher.update(file_path.read_bytes())
    return hasher.hexdigest()


def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)


def _get_collection() -> chromadb.Collection:
    client = _get_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


async def ingest(force: bool = False) -> int:
    """
    将 data/ 目录下的文档向量化后存入 ChromaDB。
    - 对比文件指纹：指纹不变则跳过，指纹变化则自动重建
    - force=True 强制清空重建，不对比指纹
    返回本次入库的 chunk 数量（0 表示跳过）。
    """
    # 计算当前文件指纹
    current_fingerprint = _compute_files_fingerprint(settings.DATA_DIR)
    if not current_fingerprint:
        logger.info("RAG: data/ 目录为空，跳过入库")
        return 0

    collection = _get_collection()

    # 对比指纹：未变且非强制 → 跳过
    if not force:
        stored = collection.metadata
        if stored and stored.get(FINGERPRINT_KEY) == current_fingerprint:
            logger.info(f"RAG: 知识库未变更，跳过入库 (指纹: {current_fingerprint[:8]}...)")
            return 0

    # 指纹变化或强制重建 → 清空旧数据
    if collection.count() > 0:
        logger.info("RAG: 检测到知识库变更，清空重建...")
        client = _get_client()
        client.delete_collection(COLLECTION_NAME)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # 加载、分块、向量化、写入
    chunks = load_and_chunk(settings.DATA_DIR)
    if not chunks:
        logger.info("RAG: data/ 目录为空，跳过入库")
        return 0

    texts = [c["content"] for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": c["source"], "title": c["title"], "chunk_index": c["chunk_index"]}
        for c in chunks
    ]

    batch_size = 20
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

    # 将当前指纹存入 collection metadata
    client = _get_client()
    collection.modify(metadata={FINGERPRINT_KEY: current_fingerprint})

    logger.info(f"RAG: 知识库入库完成，共 {total} 个 chunk (指纹: {current_fingerprint[:8]}...)")
    return total


async def retrieve(query: str, top_k: int = 5) -> list[str]:
    """
    语义检索：输入查询文本，返回最相关的 top_k 个文档片段。
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    query_vec = await embed_query(query)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
    )
    docs = results.get("documents", [[]])
    return [d for d in docs[0] if d]
