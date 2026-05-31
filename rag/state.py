"""RAG 持久化状态管理：文件级指纹、ChromaDB 连接、索引与映射的 save/load/remove。"""
import hashlib
import os
import pickle
from pathlib import Path

import chromadb
import jieba
from loguru import logger
from rank_bm25 import BM25Okapi
from settings import settings


# ── 常量 ──

COLLECTION_NAME = "hr_knowledge_base"
FILE_HASHES_FILE = "file_hashes.pkl"
BM25_STATE_FILE = "bm25_state.pkl"
PARENT_STORE_FILE = "parent_store.pkl"
CHILD_PARENT_FILE = "child_parent_map.pkl"

RRF_K = 60
RECALL_MULTIPLIER = 4


# ── 运行时全局状态 ──

_bm25_index: BM25Okapi | None = None
_bm25_texts: list[str] = []
_parent_store: dict[str, str] = {}
_child_to_parent: dict[str, str] = {}


# ── 文件级哈希 ──

def compute_file_hashes(data_dir: str) -> dict[str, str]:
    """计算 data/ 目录下每个 .md/.txt 的 SHA256，返回 {相对路径: hash}。"""
    data_path = Path(data_dir)
    hashes: dict[str, str] = {}
    if not data_path.exists():
        return hashes
    for file_path in sorted(data_path.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in {".md", ".txt"}:
            continue
        rel = str(file_path.relative_to(data_path))
        hashes[rel] = hashlib.sha256(file_path.read_bytes()).hexdigest()
    return hashes


def save_file_hashes(hashes: dict[str, str]) -> None:
    try:
        with open(_path(FILE_HASHES_FILE), "wb") as f:
            pickle.dump(hashes, f)
        logger.debug(f"文件哈希已保存: {len(hashes)} 个文件")
    except Exception as e:
        logger.warning(f"文件哈希保存失败: {e}")


def load_file_hashes() -> dict[str, str]:
    path = _path(FILE_HASHES_FILE)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            hashes = pickle.load(f)
        logger.info(f"RAG: 文件哈希已加载，共 {len(hashes)} 个文件")
        return hashes
    except Exception as e:
        logger.warning(f"文件哈希加载失败: {e}")
        return {}


def remove_file_hashes() -> None:
    path = _path(FILE_HASHES_FILE)
    if os.path.exists(path):
        os.remove(path)


# ── ChromaDB 按源删除 ──

def remove_chunks_by_source(source: str) -> int:
    """从 ChromaDB 中删除指定 source 的所有子块，返回删除数。"""
    collection = get_collection()
    try:
        result = collection.get(where={"source": source})
        ids = result.get("ids", [])
        if ids:
            collection.delete(ids=ids)
            logger.info(f"RAG: 已从索引中删除 source='{source}' 的 {len(ids)} 个子块")
        return len(ids)
    except Exception as e:
        logger.warning(f"按 source 删除子块失败: {e}")
        return 0

def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)


def get_collection(name: str = COLLECTION_NAME) -> chromadb.Collection:
    return get_chroma_client().get_or_create_collection(name=name)


def delete_collection(name: str = COLLECTION_NAME) -> None:
    client = get_chroma_client()
    try:
        client.delete_collection(name)
    except Exception:
        pass


# ── 路径工具 ──

def _path(filename: str) -> str:
    return os.path.join(settings.CHROMA_DB_PATH, filename)


# ── BM25 ──

def get_bm25_index() -> BM25Okapi | None:
    return _bm25_index


def get_bm25_texts() -> list[str]:
    return _bm25_texts


def set_bm25_index(index: BM25Okapi | None) -> None:
    global _bm25_index
    _bm25_index = index


def set_bm25_texts(texts: list[str]) -> None:
    global _bm25_texts
    _bm25_texts = texts


def build_bm25_index(texts: list[str]) -> BM25Okapi:
    tokenized = [list(jieba.cut(t)) for t in texts]
    return BM25Okapi(tokenized)


def save_bm25() -> None:
    global _bm25_index, _bm25_texts
    if _bm25_index is None:
        return
    try:
        with open(_path(BM25_STATE_FILE), "wb") as f:
            pickle.dump({"index": _bm25_index, "texts": _bm25_texts}, f)
        logger.debug(f"BM25 索引已保存，共 {len(_bm25_texts)} 篇")
    except Exception as e:
        logger.warning(f"BM25 索引保存失败: {e}")


def load_bm25() -> bool:
    global _bm25_index, _bm25_texts
    path = _path(BM25_STATE_FILE)
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
        logger.warning(f"BM25 索引加载失败: {e}")
        return False


def remove_bm25() -> None:
    path = _path(BM25_STATE_FILE)
    if os.path.exists(path):
        os.remove(path)


# ── 父块映射 ──

def get_parent_store() -> dict[str, str]:
    return _parent_store


def set_parent_store(store: dict[str, str]) -> None:
    global _parent_store
    _parent_store = store


def save_parent_store() -> None:
    global _parent_store
    if not _parent_store:
        return
    try:
        with open(_path(PARENT_STORE_FILE), "wb") as f:
            pickle.dump(_parent_store, f)
        logger.debug(f"父块映射已保存: {len(_parent_store)} 条")
    except Exception as e:
        logger.warning(f"父块映射保存失败: {e}")


def load_parent_store() -> bool:
    global _parent_store
    path = _path(PARENT_STORE_FILE)
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:
            _parent_store = pickle.load(f)
        logger.info(f"RAG: 父块映射已加载，共 {len(_parent_store)} 条")
        return True
    except Exception as e:
        logger.warning(f"父块映射加载失败: {e}")
        return False


def remove_parent_store() -> None:
    path = _path(PARENT_STORE_FILE)
    if os.path.exists(path):
        os.remove(path)


# ── 子块→父块映射 ──

def get_child_to_parent() -> dict[str, str]:
    return _child_to_parent


def set_child_to_parent(mapping: dict[str, str]) -> None:
    global _child_to_parent
    _child_to_parent = mapping


def save_child_parent() -> None:
    global _child_to_parent
    if not _child_to_parent:
        return
    try:
        with open(_path(CHILD_PARENT_FILE), "wb") as f:
            pickle.dump(_child_to_parent, f)
        logger.debug(f"子块→父块映射已保存: {len(_child_to_parent)} 条")
    except Exception as e:
        logger.warning(f"子块→父块映射保存失败: {e}")


def load_child_parent() -> bool:
    global _child_to_parent
    path = _path(CHILD_PARENT_FILE)
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:
            _child_to_parent = pickle.load(f)
        logger.info(f"RAG: 子块→父块映射已加载，共 {len(_child_to_parent)} 条")
        return True
    except Exception as e:
        logger.warning(f"子块→父块映射加载失败: {e}")
        return False


def remove_child_parent() -> None:
    path = _path(CHILD_PARENT_FILE)
    if os.path.exists(path):
        os.remove(path)


# ── 清空所有状态 ──

def reset_state() -> None:
    """清空所有内存状态并删除磁盘文件。"""
    global _bm25_index, _bm25_texts, _parent_store, _child_to_parent
    _bm25_index = None
    _bm25_texts = []
    _parent_store = {}
    _child_to_parent = {}
    remove_bm25()
    remove_parent_store()
    remove_child_parent()
    remove_file_hashes()
