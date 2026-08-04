"""external 模式下 chunks 的摄入编排: 将父块 + 子块向量写入三库

三库写入顺序：PgSQL 父块（suspending）→ ES 父块 → Milvus 子块 → PgSQL 父块标 indexed，
配合 suspending 状态机回滚保证最终一致。

核心特性:
    - ingest_doc 按固定顺序写三库（PgSQL→ES→Milvus），PgSQL 先落库（status='suspending'）作为事实源
    - 写入前先 rollback_doc 清理旧数据，保证幂等（重复摄入同一 doc_id 不产生脏数据）
    - 中间任一库失败 → PgSQL 残留 suspending 行，由 cleanup_suspending(ttl) 定时回滚，最终一致
    - IndexStore 外部模式启动时自动调用一次 cleanup_suspending() 收敛历史残留

用法示例::

    ingest_doc(parents, children, child_vectors, pgsql, es, milvus)  # 单文档摄入
    rollback_doc(doc_id, pgsql, es, milvus)                          # 清理三端指定文档
    cleanup_suspending(pgsql, es, milvus, ttl_minutes=30)            # 回滚超时 suspending

公共接口：
    - ingest_doc: 单文档摄入（幂等）
    - rollback_doc: 回滚删除三端指定 doc_id 数据
    - cleanup_suspending: 扫描并回滚超时 suspending 记录
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from config import PG_PENDING_CLEANUP_MINUTES

if TYPE_CHECKING:
    from indexing.chunk import Chunk
    from infra.db.postgres import PgSQLClient
    from infra.search.es import ESClient
    from infra.vector.milvus import MilvusClientWrapper


def rollback_doc(doc_id: str, pgsql: PgSQLClient, es: ESClient, milvus: MilvusClientWrapper) -> None:
    """删除三端中指定 doc_id 的全部数据。

    Args:
        doc_id: 待清理的文档 ID。
        pgsql: PostgreSQL 客户端（父块元数据表）。
        es: Elasticsearch 客户端（全文索引）。
        milvus: Milvus 客户端（稠密向量索引）。
    """
    # 删除顺序: milvus -> es -> pgsql
    milvus.delete_by_doc_id(doc_id)
    es.delete_by_doc_id(doc_id)
    pgsql.delete_by_doc_id(doc_id)


def ingest_doc(parents: list[Chunk], children: list[Chunk], child_vectors: list[list[float]],
               pgsql: PgSQLClient, es: ESClient, milvus: MilvusClientWrapper) -> None:
    """单文档摄入: 父块写入 PgSQL/ES，子块 + 向量写入 Milvus，写入前先清理旧数据保证幂等。

    Args:
        parents: 父块列表（检索/benchmark 单元），写入 PgSQL 与 ES。
        children: 子块列表（稠密检索单元）。
        child_vectors: 与 children 一一对应的稠密向量。与children子块列表一并写入Milvus。
        pgsql: PostgreSQL 客户端。
        es: Elasticsearch 客户端。
        milvus: Milvus 客户端。
    """
    if not parents:
        return  # 空文档分块为零，直接跳过（external 模式 parent_chunks[0] 会 IndexError）
    doc_id = parents[0].doc_id
    rollback_doc(doc_id, pgsql, es, milvus) # 每次写入前都清理旧数据(保证幂等: 防止主键冲突或残留旧版本)

    # 写入顺序 pgsql -> es -> milvus
    pgsql.insert_batch(parents)  # 父块, status = 'suspending'
    es.insert_batch(parents)
    milvus.insert_batch(children, child_vectors)
    pgsql.update_status([chunk.chunk_id for chunk in parents], "indexed")


def cleanup_suspending(pgsql: PgSQLClient, es: ESClient, milvus: MilvusClientWrapper,
                       ttl_minutes: int = PG_PENDING_CLEANUP_MINUTES) -> int:
    """扫描并回滚超时未完成的 suspending 记录，返回本次清理的文档数。

    Args:
        pgsql: PostgreSQL 客户端。
        es: Elasticsearch 客户端。
        milvus: Milvus 客户端。
        ttl_minutes: 超时阈值（分钟），超过该时长仍为 suspending 的记录视为残留。

    Returns:
        int：本次回滚的文档数量（= pgsql.cleanup_suspending 返回的行数）。
    """
    rows = pgsql.cleanup_suspending(ttl_minutes)
    for chunk_id, doc_id in rows:
        rollback_doc(doc_id, pgsql, es, milvus)
    return len(rows)
