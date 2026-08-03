"""chunks 的摄入编排：将父块 + 子块向量写入三库。
三库写入顺序: PgSQL 父块(suspending) → ES 父块 → Milvus 子块 → PgSQL 父块标 indexed
  suspending 状态机回滚

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
    - rollback_doc: 删除三端指定 doc_id 数据
    - cleanup_suspending: 扫描并回滚超时 suspending 记录
"""
from config import PG_PENDING_CLEANUP_MINUTES


def rollback_doc(doc_id, pgsql, es, milvus):
    """删除三端中指定 doc_id 的全部数据。"""
    es.delete_by_doc_id(doc_id)
    milvus.delete_by_doc_id(doc_id)
    pgsql.delete_by_doc_id(doc_id)


def ingest_doc(parents, children, child_vectors, pgsql, es, milvus):
    """单文档摄入：父块写入 PgSQL/ES，子块 + 向量写入 Milvus。写入前先清理旧数据保证幂等。"""
    if not parents:
        return  # 空文档分块为零，直接跳过（external 模式 parent_chunks[0] 会 IndexError）
    doc_id = parents[0].doc_id
    rollback_doc(doc_id, pgsql, es, milvus)

    pgsql.insert_batch(parents)  # 父块, status = 'suspending'
    es.insert_batch(parents)
    milvus.insert_batch(children, child_vectors)
    pgsql.update_status([c.chunk_id for c in parents], "indexed")


def cleanup_suspending(pgsql, es, milvus, ttl_minutes=PG_PENDING_CLEANUP_MINUTES):
    """扫描并回滚超时未完成的 suspending 记录。"""
    rows = pgsql.cleanup_suspending(ttl_minutes)
    for chunk_id, doc_id in rows:
        rollback_doc(doc_id, pgsql, es, milvus)
    return len(rows)
