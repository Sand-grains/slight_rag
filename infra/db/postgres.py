"""PostgreSQL 客户端：chunk 存储 + status 状态机 + 回滚清理。"""
import psycopg2
import psycopg2.extras
from config import POSTGRES_CONNECTION_URI, PG_PENDING_CLEANUP_MINUTES


class PgSQLClient:
    def __init__(self, conn_url=POSTGRES_CONNECTION_URI):
        self._url = conn_url
        self._conn = psycopg2.connect(conn_url)
        self._conn.autocommit = True

    def ensure_table(self):
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id   TEXT PRIMARY KEY,
                doc_id     TEXT NOT NULL,
                content    TEXT NOT NULL,
                metadata   JSONB DEFAULT '{}',
                status     TEXT DEFAULT 'suspending',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON chunks(doc_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_status ON chunks(status)")
        cur.close()

    def drop_table(self):
        cur = self._conn.cursor()
        cur.execute("DROP TABLE IF EXISTS chunks")
        cur.close()

    def insert_batch(self, chunks):
        rows = [
            (c.chunk_id, c.doc_id, c.content, "{}", "suspending")
            for c in chunks
        ]
        cur = self._conn.cursor()
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO chunks (chunk_id, doc_id, content, metadata, status) VALUES %s",
            rows,
        )
        cur.close()
        return len(rows)

    def update_status(self, chunk_ids, status="indexed"):
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE chunks SET status = %s WHERE chunk_id = ANY(%s::text[])",
            (status, list(chunk_ids)),
        )
        cur.close()

    def get_by_ids(self, chunk_ids):
        """按 chunk_id 批量查询，返回原始行（Phase 2 由 IndexStore 组装为 Chunk）。"""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT chunk_id, doc_id, content, metadata FROM chunks WHERE chunk_id = ANY(%s)",
            (chunk_ids,),
        )
        rows = cur.fetchall()
        cur.close()
        return rows  # [(chunk_id, doc_id, content, metadata_dict), ...]

    def cleanup_suspending(self, ttl_minutes=PG_PENDING_CLEANUP_MINUTES):
        cur = self._conn.cursor()
        cur.execute(
            "SELECT chunk_id, doc_id FROM chunks WHERE status = 'suspending'"
            " AND created_at < NOW() - INTERVAL '%s minutes'",
            (ttl_minutes,),
        )
        rows = cur.fetchall()
        cur.close()
        return rows  # [(chunk_id, doc_id), ...]

    def delete_by_doc_id(self, doc_id):
        cur = self._conn.cursor()
        cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
        cur.close()

    def delete_by_chunk_ids(self, chunk_ids):
        cur = self._conn.cursor()
        cur.execute("DELETE FROM chunks WHERE chunk_id = ANY(%s)", (chunk_ids,))
        cur.close()
