"""服务连通性检查: PgSQL + ES + Milvus + Redis。  全部通过返回 0，任一失败返回 1。"""
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()


def _check_pgsql():
    import psycopg2
    url = os.getenv("POSTGRES_CONNECTION_URI", "postgresql://postgres@localhost:5432/postgres")
    t0 = time.monotonic()
    conn = psycopg2.connect(url)
    cur = conn.cursor()
    cur.execute("SELECT 1")
    cur.close()
    conn.close()
    elapsed = (time.monotonic() - t0) * 1000
    return True, f"[OK] PostgreSQL  {url.split('@')[1] if '@' in url else url}  ({elapsed:.0f}ms)"


def _check_es():
    from elasticsearch import Elasticsearch
    url = os.getenv("ES_CONNECTION_URI", "http://localhost:9200")
    t0 = time.monotonic()
    es = Elasticsearch(url)
    info = es.info()
    elapsed = (time.monotonic() - t0) * 1000
    version = info["version"]["number"]
    return True, f"[OK] Elasticsearch  {url}  v{version}  ({elapsed:.0f}ms)"


def _check_milvus():
    from pymilvus import MilvusClient
    url = os.getenv("MILVUS_CONNECTION_URI", "http://localhost:19530")
    t0 = time.monotonic()
    client = MilvusClient(uri=url)
    version = client.get_server_version()
    elapsed = (time.monotonic() - t0) * 1000
    return True, f"[OK] Milvus  {url}  v{version}  ({elapsed:.0f}ms)"


def _check_redis():
    import redis
    url = os.getenv("REDIS_CONNECTION_URL", "redis://localhost:6379/0")
    t0 = time.monotonic()
    r = redis.from_url(url)
    r.ping()
    r.close()
    elapsed = (time.monotonic() - t0) * 1000
    return True, f"[OK] Redis  {url.split('@')[1] if '@' in url else url}  ({elapsed:.0f}ms)"


_CHECKS = [
    ("PgSQL", _check_pgsql),
    ("Elasticsearch", _check_es),
    ("Milvus", _check_milvus),
    ("Redis", _check_redis),
]


def main():
    failed = 0
    for name, check in _CHECKS:
        try:
            ok, msg = check()
            print(msg)
            if not ok:
                failed += 1
        except Exception as e:
            print(f"[FAIL] {name}  —  {e}")
            failed += 1

    print()
    if failed == 0:
        print("All 4 services healthy.")
    else:
        print(f"{failed}/4 service(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
