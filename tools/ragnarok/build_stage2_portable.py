#!/usr/bin/env python3
"""Merge the verified Supabase Ragnarok data into the portable DuckDB bundle.

The script intentionally exports only public game/reference data. User profiles,
credentials, comments, seller names, and private application tables are excluded
at the export endpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import zstandard as zstd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_table_name(dataset: str) -> str:
    return "sb_" + re.sub(r"[^a-z0-9_]+", "_", dataset.lower()).strip("_")


def json_default(value: Any) -> str:
    return str(value)


def flatten_for_parquet(row: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            flat[key] = json.dumps(value, ensure_ascii=False, sort_keys=True, default=json_default)
        elif value is None or isinstance(value, (str, int, float, bool)):
            flat[key] = value
        else:
            flat[key] = str(value)
    return flat


def build_session() -> requests.Session:
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        status=8,
        backoff_factor=1.2,
        status_forcelist=(408, 425, 429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "Ragnarok-AI-Portable-Exporter/2.0"})
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8))
    return session


def get_json(session: requests.Session, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    response = session.get(endpoint, params=params, timeout=(20, 120))
    response.raise_for_status()
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(f"non-JSON response from {response.url}: {response.text[:300]}") from exc
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"export endpoint error for {params}: {data}")
    if not isinstance(data, dict):
        raise RuntimeError(f"unexpected response type for {params}: {type(data)!r}")
    return data


def write_exact_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    compressor = zstd.ZstdCompressor(level=12, threads=-1)
    with path.open("wb") as raw:
        with compressor.stream_writer(raw, closefd=False) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", write_through=True) as text:
                for row in rows:
                    text.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=json_default))
                    text.write("\n")


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        table = pa.table({"__empty": pa.array([], type=pa.bool_())})
        pq.write_table(table, path, compression="zstd", compression_level=12)
        return
    flattened = [flatten_for_parquet(row) for row in rows]
    columns = sorted({key for row in flattened for key in row})
    normalized = [{column: row.get(column) for column in columns} for row in flattened]
    frame = pd.DataFrame(normalized, columns=columns)
    # Pandas object columns containing mixed numeric/string values are made
    # deterministic as UTF-8 strings instead of relying on Arrow coercion.
    for column in frame.columns:
        if frame[column].dtype != "object":
            continue
        non_null = frame[column].dropna()
        if non_null.empty:
            continue
        kinds = {type(value) for value in non_null.head(1000)}
        if not kinds.issubset({str}):
            frame[column] = frame[column].map(lambda value: None if value is None else str(value))
    table = pa.Table.from_pandas(frame, preserve_index=False, safe=False)
    pq.write_table(table, path, compression="zstd", compression_level=12, use_dictionary=True)


def fetch_dataset(
    session: requests.Session,
    endpoint: str,
    dataset: str,
    expected: int,
    max_limit: int,
    exact_dir: Path,
    parquet_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    limit = max(1, min(max_limit, 1000))
    offset = 0
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    while True:
        page = get_json(session, endpoint, {"dataset": dataset, "offset": offset, "limit": limit})
        page_rows = page.get("rows")
        if not isinstance(page_rows, list):
            raise RuntimeError(f"dataset {dataset} returned no rows list at offset {offset}")
        if int(page.get("returned", len(page_rows))) != len(page_rows):
            raise RuntimeError(f"dataset {dataset} returned inconsistent page count at offset {offset}")
        rows.extend(page_rows)
        offset += len(page_rows)
        print(f"[{dataset}] {len(rows)}/{expected}", flush=True)
        if not page.get("has_more") or not page_rows:
            break
        if len(rows) > expected + limit:
            raise RuntimeError(f"dataset {dataset} exceeded manifest count ({len(rows)} > {expected})")
    if len(rows) != expected:
        raise RuntimeError(f"dataset {dataset} count mismatch: expected {expected}, fetched {len(rows)}")

    exact_path = exact_dir / f"{dataset}.jsonl.zst"
    parquet_path = parquet_dir / f"{dataset}.parquet"
    write_exact_jsonl(exact_path, rows)
    write_parquet(parquet_path, rows)
    info = {
        "dataset": dataset,
        "rows": len(rows),
        "seconds": round(time.monotonic() - started, 3),
        "exact_file": exact_path.name,
        "exact_bytes": exact_path.stat().st_size,
        "exact_sha256": sha256_file(exact_path),
        "parquet_file": parquet_path.name,
        "parquet_bytes": parquet_path.stat().st_size,
        "parquet_sha256": sha256_file(parquet_path),
    }
    return rows, info


def locate_stage1(stage1_root: Path) -> Path:
    matches = sorted(stage1_root.rglob("ragnarok-ai.duckdb.zst"))
    if not matches:
        matches = sorted(stage1_root.rglob("*.duckdb.zst"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one stage-1 DuckDB archive, found {len(matches)}: {matches}")
    return matches[0]


def decompress_stage1(source: Path, destination: Path) -> None:
    decompressor = zstd.ZstdDecompressor()
    with source.open("rb") as src, destination.open("wb") as dst:
        decompressor.copy_stream(src, dst)


def qpath(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def json_key_expr(column: str, keys: list[str]) -> str:
    parts = [f"nullif(json_extract_string({column}, '$.{key}'), '')" for key in keys]
    return "coalesce(" + ",".join(parts) + ")"


def materialize_ai_tables(db_path: Path, parquet_dir: Path, manifest: dict[str, Any]) -> dict[str, int]:
    connection = duckdb.connect(str(db_path))
    try:
        for parquet_path in sorted(parquet_dir.glob("*.parquet")):
            dataset = parquet_path.stem
            table = safe_table_name(dataset)
            connection.execute(f'DROP TABLE IF EXISTS "{table}"')
            connection.execute(
                f'CREATE TABLE "{table}" AS SELECT * FROM read_parquet(\'{qpath(parquet_path)}\')'
            )

        connection.execute("DROP TABLE IF EXISTS portable_export_manifest")
        connection.execute("CREATE TABLE portable_export_manifest(exported_at TIMESTAMP, export_version INTEGER, manifest_json JSON)")
        generated_at = str(manifest.get("generated_at", "")).replace("'", "''")
        manifest_json = json.dumps(manifest, ensure_ascii=False, sort_keys=True).replace("'", "''")
        connection.execute(
            f"INSERT INTO portable_export_manifest VALUES (try_cast('{generated_at}' AS TIMESTAMP), "
            f"{int(manifest.get('export_version', 2))}, '{manifest_json}'::JSON)"
        )

        name_expr = json_key_expr(
            "payload",
            ["name", "Name", "displayName", "identifiedDisplayName", "rawName", "description", "Description", "databaseName", "aegisName"],
        )
        description_expr = json_key_expr("payload", ["description", "Description", "rawDescription", "identifiedDescriptionName"])
        database_expr = json_key_expr("payload", ["databaseName", "DatabaseName", "aegisName", "AegisName", "key"])
        connection.execute("DROP TABLE IF EXISTS ai_current_entities")
        connection.execute(
            f"""
            CREATE TABLE ai_current_entities AS
            SELECT
              entity_type,
              entity_id,
              {name_expr} AS name,
              {description_expr} AS description,
              {database_expr} AS database_name,
              source_name,
              source_file,
              source_sha256,
              observed_at,
              payload AS payload_json
            FROM sb_current_entities
            """
        )

        connection.execute("DROP TABLE IF EXISTS ai_alias_aggregate")
        alias_columns = {row[0] for row in connection.execute("DESCRIBE sb_entity_aliases").fetchall()}
        if {"entity_type", "entity_id", "alias"}.issubset(alias_columns):
            connection.execute(
                """
                CREATE TABLE ai_alias_aggregate AS
                SELECT entity_type, entity_id,
                       string_agg(DISTINCT cast(alias AS VARCHAR), ' ' ORDER BY cast(alias AS VARCHAR)) AS aliases
                FROM sb_entity_aliases
                GROUP BY entity_type, entity_id
                """
            )
        else:
            connection.execute("CREATE TABLE ai_alias_aggregate(entity_type VARCHAR, entity_id VARCHAR, aliases VARCHAR)")

        connection.execute("DROP TABLE IF EXISTS ai_entity_search")
        connection.execute(
            """
            CREATE TABLE ai_entity_search AS
            SELECT e.entity_type,e.entity_id,e.name,e.description,e.database_name,e.source_name,e.observed_at,
                   coalesce(a.aliases,'') AS aliases,
                   lower(concat_ws(' ',e.entity_type,e.entity_id,e.name,e.database_name,e.description,a.aliases)) AS search_text,
                   e.payload_json
            FROM ai_current_entities e
            LEFT JOIN ai_alias_aggregate a USING(entity_type,entity_id)
            """
        )

        connection.execute("CREATE OR REPLACE VIEW ai_relations AS SELECT * FROM sb_entity_links")
        connection.execute("CREATE OR REPLACE VIEW ai_divine_pride AS SELECT * FROM sb_dp_entities")
        connection.execute("CREATE OR REPLACE VIEW ai_market AS SELECT * FROM sb_market")
        connection.execute("CREATE OR REPLACE VIEW ai_official_updates AS SELECT * FROM sb_official_updates")
        connection.execute("CREATE OR REPLACE VIEW ai_damage_rules AS SELECT * FROM sb_damage_rules")
        connection.execute("CREATE OR REPLACE VIEW ai_damage_items AS SELECT * FROM sb_damage_items")
        connection.execute("CREATE OR REPLACE VIEW ai_damage_monsters AS SELECT * FROM sb_damage_monsters")
        connection.execute("CREATE OR REPLACE VIEW ai_damage_skills AS SELECT * FROM sb_damage_skills")
        connection.execute("CREATE OR REPLACE VIEW ai_damage_effects AS SELECT * FROM sb_damage_effects")
        connection.execute("CREATE OR REPLACE VIEW ai_virtual_entities AS SELECT * FROM sb_virtual_entities")

        connection.execute("DROP TABLE IF EXISTS ai_search_index")
        connection.execute(
            """
            CREATE TABLE ai_search_index AS
            SELECT 'entity' AS record_kind, entity_type AS category, entity_id AS record_id,
                   name AS title, search_text, source_name, payload_json AS evidence_json
            FROM ai_entity_search
            UNION ALL
            SELECT 'official_update', coalesce(entry_type,'update'), coalesce(external_id,cast(id AS VARCHAR)),
                   title, lower(concat_ws(' ',title,body)), source_key, to_json(sb_official_updates)::VARCHAR
            FROM sb_official_updates
            UNION ALL
            SELECT 'market', coalesce(listing_type,'listing'), coalesce(external_id,cast(id AS VARCHAR)),
                   item_name, lower(concat_ws(' ',item_id,item_name,server_code,grade,refine)), source_key,
                   to_json(sb_market)::VARCHAR
            FROM sb_market
            """
        )

        connection.execute("CHECKPOINT")
        connection.execute("VACUUM")

        counts: dict[str, int] = {}
        for table in [
            "ai_current_entities",
            "ai_entity_search",
            "ai_search_index",
            "sb_dp_entities",
            "sb_entity_links",
            "sb_market",
            "sb_official_updates",
        ]:
            counts[table] = int(connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])
        return counts
    finally:
        connection.close()


def zip_directory(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source))


def compress_database(source: Path, destination: Path) -> None:
    compressor = zstd.ZstdCompressor(level=17, threads=-1)
    with source.open("rb") as src, destination.open("wb") as dst:
        compressor.copy_stream(src, dst)


def write_checksums(paths: list[Path], destination: Path) -> None:
    lines = [f"{sha256_file(path)}  {path.name}" for path in sorted(paths, key=lambda p: p.name)]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output: Path = args.output
    if output.exists():
        shutil.rmtree(output)
    exact_dir = output / "exact"
    parquet_dir = output / "parquet"
    bundle_dir = output / "bundle"
    for directory in (exact_dir, parquet_dir, bundle_dir):
        directory.mkdir(parents=True, exist_ok=True)

    session = build_session()
    manifest = get_json(session, args.endpoint, {"dataset": "manifest"})
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise RuntimeError("export manifest has no datasets")
    errors = {name: spec.get("error") for name, spec in datasets.items() if spec.get("error")}
    if errors:
        raise RuntimeError(f"export manifest contains dataset errors: {errors}")

    dataset_infos: list[dict[str, Any]] = []
    fetched_rows: dict[str, list[dict[str, Any]]] = {}
    for dataset, spec in datasets.items():
        expected = int(spec.get("count") or 0)
        max_limit = int(spec.get("max_limit") or 250)
        rows, info = fetch_dataset(session, args.endpoint, dataset, expected, max_limit, exact_dir, parquet_dir)
        fetched_rows[dataset] = rows
        dataset_infos.append(info)

    stage1_archive = locate_stage1(args.stage1_root)
    db_path = bundle_dir / "ragnarok-ai-full.duckdb"
    decompress_stage1(stage1_archive, db_path)
    stage1_hash = sha256_file(db_path)
    ai_counts = materialize_ai_tables(db_path, parquet_dir, manifest)

    expected_current = int(datasets["current_entities"]["count"])
    expected_links = int(datasets["entity_links"]["count"])
    if ai_counts["ai_current_entities"] != expected_current:
        raise RuntimeError("AI current entity count failed validation")
    if ai_counts["sb_entity_links"] != expected_links:
        raise RuntimeError("AI relation count failed validation")

    exact_zip = bundle_dir / "05_supabase_exact_jsonl.zip"
    parquet_zip = bundle_dir / "06_supabase_direct_parquet.zip"
    zip_directory(exact_dir, exact_zip)
    zip_directory(parquet_dir, parquet_zip)

    compressed_db = bundle_dir / "ragnarok-ai-full.duckdb.zst"
    compress_database(db_path, compressed_db)
    db_hash = sha256_file(db_path)
    compressed_hash = sha256_file(compressed_db)

    stage2_manifest = {
        "format": "Ragnarok AI portable database",
        "stage": 2,
        "exported_at": manifest.get("generated_at"),
        "export_endpoint_version": manifest.get("export_version"),
        "scope": "stage-1 rAthena/OpenKore plus Supabase current, Divine Pride, aliases, relations, GNJOY market/updates, and damage tables",
        "privacy": manifest.get("privacy"),
        "stage1_archive": stage1_archive.name,
        "stage1_uncompressed_sha256": stage1_hash,
        "datasets": dataset_infos,
        "ai_counts": ai_counts,
        "files": {
            exact_zip.name: {"bytes": exact_zip.stat().st_size, "sha256": sha256_file(exact_zip)},
            parquet_zip.name: {"bytes": parquet_zip.stat().st_size, "sha256": sha256_file(parquet_zip)},
            compressed_db.name: {"bytes": compressed_db.stat().st_size, "sha256": compressed_hash},
            db_path.name: {"bytes": db_path.stat().st_size, "sha256": db_hash, "uploaded": False},
        },
    }
    manifest_path = bundle_dir / "stage2_manifest.json"
    manifest_path.write_text(json.dumps(stage2_manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    readme_path = bundle_dir / "README_STAGE2_KO.md"
    readme_path.write_text(
        "# Ragnarok AI DB — 2차 통합본\n\n"
        "이 묶음은 1차 rAthena/OpenKore 원천 DB에 Supabase의 현재 정본, Divine Pride kROM 상세, "
        "한국명 별칭, 전체 관계, GNJOY 공식 시세·업데이트, 데미지 관련 표를 합친 휴대용 DB다.\n\n"
        f"- 현재 엔티티: {ai_counts['ai_current_entities']:,}건\n"
        f"- 관계: {ai_counts['sb_entity_links']:,}건\n"
        f"- Divine Pride 상세: {ai_counts['sb_dp_entities']:,}건\n"
        f"- 공식 시세 행: {ai_counts['sb_market']:,}건\n"
        f"- 공식 업데이트 행: {ai_counts['sb_official_updates']:,}건\n"
        f"- AI 통합 검색행: {ai_counts['ai_search_index']:,}건\n\n"
        "`ragnarok-ai-full.duckdb.zst`를 압축 해제하면 DuckDB에서 바로 검색할 수 있다. "
        "원문 중첩값은 JSON 문자열로 보존했으며, `05_supabase_exact_jsonl.zip`에는 API 응답을 줄 단위 JSON으로 그대로 보존했다.\n",
        encoding="utf-8",
    )

    checksum_path = bundle_dir / "SHA256SUMS_STAGE2.txt"
    checksum_targets = [exact_zip, parquet_zip, compressed_db, manifest_path, readme_path]
    write_checksums(checksum_targets, checksum_path)

    # Do not upload the large uncompressed DB as an artifact.
    db_path.unlink()
    summary = {
        "status": "ok",
        "current_entities": ai_counts["ai_current_entities"],
        "relations": ai_counts["sb_entity_links"],
        "divine_pride": ai_counts["sb_dp_entities"],
        "market": ai_counts["sb_market"],
        "official_updates": ai_counts["sb_official_updates"],
        "ai_search_rows": ai_counts["ai_search_index"],
        "compressed_db_bytes": compressed_db.stat().st_size,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
