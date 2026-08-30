# AI Revenue Factory G0 — Deployment Status

Observed at: 2026-08-30 UTC

## Current verdict

`LOCAL_SOURCE_PASS / REAL_SERVER_AND_GX10_DEPLOYMENT_BLOCKED_BY_NETWORK_ACCESS`

## Verified in a fresh Linux runtime

- Source bundle: `ai_revenue_factory_g0_v1.zip`
- ZIP SHA-256: `2e39ae52c4db2eb5ec2e3a210c7f38132d77d706f638624be71a962ea9bbf796`
- Archive entries: 106
- Test command: `PYTHONPATH=. pytest -q`
- Result: `33 passed`
- Corrupted temporary transfer fragment was removed from this branch.

## User-owned artifact copy

The verified ZIP is stored in the user's Google Drive as file ID:

`10N4eKRwMVEpVdiW8uQtGQXOf34Aoa3Sk`

## What is implemented

```text
control app
PostgreSQL + pgvector
Redis
Prefect server/services/worker
read-only MCP gateway
Caddy reverse proxy
GX10 outbound worker
immutable job input SHA-256
AgentRun output receipt
Tool Registry default deny
```

Read-only MCP tools:

```text
get_dashboard_overview
get_content_case
get_evidence_packet
```

## What was not claimed as completed

```text
personal-server Docker runtime          NOT_RUN
actual GX10 GPU/model                   NOT_RUN
ChatGPT remote MCP connection           NOT_RUN
Gemini independent live call            NOT_RUN
```

The current execution environment is not joined to the user's Tailscale network and cannot reach `100.110.91.126` on SSH/model/dashboard ports. No SSH username, key, or public HTTPS deployment endpoint is available in the project context.

## Exact next execution

Personal server:

```bash
unzip ai_revenue_factory_g0_v1.zip
cd ai_revenue_factory_g0_v1
cp .env.example .env
python scripts/generate_env.py --env-file .env
bash scripts/bootstrap_server.sh
```

GX10 after the server is healthy:

```bash
cd ai_revenue_factory_g0_v1
python3 -m venv .venv
.venv/bin/pip install -e .
CORE_URL=https://YOUR_DOMAIN \
WORKER_TOKEN=YOUR_WORKER_TOKEN \
WORKER_ID=GX10-1 \
MODEL_URL=http://127.0.0.1:8000 \
MODEL_NAME=ACTUAL_MODEL_NAME \
.venv/bin/python scripts/run_gx10_worker.py --once
```

## Closure evidence still required

```text
docker compose ps: all required services healthy
/api/g0/status: PASS
JOB-G0-REAL-001: actual GX10 model input/output hashes
MCP tools/list: exactly three read-only tools
unauthorized publish/delete/money actions: zero
Gemini independent evidence packet: one actual result
```
