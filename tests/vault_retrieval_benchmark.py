#!/usr/bin/env python3
"""Opt-in, read-only live-vault benchmark via TurboVault's MCP stdio interface.

Use a built fork binary and a vault explicitly supplied by the operator. The
fixture contains only queries and labeled paths; no vault text is written.
Outputs aggregate top-k recall, MRR, latency, channel degradation and passage
provenance/bounds. No-answer probes are for human review, not precision scores:
retrieval without a calibrated abstention gate will usually return candidates.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

def request(proc, method, params, ident):
    proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": ident, "method": method, "params": params}) + "\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(f"MCP process ended while waiting for {method} (exit={proc.poll()})")
        try:
            reply = json.loads(line)
        except json.JSONDecodeError:
            continue
        if reply.get("id") == ident:
            if "error" in reply:
                raise RuntimeError(str(reply["error"]))
            return reply["result"]


def payload(result):
    if result.get("isError"):
        raise RuntimeError(str(result))
    if "structuredContent" in result:
        return result["structuredContent"]
    blocks = result.get("content", [])
    return json.loads(next(block["text"] for block in blocks if block.get("type") == "text"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--fixture", type=Path, required=True,
                        help="Private operator-supplied JSON labels; never commit personal vault inventory")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    probes = json.loads(args.fixture.read_text())["queries"]
    # The executable is the MCP server, not the installed pin unless passed explicitly.
    with subprocess.Popen([str(args.binary), "--vault", str(args.vault), "--profile", "production"],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
                          text=True, bufsize=1) as proc:
        request(proc, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                      "clientInfo": {"name": "vault-retrieval-benchmark", "version": "1"}}, 1)
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        proc.stdin.flush()
        tools = request(proc, "tools/list", {}, 2)["tools"]
        has_passage = any(t["name"] == "read_passage" for t in tools)
        stats = {"n": 0, "hit_1": 0, "hit_3": 0, "hit_10": 0, "mrr_10": 0.0,
                 "latency_ms": [], "degraded": 0, "passage_checked": 0, "passage_failed": 0}
        for idx, probe in enumerate(probes, 3):
            start = time.monotonic()
            result = payload(request(proc, "tools/call", {"name": "semantic_search", "arguments":
                             {"query": probe["query"], "limit": args.limit}}, idx * 2))
            elapsed_ms = round((time.monotonic() - start) * 1000)
            if not result.get("success", False):
                raise RuntimeError(str(result))
            paths = [row["path"] for row in result.get("data", [])]
            warnings = result.get("warnings", [])
            if warnings:
                stats["degraded"] += 1
            expected = set(probe["expected_paths"])
            rank = next((i for i, p in enumerate(paths[:10], 1) if p in expected), None)
            if expected:
                stats["n"] += 1
                for k in (1, 3, 10):
                    stats[f"hit_{k}"] += int(rank is not None and rank <= k)
                stats["mrr_10"] += 1 / rank if rank else 0
            print(f"{probe['kind']:10} rank={rank or '-':>2} ms={elapsed_ms:>5} "
                  f"warnings={len(warnings)} query={probe['query']} "
                  f"top={paths[:3]}")
            stats["latency_ms"].append(elapsed_ms)
            # One source-bound passage per labeled hit; no claim of answer validity.
            row = next((r for r in result.get("data", []) if r["path"] in expected and r.get("chunk_id") and r.get("chunk_hash")), None)
            if has_passage and row:
                read = payload(request(proc, "tools/call", {"name": "read_passage", "arguments":
                     {"path": row["path"], "chunk_id": row["chunk_id"], "expected_hash": row["chunk_hash"], "neighbors": 1,
                      "max_chars": 3000}}, idx * 2 + 1))
                data = read.get("data", {})
                chunks = data.get("chunks", [])
                valid = (read.get("success") and data.get("path") == row["path"]
                         and data.get("anchor_id") == row["chunk_id"]
                         and any(c["chunk_id"] == row["chunk_id"] for c in chunks)
                         and all(c["chunk_id"].startswith(row["path"] + "#") for c in chunks)
                         and sum(len(c["text"]) for c in chunks) <= 3000)
                stats["passage_checked"] += 1
                stats["passage_failed"] += int(not valid)
        n = stats["n"]
        print(json.dumps({"labeled": n, "recall_at_1": stats["hit_1"] / n,
                          "recall_at_3": stats["hit_3"] / n, "recall_at_10": stats["hit_10"] / n,
                          "mrr_at_10": stats["mrr_10"] / n,
                          "mean_latency_ms": round(sum(stats["latency_ms"]) / len(probes)),
                          "degraded_queries": stats["degraded"],
                          "passage_checked": stats["passage_checked"],
                          "passage_failed": stats["passage_failed"]}, indent=2))
        proc.stdin.close()
        if stats["passage_failed"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
