#!/usr/bin/env python3
"""Disposable-vault MCP round trip: index -> search -> hash-guarded read_passage.

Requires a locally built TurboVault executable, not Samuel's live vault or
inference sidecars: python3 tests/test_vault_retrieval_benchmark.py BINARY
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from vault_retrieval_benchmark import payload, request


class Embedder(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        inputs = body['input']
        # Stable, non-zero vectors; this checks wiring/provenance, not ranking.
        data = [{'index': i, 'embedding': [1.0, 0.0, 0.0, 0.0]}
                for i, _ in enumerate(inputs)]
        raw = json.dumps({'data': data}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


def main(binary):
    server = ThreadingHTTPServer(('127.0.0.1', 0), Embedder)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with tempfile.TemporaryDirectory(prefix='tv-passage-test-') as temp:
            root = Path(temp)
            vault = root / 'notes'
            vault.mkdir()
            (vault / 'guide.md').write_text(
                '---\ndescription: Synthetic test note\n---\n# Widget setup\n'
                'Calibrate the blue widget before the first test.\n'
                '## Safety\nRecord the calibration date.\n'
            )
            fixture = root / 'queries.json'
            fixture.write_text(json.dumps({'queries': [{
                'kind': 'conceptual', 'query': 'How should I prepare the blue widget?',
                'expected_paths': ['guide.md']
            }]}))
            env = os.environ.copy()
            env.update({'TURBOVAULT_EMBEDDING_ENDPOINT':
                        f'http://127.0.0.1:{server.server_port}/v1/embeddings',
                        'TURBOVAULT_RERANKER_ENABLED': 'false',
                        'TURBOVAULT_EMBEDDING_INDEX_DIR': str(root / 'index')})
            with subprocess.Popen([binary, '--vault', str(vault), '--profile', 'production'],
                                  env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1) as proc:
                request(proc, 'initialize', {'protocolVersion': '2025-06-18',
                        'capabilities': {}, 'clientInfo': {'name': 'passage-test', 'version': '1'}}, 1)
                proc.stdin.write(json.dumps({'jsonrpc': '2.0', 'method': 'notifications/initialized'}) + '\n')
                proc.stdin.flush()
                started = payload(request(proc, 'tools/call',
                    {'name': 'reindex_embeddings', 'arguments': {}}, 2))
                assert started.get('success'), started
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    status = payload(request(proc, 'tools/call',
                        {'name': 'embedding_index_status', 'arguments': {}}, 3))['data']
                    if status.get('reindex_phase') == 'complete':
                        break
                    if status.get('reindex_phase') == 'failed':
                        raise RuntimeError(str(status))
                    time.sleep(0.1)
                else:
                    raise TimeoutError('disposable index build did not finish')
                proc.stdin.close()
            result = subprocess.run([sys.executable,
                str(Path(__file__).with_name('vault_retrieval_benchmark.py')),
                '--binary', binary, '--vault', str(vault), '--fixture', str(fixture)],
                env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
                check=True)
            assert '"passage_checked": 1' in result.stdout, result.stdout
            assert '"passage_failed": 0' in result.stdout, result.stdout
            print(result.stdout)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main(str(Path(sys.argv[1]).resolve()))
