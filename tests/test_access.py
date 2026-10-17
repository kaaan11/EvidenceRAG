import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from app.access import LocalAccessMiddleware, access_denial
from app.rag.config import Settings
import main


def scope(host="127.0.0.1", origin=None, peer="127.0.0.1", method="POST", scheme="http", extra=()):
    headers = [(b"host", host.encode())] if host is not None else []
    if origin is not None: headers.append((b"origin", origin.encode()))
    return {"type": "http", "method": method, "scheme": scheme, "path": "/documents", "headers": headers + list(extra), "client": (peer, 50000) if peer else None}


@pytest.mark.parametrize("host,origin,peer,scheme", [
    ("localhost", "http://localhost", "127.0.0.1", "http"),
    ("localhost:8000", "http://localhost:8000", "127.0.0.1", "http"),
    ("127.0.0.1:1234", "http://127.0.0.1:1234", "127.0.0.1", "http"),
    ("[::1]", "http://[::1]", "::1", "http"),
    ("[::1]:8000", "http://[::1]:8000", "::1", "http"),
    ("localhost:443", "https://localhost", "::1", "https"),
    ("LOCALHOST:80", "http://localhost", "127.0.0.1", "http"),
    ("127.0.0.1", None, "127.0.0.1", "http"),
])
def test_valid_loopback_host_origin_and_cli(host, origin, peer, scheme):
    assert access_denial(scope(host, origin, peer, scheme=scheme)) is None


@pytest.mark.parametrize("host", [None, "localhost.evil.example", "127.0.0.1.evil.example", "evil@localhost", "localhost:0", "localhost:65536", "localhost:bad", "localhost:", "::1", "[::1]:-1", "localhost.", "localhost/path", "localhost#x", "localhost?x", "localhost\\evil", "localhost ", "192.168.1.2"])
def test_invalid_hosts_rejected(host):
    assert access_denial(scope(host))[0] == 400


@pytest.mark.parametrize("origin", ["null", "http://localhost.evil.example", "http://localhost", "http://127.0.0.1:81", "https://127.0.0.1", "http://user@127.0.0.1", "http://127.0.0.1/", "http://127.0.0.1?x", "http://127.0.0.1#x", "garbage"])
def test_bad_or_cross_origin_mutations_rejected(origin):
    assert access_denial(scope(origin=origin))[0] == 403


@pytest.mark.parametrize("peer", ["192.168.1.8", "8.8.8.8", "testclient", None])
def test_non_loopback_peer_cannot_impersonate_localhost(peer):
    assert access_denial(scope(peer=peer))[0] == 403


@pytest.mark.parametrize("name", [b"forwarded", b"x-forwarded-for", b"x-forwarded-host", b"x-forwarded-proto", b"x-forwarded-port", b"x-real-ip"])
def test_proxy_headers_do_not_establish_local_trust(name):
    assert access_denial(scope(extra=[(name, b"127.0.0.1")]))[0] == 400


def test_duplicate_host_and_origin_are_rejected():
    assert access_denial(scope(extra=[(b"host", b"localhost")]))[0] == 400
    assert access_denial(scope(origin="http://127.0.0.1", extra=[(b"origin", b"http://127.0.0.1")]))[0] == 403


def test_access_denial_precedes_receive_parser_or_model():
    calls = []
    async def app(scope, receive, send): calls.append("downstream")
    async def receive(): raise AssertionError("No body may be read")
    messages = []
    async def send(message): messages.append(message)
    denied = scope(origin="null")
    asyncio.run(LocalAccessMiddleware(app)(denied, receive, send))
    assert messages[0]["status"] == 403 and calls == []


@pytest.mark.parametrize("method,path", [("GET", "/documents"), ("POST", "/documents"), ("DELETE", "/documents/x"), ("POST", "/ask"), ("GET", "/health")])
def test_real_app_private_routes_reject_remote_peer_before_body(method, path):
    with TestClient(main.app, base_url="http://localhost", client=("192.168.1.2", 123)) as client:
        assert client.request(method, path).status_code == 403


def test_local_shell_health_and_docs_are_available():
    with TestClient(main.app, base_url="http://localhost", client=("127.0.0.1", 123)) as client:
        for route in ["/", "/static/app.js", "/health", "/docs", "/openapi.json"]:
            assert client.get(route).status_code == 200


@pytest.mark.parametrize("profile", ["shared", "public", "LOCAL", "unknown"])
def test_unsupported_shared_profile_fails_startup(profile):
    with pytest.raises(ValueError, match="shared/public access is not supported"):
        Settings(access_profile=profile)


def test_shared_profile_fails_even_with_an_unrelated_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "synthetic-test-value")
    with pytest.raises(ValueError): Settings(access_profile="shared")


def test_real_app_origin_denial_has_zero_parser_or_pipeline_calls(monkeypatch):
    import app.request_limits as limits
    calls = []
    def forbidden(*args, **kwargs):
        calls.append("unexpected work")
        raise AssertionError("Access must be checked first")
    monkeypatch.setattr(limits, "LimitedMultipartParser", forbidden)
    for name in ["ask", "index_file", "delete_document"]:
        monkeypatch.setattr(main.rag_pipeline, name, forbidden)
    with TestClient(main.app, base_url="http://127.0.0.1", client=("127.0.0.1", 1)) as client:
        headers = {"Origin": "null"}
        assert client.post("/documents", files={"file": ("a.txt", b"x")}, headers=headers).status_code == 403
        assert client.post("/ask", json={"question": "ok"}, headers=headers).status_code == 403
        assert client.delete("/documents/x", headers=headers).status_code == 403
    assert calls == []


def test_ui_uses_text_nodes_and_normalizes_error_details(tmp_path):
    node = os.getenv("EVIDENCERAG_NODE_BINARY") or shutil.which("node")
    if not node: pytest.skip("Node required for DOM-stub rendering check")
    script = r'''
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
function element() {
  const e = {children:[], style:{}, textContent:'', value:'', addEventListener(){}, setAttribute(){},
    classList:{add(){}, remove(){}, toggle(){}}, append(...xs){this.children.push(...xs)},
    appendChild(x){this.children.push(x)}, replaceChildren(...xs){this.children=xs}, querySelector(){return element()}};
  Object.defineProperty(e,'innerHTML',{set(){throw Error('HTML sink forbidden')}}); return e;
}
const els = new Map(), document={getElementById(id){if(!els.has(id))els.set(id,element());return els.get(id)},createElement:element};
const attack='<script>globalThis.compromised=true</script>', captured=[];
const context=vm.createContext({document,window:{addEventListener(){}},fetch:async(url,options)=>{
  captured.push([url,options]); return {ok:true,json:async()=>({answer:attack,sources:[{filename:attack,text:attack,score:1,cited:true}],verification:null})};
}});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
(async()=>{
  els.get('question').value=attack; await context.askQuestion();
  assert.equal(JSON.parse(captured[0][1].body).question,attack);
  const texts=[]; function walk(e){texts.push(e.textContent);for(const c of e.children)walk(c)}
  walk(els.get('answerCard')); walk(els.get('sourcesList'));
  assert.ok(texts.filter(x=>x===attack).length>=3); assert.equal(context.compromised,undefined);
  const parsed=await context.parseResponse({json:async()=>({detail:[{msg:'Too long'},{msg:'Invalid'}]})});
  assert.equal(parsed.detail,'Too long; Invalid');
  console.log('UI text sinks, question payload and error details passed (stubbed DOM, not browser).');
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    result = subprocess.run([node, "-e", script, str(Path(__file__).resolve().parents[1]/"static/app.js")], capture_output=True,text=True,timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
