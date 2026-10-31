"""HTTP layer: static files, share API, health check, optional basic auth (no network)."""
import base64
import importlib
import os

import pytest
from fastapi.testclient import TestClient


def make_client(tmp_path, monkeypatch, auth=""):
    monkeypatch.setenv("ELB_DB", str(tmp_path / "t.sqlite"))
    monkeypatch.setenv("ELB_SCHEDULER", "0")
    monkeypatch.setenv("ELB_BASIC_AUTH", auth)
    import app.main as m
    m = importlib.reload(m)
    return TestClient(m.app)


def test_static_share_and_health(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as c:
        r = c.get("/")
        assert r.status_code == 200 and "ELB" in r.text and r.headers["cache-control"] == "no-cache"
        assert c.get("/api/status").status_code == 200
        sid = c.post("/api/share", json=dict(page="timeline", state=dict(tickers=[dict(ticker="AMD")]))).json()["id"]
        assert c.get(f"/api/share/{sid}").json()["state"]["tickers"][0]["ticker"] == "AMD"
        assert c.get("/api/share/zzzzzzzzzz").status_code == 404
        assert c.post("/api/share", json=dict(page="nope", state={})).status_code == 400


def test_basic_auth_when_configured(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch, auth="team:s3cret") as c:
        assert c.get("/").status_code == 401
        assert c.get("/api/share/0123456789").status_code == 401
        assert c.get("/api/status").status_code == 200                 # health check stays open for Render
        ok = {"Authorization": "Basic " + base64.b64encode(b"team:s3cret").decode()}
        bad = {"Authorization": "Basic " + base64.b64encode(b"team:wrong").decode()}
        assert c.get("/", headers=ok).status_code == 200
        assert c.get("/", headers=bad).status_code == 401
    monkeypatch.setenv("ELB_BASIC_AUTH", "")
    import app.main as m
    importlib.reload(m)
