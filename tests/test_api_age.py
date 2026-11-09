import os

import pytest
from fastapi.testclient import TestClient

import solgraph.api as api
from solgraph.app import App
from solgraph.config import Settings
from solgraph.graph import AgeGraph, trace_origin
from solgraph.ingest import ingest_wallet
from solgraph.models import SOL_MINT

from .conftest import ALICE, BOB, FUNDER


@pytest.fixture
def client(box):
    s = Settings("devnet", "http://fake", None, "solgraph", "test")
    api._state["app"] = App(s, box.rpc, box.graph, box)
    yield TestClient(api.app)
    api._state.clear()


def test_api_routes(client):
    assert client.get("/health").json()["network"] == "devnet"
    assert client.get(f"/wallet/{ALICE}/balances").json()["sol"] == 1.1
    assert client.post(f"/wallet/{BOB}/ingest?limit=5").json()["loaded_transactions"] == 2
    client.post(f"/wallet/{ALICE}/ingest")
    assert len(client.get(f"/wallet/{BOB}/origin/SOL").json()["transfers"]) == 2
    assert client.get("/wallet/nope/balances").status_code == 422


DSN = os.environ.get("SOLGRAPH_TEST_PG_DSN")


@pytest.mark.skipif(not DSN, reason="set SOLGRAPH_TEST_PG_DSN to a Postgres with Apache AGE")
def test_age_backend_matches_memory(rpc):
    g = AgeGraph(DSN, "solgraph_test")
    g._cypher("MATCH (n) DETACH DELETE n RETURN 1")
    for w in (ALICE, BOB, FUNDER):
        ingest_wallet(rpc, g, w)
    edges = trace_origin(g, BOB, SOL_MINT)
    assert sorted((e.source, e.destination) for e in edges) == [(ALICE, BOB), (FUNDER, ALICE)]
    assert g.holdings(ALICE)["SOL"] == 1.1
    assert g.stats()["transfers"] >= 3
    assert g.kinds([ALICE, BOB, "Vau1t" + "V" * 39]) == {
        ALICE: "user", BOB: "user", "Vau1t" + "V" * 39: "unknown"}
