from fastapi.testclient import TestClient

from literature_kb.api import create_app


def test_api_import_search_citation_delete_contract(settings, write_source):
    write_source()
    with TestClient(create_app(settings)) as client:
        health = client.get("/v1/health")
        assert health.json()["answer_generation_enabled"] is False
        assert health.json()["external_calls_enabled"] is False
        assert health.headers["X-Request-ID"]
        result = client.post("/v1/documents/import", json={"relative_path": "test.md"})
        doc = result.json()["document_id"]
        search = client.post(
            "/v1/search",
            json={"query": "证据定位", "top_k": 3},
            headers={"X-Request-ID": "test_request_1"},
        )
        assert search.status_code == 200
        assert search.headers["X-Request-ID"] == "test_request_1"
        hit = search.json()["hits"][0]
        evidence = client.get(hit["evidence_url"]).json()
        assert (
            evidence["page_text"][evidence["page_char_start"] : evidence["page_char_end"]]
            == evidence["text"]
        )
        assert client.delete(f"/v1/documents/{doc}").status_code == 204
        assert client.get(hit["evidence_url"]).status_code == 410
        assert not client.post("/v1/search", json={"query": "证据定位"}).json()["hits"]


def test_api_validation_paths_unknown_ids_and_openapi(settings):
    with TestClient(create_app(settings)) as client:
        for query in (
            {"query": " "},
            {"query": "test", "top_k": 21},
            {"query": "x", "api_key": "fake"},
        ):
            assert client.post("/v1/search", json=query).status_code == 422
        blocked = client.post("/v1/documents/import", json={"relative_path": "../secret.csv"})
        assert blocked.status_code == 400
        assert blocked.json()["error"]["code"] == "invalid_source"
        assert client.get("/v1/evidence/unknown").status_code == 404
        assert client.get("/openapi.json").status_code == 200
