import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))
import api_client as api

OK_PREDICTION = {
    api.FIELD_RES_PREDICTION: 412.4,
    api.FIELD_RES_THRESHOLD: 380.0,
    api.FIELD_RES_ALERT: True,
    api.FIELD_RES_RISK: 1.085,
    api.FIELD_RES_FACTORS: [
        {
            "feature": "lag_1",
            "label": "1시간 전 교통량",
            "contribution": 31.2,
            "effect": "increase",
        }
    ],
    api.FIELD_RES_RECENT_TRAFFIC: [
        {"timestamp": "2026-08-31T22:00:00", "volume": 350.0},
        {"timestamp": "2026-08-31T23:00:00", "volume": 370.0},
    ],
}


def start_server(routes):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, method):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            received.append((method, self.path, json.loads(body) if body else None))
            status, payload = routes.get((method, self.path), (404, {"detail": "Not Found"}))
            raw = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            self._reply("GET")

        def do_POST(self):
            self._reply("POST")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}", received


@pytest.fixture
def serve():
    servers = []

    def _serve(routes):
        server, url, received = start_server(routes)
        servers.append(server)
        return url, received

    yield _serve
    for s in servers:
        s.shutdown()


def test_api_url_uses_env(monkeypatch):
    monkeypatch.delenv("TRAFFIC_API_URL", raising=False)
    assert api.api_url() == "http://127.0.0.1:8000"
    monkeypatch.setenv("TRAFFIC_API_URL", "http://example:9000/")
    assert api.api_url() == "http://example:9000"


def test_health_ok(serve):
    url, _ = serve({("GET", "/health"): (200, {"status": "ok"})})
    assert api.check_health(url) is True


def test_connection_failure_message():
    with pytest.raises(api.ApiError, match="연결할 수 없습니다"):
        api.check_health("http://127.0.0.1:1")


@pytest.mark.parametrize("payload", [
    {"불당사거리": ["A", "B"], "두정사거리": ["C"]},
    {"locations": [{"intersection": "불당사거리", "approaches": ["A", "B"]},
                   {"intersection": "두정사거리", "approaches": [{"name": "C"}]}]},
    [{"intersection": "불당사거리", "approach": "A"},
     {"intersection": "불당사거리", "approach": "B"},
     {"intersection": "두정사거리", "approach": "C"}],
])
def test_locations_shapes(serve, payload):
    url, _ = serve({("GET", "/locations"): (200, payload)})
    assert api.get_locations(url) == {"불당사거리": ["A", "B"], "두정사거리": ["C"]}


def test_locations_empty_is_error(serve):
    url, _ = serve({("GET", "/locations"): (200, [])})
    with pytest.raises(api.ApiError, match="목록"):
        api.get_locations(url)


def test_predict_ok_and_payload(serve):
    url, received = serve({("POST", "/predict"): (200, OK_PREDICTION)})
    r = api.predict("불당사거리", "A", base_url=url)
    assert r.predicted_volume == 412.4 and r.threshold == 380.0
    assert r.high_traffic_alert is True and r.risk_score == pytest.approx(1.085)
    assert r.major_factors[0].label == "1시간 전 교통량"
    assert r.major_factors[0].contribution == pytest.approx(31.2)
    assert [point.volume for point in r.recent_traffic] == [350.0, 370.0]
    assert received[-1] == ("POST", "/predict",
                            {api.FIELD_REQ_INTERSECTION: "불당사거리", api.FIELD_REQ_APPROACH: "A"})


def test_predict_404_message(serve):
    url, _ = serve({("POST", "/predict"): (404, {"detail": "unknown approach"})})
    with pytest.raises(api.ApiError, match="자료가 없어.*unknown approach"):
        api.predict("X", "Y", base_url=url)


def test_predict_422_message(serve):
    detail = [{"loc": ["body", "approach"], "msg": "field required", "type": "missing"}]
    url, _ = serve({("POST", "/predict"): (422, {"detail": detail})})
    with pytest.raises(api.ApiError, match="168시간.*field required"):
        api.predict("X", "", base_url=url)


def test_predict_500_message(serve):
    url, _ = serve({("POST", "/predict"): (500, {"detail": "boom"})})
    with pytest.raises(api.ApiError, match="HTTP 500"):
        api.predict("X", "Y", base_url=url)


def test_predict_missing_field(serve):
    bad = {k: v for k, v in OK_PREDICTION.items() if k != api.FIELD_RES_RISK}
    url, _ = serve({("POST", "/predict"): (200, bad)})
    with pytest.raises(api.ApiError, match=api.FIELD_RES_RISK):
        api.predict("X", "Y", base_url=url)


def test_predict_non_json(serve):
    url, _ = serve({("POST", "/predict"): (200, "<html>oops</html>")})
    with pytest.raises(api.ApiError, match="읽을 수 없습니다"):
        api.predict("X", "Y", base_url=url)


@pytest.mark.parametrize("raw, label", [
    ("불당4_남", "남쪽에서 진입 (불당4_남)"),
    ("번영로(갤러리아)4_북", "북쪽에서 진입 (번영로(갤러리아)4_북)"),
    ("도로원점3_동", "동쪽에서 진입 (도로원점3_동)"),
    ("이름형식다름", "이름형식다름"),
    ("방죽안5_북서", "방죽안5_북서"),
])
def test_approach_label(raw, label):
    assert api.approach_label(raw) == label


def test_predict_sends_target_timestamp(serve):
    body = dict(OK_PREDICTION, target_timestamp="2026-08-31T13:00:00")
    url, received = serve({("POST", "/predict"): (200, body)})
    r = api.predict("불당4", "불당4_남", target_timestamp="2026-08-31T13:00:00", base_url=url)
    assert received[-1][2] == {"intersection": "불당4", "approach": "불당4_남",
                               "target_timestamp": "2026-08-31T13:00:00"}
    assert r.target_time == "2026-08-31T13:00:00"


def test_backend_readme_example_response(serve):
    body = {"intersection": "도로원점3", "approach": "도로원점3_남",
            "target_timestamp": "2026-09-01T00:00:00", "predicted_volume": 193.502,
            "high_traffic_threshold": 1898.0, "high_traffic_alert": False, "risk_score": 0.102}
    url, _ = serve({("POST", "/predict"): (200, body)})
    r = api.predict("도로원점3", "도로원점3_남", base_url=url)
    assert (r.predicted_volume, r.threshold, r.high_traffic_alert, r.risk_score) == \
        (193.502, 1898.0, False, 0.102)


def test_predict_data_shortage_message(serve):
    body = {"status": "insufficient_history", "detail": "168시간 이력이 부족합니다"}
    url, _ = serve({("POST", "/predict"): (200, body)})
    with pytest.raises(api.ApiError, match="자료가 부족해.*168시간"):
        api.predict("불당4", "불당4_남", base_url=url)


def test_response_fields_match_model_handoff():
    assert (api.FIELD_RES_PREDICTION, api.FIELD_RES_THRESHOLD, api.FIELD_RES_ALERT,
            api.FIELD_RES_RISK) == ("predicted_volume", "high_traffic_threshold",
                                    "high_traffic_alert", "risk_score")


def test_locations_endpoint_missing_message(serve):
    url, _ = serve({})
    with pytest.raises(api.ApiError, match="/locations\\)이 없습니다"):
        api.get_locations(url)
