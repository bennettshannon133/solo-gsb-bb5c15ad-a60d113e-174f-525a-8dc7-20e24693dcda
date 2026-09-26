"""POST /inspection-route 的端到端测试。

测试用树：

source --s1--> a --s2--> b
               |
               +--s3--> c --s4--> d

巡检参数：s1(t=5, r=10), s2(t=3, r=6), s3(t=2, r=4), s4(t=1, r=8)
祖先闭合子树（总耗时=单程和×2, 总风险分）：
  {}=(0,0)  {s1}=(10,10)  {s1,s2}=(16,16)  {s1,s3}=(14,14)
  {s1,s2,s3}=(20,20)  {s1,s3,s4}=(16,22)  {s1,s2,s3,s4}=(22,28)
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def base_payload(**overrides):
    payload = {
        "segments": [
            {"id": "s1", "parent": "source", "child": "a", "resistance": 0.1, "rated_current": 20.0},
            {"id": "s2", "parent": "a", "child": "b", "resistance": 0.2, "rated_current": 10.0},
            {"id": "s3", "parent": "a", "child": "c", "resistance": 0.05, "rated_current": 15.0},
            {"id": "s4", "parent": "c", "child": "d", "resistance": 0.1, "rated_current": 10.0},
        ],
        "segment_params": [
            {"segment_id": "s1", "travel_minutes": 5, "risk_score": 10.0},
            {"segment_id": "s2", "travel_minutes": 3, "risk_score": 6.0},
            {"segment_id": "s3", "travel_minutes": 2, "risk_score": 4.0},
            {"segment_id": "s4", "travel_minutes": 1, "risk_score": 8.0},
        ],
        "max_minutes": 22,
    }
    payload.update(overrides)
    return payload


# ---------- 路线择优 ----------


def test_full_tree_when_time_allows():
    """时限 22 恰好容纳全树：总耗时 22 = (5+3+2+1)×2，总风险分 28。"""
    resp = client.post("/inspection-route", json=base_payload())
    assert resp.status_code == 200
    data = resp.json()

    assert data["selected_segments"] == ["s1", "s2", "s3", "s4"]
    assert data["total_risk_score"] == pytest.approx(28.0)
    assert data["total_travel_minutes"] == 22
    # 子节点按 segment_id 升序（a 的子线段 s2 先于 s3）的深度优先往返路径
    assert data["path"] == ["source", "a", "b", "a", "c", "d", "c", "a", "source"]


def test_best_subtree_within_limit():
    """时限 16：{s1,s3,s4}（风险 22，耗时 16）优于 {s1,s2,s3}（风险 20，耗时 20 超限）。"""
    resp = client.post("/inspection-route", json=base_payload(max_minutes=16))
    assert resp.status_code == 200
    data = resp.json()

    assert data["selected_segments"] == ["s1", "s3", "s4"]
    assert data["total_risk_score"] == pytest.approx(22.0)
    assert data["total_travel_minutes"] == 16
    assert data["path"] == ["source", "a", "c", "d", "c", "a", "source"]


def test_exact_limit_is_feasible():
    """总耗时恰好等于时限时可行（时限 10 恰容纳 {s1}）。"""
    resp = client.post("/inspection-route", json=base_payload(max_minutes=10))
    assert resp.status_code == 200
    data = resp.json()

    assert data["selected_segments"] == ["s1"]
    assert data["total_risk_score"] == pytest.approx(10.0)
    assert data["total_travel_minutes"] == 10
    assert data["path"] == ["source", "a", "source"]


def test_ancestor_segments_included():
    """巡检深层高风险线段必须连同其全部祖先段一起入选。"""
    payload = base_payload(
        segments=[
            {"id": "s1", "parent": "source", "child": "a", "resistance": 0.1, "rated_current": 20.0},
            {"id": "s2", "parent": "a", "child": "b", "resistance": 0.2, "rated_current": 10.0},
        ],
        segment_params=[
            {"segment_id": "s1", "travel_minutes": 5, "risk_score": 1.0},
            {"segment_id": "s2", "travel_minutes": 1, "risk_score": 100.0},
        ],
        max_minutes=12,
    )
    resp = client.post("/inspection-route", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # s2 风险 100 但须带上祖先 s1：总耗时 (5+1)×2 = 12 恰触时限
    assert data["selected_segments"] == ["s1", "s2"]
    assert data["total_risk_score"] == pytest.approx(101.0)
    assert data["total_travel_minutes"] == 12
    assert data["path"] == ["source", "a", "b", "a", "source"]


def test_insufficient_time_returns_empty_route():
    """最便宜的非空子树 {s1} 需 10 分钟，时限 9 时返回仅含 source 的空路线。"""
    resp = client.post("/inspection-route", json=base_payload(max_minutes=9))
    assert resp.status_code == 200
    data = resp.json()

    assert data["selected_segments"] == []
    assert data["total_risk_score"] == pytest.approx(0.0)
    assert data["total_travel_minutes"] == 0
    assert data["path"] == ["source"]


def test_zero_limit_returns_empty_route():
    resp = client.post("/inspection-route", json=base_payload(max_minutes=0))
    assert resp.status_code == 200
    data = resp.json()

    assert data["selected_segments"] == []
    assert data["total_travel_minutes"] == 0
    assert data["path"] == ["source"]


def test_zero_risk_scores_yield_empty_route():
    """全部风险分为 0 时，空路线凭耗时更少（0）在同分中胜出。"""
    payload = base_payload(
        segment_params=[
            {"segment_id": "s1", "travel_minutes": 5, "risk_score": 0.0},
            {"segment_id": "s2", "travel_minutes": 3, "risk_score": 0.0},
            {"segment_id": "s3", "travel_minutes": 2, "risk_score": 0.0},
            {"segment_id": "s4", "travel_minutes": 1, "risk_score": 0.0},
        ]
    )
    resp = client.post("/inspection-route", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["selected_segments"] == []
    assert data["total_risk_score"] == pytest.approx(0.0)
    assert data["total_travel_minutes"] == 0
    assert data["path"] == ["source"]


# ---------- 平局择优 ----------


def test_tie_prefers_less_time():
    """同分（5）时取耗时更少者：{s1} 耗时 4 优于 {s2} 耗时 6。"""
    payload = base_payload(
        segments=[
            {"id": "s1", "parent": "source", "child": "a", "resistance": 0.1, "rated_current": 20.0},
            {"id": "s2", "parent": "source", "child": "b", "resistance": 0.2, "rated_current": 10.0},
        ],
        segment_params=[
            {"segment_id": "s1", "travel_minutes": 2, "risk_score": 5.0},
            {"segment_id": "s2", "travel_minutes": 3, "risk_score": 5.0},
        ],
        max_minutes=6,
    )
    resp = client.post("/inspection-route", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["selected_segments"] == ["s1"]
    assert data["total_risk_score"] == pytest.approx(5.0)
    assert data["total_travel_minutes"] == 4
    assert data["path"] == ["source", "a", "source"]


def test_tie_prefers_lexicographically_smaller_ids():
    """同分且同耗时（均为 6）时取排序后 segment_id 列表字典序更小者。"""
    payload = base_payload(
        segments=[
            {"id": "s2", "parent": "source", "child": "b", "resistance": 0.2, "rated_current": 10.0},
            {"id": "s1", "parent": "source", "child": "a", "resistance": 0.1, "rated_current": 20.0},
        ],
        segment_params=[
            {"segment_id": "s1", "travel_minutes": 3, "risk_score": 5.0},
            {"segment_id": "s2", "travel_minutes": 3, "risk_score": 5.0},
        ],
        max_minutes=6,
    )
    resp = client.post("/inspection-route", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # {s1} 与 {s2} 风险、耗时相同，["s1"] < ["s2"]
    assert data["selected_segments"] == ["s1"]
    assert data["total_travel_minutes"] == 6
    assert data["path"] == ["source", "a", "source"]


def test_children_visited_in_segment_id_order():
    """DFS 子节点按 segment_id 升序访问（与线段输入顺序、节点名称无关）。"""
    payload = base_payload(
        segments=[
            {"id": "s2", "parent": "source", "child": "y", "resistance": 0.2, "rated_current": 10.0},
            {"id": "s1", "parent": "source", "child": "z", "resistance": 0.1, "rated_current": 20.0},
        ],
        segment_params=[
            {"segment_id": "s1", "travel_minutes": 1, "risk_score": 1.0},
            {"segment_id": "s2", "travel_minutes": 1, "risk_score": 1.0},
        ],
        max_minutes=10,
    )
    resp = client.post("/inspection-route", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["selected_segments"] == ["s1", "s2"]
    # s1（子节点 z）先于 s2（子节点 y），与输入顺序相反
    assert data["path"] == ["source", "z", "source", "y", "source"]


# ---------- 数值展示与规模 ----------


def test_total_risk_score_rounded_to_six_decimals():
    payload = base_payload(
        segments=[
            {"id": "s1", "parent": "source", "child": "a", "resistance": 0.1, "rated_current": 20.0},
        ],
        segment_params=[
            {"segment_id": "s1", "travel_minutes": 1, "risk_score": 0.123456789},
        ],
        max_minutes=10,
    )
    resp = client.post("/inspection-route", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["total_risk_score"] == pytest.approx(0.123457, abs=1e-9)
    assert data["total_travel_minutes"] == 2


def test_twelve_segments_accepted():
    """12 段链条为规模上限：全部入选，往返路径共 25 个节点。"""
    segments = [
        {
            "id": f"seg-{i}",
            "parent": "source" if i == 1 else f"n{i - 1}",
            "child": f"n{i}",
            "resistance": 0.1,
            "rated_current": 20.0,
        }
        for i in range(1, 13)
    ]
    payload = base_payload(
        segments=segments,
        segment_params=[
            {"segment_id": f"seg-{i}", "travel_minutes": 1, "risk_score": 1.0}
            for i in range(1, 13)
        ],
        max_minutes=24,
    )
    resp = client.post("/inspection-route", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # 入选线段按 segment_id 字符串升序（seg-10/11/12 排在 seg-2 之前）
    assert data["selected_segments"] == sorted(f"seg-{i}" for i in range(1, 13))
    assert data["total_risk_score"] == pytest.approx(12.0)
    assert data["total_travel_minutes"] == 24
    assert len(data["path"]) == 25
    assert data["path"][0] == data["path"][-1] == "source"


def test_thirteen_segments_rejected():
    segments = [
        {
            "id": f"seg-{i}",
            "parent": "source" if i == 1 else f"n{i - 1}",
            "child": f"n{i}",
            "resistance": 0.1,
            "rated_current": 20.0,
        }
        for i in range(1, 14)
    ]
    payload = base_payload(
        segments=segments,
        segment_params=[
            {"segment_id": f"seg-{i}", "travel_minutes": 1, "risk_score": 1.0}
            for i in range(1, 14)
        ],
    )
    resp = client.post("/inspection-route", json=payload)
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert any("12" in d["message"] for d in error["details"])


def test_segment_ids_auto_generated():
    """线段 id 缺省自动生成 seg-<序号>，巡检参数按生成 id 对应。"""
    payload = base_payload()
    for seg in payload["segments"][:2]:
        del seg["id"]
    payload["segments"] = payload["segments"][:2]
    payload["segment_params"] = [
        {"segment_id": "seg-1", "travel_minutes": 5, "risk_score": 10.0},
        {"segment_id": "seg-2", "travel_minutes": 3, "risk_score": 6.0},
    ]
    payload["max_minutes"] = 16
    resp = client.post("/inspection-route", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["selected_segments"] == ["seg-1", "seg-2"]
    assert data["total_travel_minutes"] == 16


# ---------- 请求校验（422） ----------


def test_non_positive_travel_minutes_rejected():
    for bad in (0, -1):
        payload = base_payload()
        payload["segment_params"][0]["travel_minutes"] = bad
        resp = client.post("/inspection-route", json=payload)
        assert resp.status_code == 422
        error = resp.json()["error"]
        assert error["code"] == "validation_error"
        assert any("单程耗时必须为正整数" in d["message"] for d in error["details"])


def test_non_integer_travel_minutes_rejected():
    payload = base_payload()
    payload["segment_params"][0]["travel_minutes"] = 1.5
    resp = client.post("/inspection-route", json=payload)
    assert resp.status_code == 422


def test_negative_risk_score_rejected():
    payload = base_payload()
    payload["segment_params"][0]["risk_score"] = -1.0
    resp = client.post("/inspection-route", json=payload)
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert any("风险分不能为负" in d["message"] for d in error["details"])


def test_negative_max_minutes_rejected():
    resp = client.post("/inspection-route", json=base_payload(max_minutes=-1))
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert any("巡检时限不能为负" in d["message"] for d in error["details"])


def test_non_integer_max_minutes_rejected():
    resp = client.post("/inspection-route", json=base_payload(max_minutes=1.5))
    assert resp.status_code == 422


def test_empty_segment_params_rejected():
    resp = client.post("/inspection-route", json=base_payload(segment_params=[]))
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "validation_error"


def test_empty_segments_rejected():
    resp = client.post("/inspection-route", json=base_payload(segments=[], segment_params=[]))
    assert resp.status_code == 422


# ---------- 业务校验（400 invalid_input） ----------


def test_missing_segment_param_rejected():
    payload = base_payload()
    payload["segment_params"] = payload["segment_params"][:3]  # 缺 s4
    resp = client.post("/inspection-route", json=payload)
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["code"] == "invalid_input"
    assert any("s4" in d["message"] and "缺少巡检参数" in d["message"] for d in error["details"])


def test_duplicate_segment_param_rejected():
    payload = base_payload()
    payload["segment_params"].append(
        {"segment_id": "s1", "travel_minutes": 1, "risk_score": 1.0}
    )
    resp = client.post("/inspection-route", json=payload)
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["code"] == "invalid_input"
    assert any("重复定义" in d["message"] for d in error["details"])


def test_unknown_segment_param_rejected():
    payload = base_payload()
    payload["segment_params"].append(
        {"segment_id": "ghost", "travel_minutes": 1, "risk_score": 1.0}
    )
    resp = client.post("/inspection-route", json=payload)
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["code"] == "invalid_input"
    assert any("不存在于树中" in d["message"] for d in error["details"])


def test_invalid_tree_rejected():
    payload = base_payload()
    payload["segments"].append(
        {"id": "s5", "parent": "x", "child": "y", "resistance": 0.1, "rated_current": 10.0}
    )
    payload["segment_params"].append(
        {"segment_id": "s5", "travel_minutes": 1, "risk_score": 1.0}
    )
    resp = client.post("/inspection-route", json=payload)
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["code"] == "invalid_input"
    assert any("不连通" in d["message"] for d in error["details"])
