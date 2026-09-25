#!/usr/bin/env python3
"""调用 u2_decision_4b HTTP 服务：三条 DEMO 冒烟测试。

先启动：
  python serve/serve.py --serve --port 8000 --base /path/to/Qwen3.5-4B

再跑：
  python serve/call_demo.py
  python serve/call_demo.py --case safety
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root


def _load_jsonl_by_id(path: Path, rid: str) -> dict:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("id") == rid:
                return o
    raise KeyError(f"{rid} not in {path}")


def _strip_train_fields(questions: dict) -> dict:
    out = {}
    for k, q in (questions or {}).items():
        qq = {kk: vv for kk, vv in dict(q or {}).items() if kk not in ("label", "soft_target", "src")}
        out[k] = qq
    return out


def case_safety() -> dict:
    row = _load_jsonl_by_id(ROOT / "data/safety_holdout500.jsonl", "safety001")
    return {
        "name": "safety / Demo1",
        "id": row["id"],
        "gold": "safe",
        "payload": {"state": row["state"], "questions": _strip_train_fields(row["questions"])},
        "expect_field": "safety",
        "expect_choice": "safe",
    }


def case_deepseek() -> dict:
    row = _load_jsonl_by_id(ROOT / "data/kev_deepseek_holdout.jsonl", "common052")
    return {
        "name": "decision / Demo1",
        "id": row["id"],
        "gold": "technical_consultation",
        "payload": {"state": row["state"], "questions": _strip_train_fields(row["questions"])},
        "expect_field": "decision",
        "expect_choice": "technical_consultation",
    }


def case_routing() -> dict:
    rows = json.loads((ROOT / "data/routing12/test_query_weights.json").read_text(encoding="utf-8"))
    query = rows[0]["query"]
    dims = ["Shell_execution", "Debugging", "Code_generation"]
    questions = {
        f"noul__{d}": {
            "type": "noul",
            "instructions": (
                f"Does successfully completing this task substantively require {d}?\n"
                "Answer true if required for a core step; false if incidental."
            ),
        }
        for d in dims
    }
    return {
        "name": "routing12 / DemoA task0",
        "id": "routing_task0",
        "gold": "Top-3: " + " · ".join(dims),
        "payload": {"state": query, "questions": questions},
        "expect_field": None,
        "expect_choice": None,
        "routing_dims": dims,
    }


CASES = {"safety": case_safety, "deepseek": case_deepseek, "routing": case_routing}


def post_systemone(base_url: str, payload: dict, timeout: float = 120.0) -> dict:
    url = base_url.rstrip("/") + "/v1/systemone"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_health(base_url: str) -> None:
    url = base_url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            print(f"[health] {resp.status} {resp.read().decode('utf-8')}")
    except Exception as e:
        print(f"[health] FAIL {url}: {e}", file=sys.stderr)
        print("先启动: python serve/serve.py --serve --port 8000 --base <Qwen3.5-4B>", file=sys.stderr)
        raise SystemExit(1)


def summarize(case: dict, resp: dict) -> None:
    answers = resp.get("answers") or {}
    print(f"\n=== {case['name']}  id={case['id']}")
    print(f"    gold: {case['gold']}")
    print(f"    latency_ms: {resp.get('latency_ms')}  model: {resp.get('model')}")
    if case.get("routing_dims"):
        scored = [(d, float((answers.get(f"noul__{d}") or {}).get("noul", 0.0))) for d in case["routing_dims"]]
        scored.sort(key=lambda x: -x[1])
        print("    noul P(true) on gold Top-3 dims:")
        for d, p in scored:
            print(f"      {d:20s}  {p:.4f}")
        return
    field = case["expect_field"]
    a = answers.get(field) or {}
    print(f"    answer[{field}]: {json.dumps(a, ensure_ascii=False)}")
    got = a.get("choice")
    expect = case.get("expect_choice")
    if expect is not None:
        print(f"    check: got={got!r} expect={expect!r}  {'OK' if str(got) == str(expect) else 'FAIL'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--case", choices=["all", *CASES.keys()], default="all")
    ap.add_argument("--dump-payload", action="store_true")
    args = ap.parse_args()
    names = list(CASES.keys()) if args.case == "all" else [args.case]
    cases = [CASES[n]() for n in names]
    if args.dump_payload:
        for c in cases:
            print(json.dumps({"name": c["name"], **c["payload"]}, ensure_ascii=False, indent=2))
            print("---")
        return 0
    check_health(args.base_url)
    for c in cases:
        try:
            resp = post_systemone(args.base_url, c["payload"])
        except Exception as e:
            print(f"\n=== {c['name']} ERROR: {e}", file=sys.stderr)
            return 1
        summarize(c, resp)
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
