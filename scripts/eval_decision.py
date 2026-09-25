#!/usr/bin/env python3
"""评测：通用决策 holdout（choice / noul / score）。

数据: data/kev_deepseek_holdout.jsonl（212 条）

用法:
  python scripts/eval_decision.py --base /path/to/Qwen3.5-4B
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "serve"))
import serve as srv  # noqa: E402


def prep_question(q: dict):
    q = dict(q)
    kind = q.get("type") or "choice"
    gold = q.pop("label", None)
    q.pop("soft_target", None)
    q.pop("src", None)
    crit = q.get("criteria")
    if kind == "score" and isinstance(crit, list):
        q["type"] = "score"
        gold = int(gold) if gold is not None else None
        kind = "score"
    elif kind == "noul":
        if isinstance(gold, str):
            gold = gold.lower() == "true"
        kind = "noul"
    else:
        kind = "choice"
        if gold is not None:
            gold = str(gold)
    return q, gold, kind


def pred_from_answer(answer, kind):
    if not answer or not isinstance(answer, dict):
        return None
    if kind == "noul":
        if "noul" in answer:
            try:
                return float(answer["noul"]) >= 0.5
            except Exception:
                return None
        return str(answer.get("choice", "")).lower() == "true"
    ch = answer.get("choice")
    if kind == "score":
        try:
            return int(ch)
        except Exception:
            return ch
    return str(ch) if ch is not None else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(REPO / "adapter"))
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", default=str(REPO / "data/kev_deepseek_holdout.jsonl"))
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.n > 0:
        rows = rows[: args.n]

    model, tok, device = srv.load_model(args.adapter, args.base)
    by_kind = Counter()
    ok_kind = Counter()
    results = []
    for i, r in enumerate(rows):
        # typically one field "decision"
        fname, qraw = next(iter(r["questions"].items()))
        q, gold, kind = prep_question(qraw)
        answers = srv.predict_answers(
            model, tok, device, r["state"], {fname: q}, batch_size=args.batch_size
        )
        pred = pred_from_answer(answers.get(fname), kind)
        ok = pred == gold
        by_kind[kind] += 1
        ok_kind[kind] += int(ok)
        results.append({"id": r.get("id"), "kind": kind, "gold": gold, "pred": pred, "ok": ok})
        if (i + 1) % 30 == 0 or i + 1 == len(rows):
            acc = sum(ok_kind.values()) / max(1, sum(by_kind.values()))
            print(f"[{i+1}/{len(rows)}] running_acc={acc:.4f}", flush=True)

    total = sum(by_kind.values())
    summary = {
        "suite": "kev_deepseek_holdout",
        "n": total,
        "acc": round(sum(ok_kind.values()) / max(1, total), 4),
        "by_type": {
            k: {"n": by_kind[k], "acc": round(ok_kind[k] / by_kind[k], 4)} for k in sorted(by_kind)
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for o in results:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
