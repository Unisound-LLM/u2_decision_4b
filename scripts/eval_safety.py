#!/usr/bin/env python3
"""评测：安全审核 holdout500（500 条；口径：3 分类 accuracy）。

用法:
  python scripts/eval_safety.py --base /path/to/Qwen3.5-4B
  python scripts/eval_safety.py --base Qwen/Qwen3.5-4B --adapter ./adapter --n 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "serve"))
import serve as srv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(REPO / "adapter"))
    ap.add_argument("--base", required=True, help="Qwen3.5-4B 基座路径或 HF id")
    ap.add_argument("--data", default=str(REPO / "data/safety_holdout500.jsonl"))
    ap.add_argument("--n", type=int, default=0, help="只评前 N 条，0=全部")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="", help="可选：写出逐条 jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.n > 0:
        rows = rows[: args.n]

    model, tok, device = srv.load_model(args.adapter, args.base)
    correct = 0
    results = []
    for i, r in enumerate(rows):
        q = dict(r["questions"]["safety"])
        gold = q.pop("label", None)
        q.pop("soft_target", None)
        q.pop("src", None)
        answers = srv.predict_answers(
            model, tok, device, r["state"], {"safety": q}, batch_size=args.batch_size
        )
        pred = (answers.get("safety") or {}).get("choice")
        ok = pred == gold
        correct += int(ok)
        results.append({"id": r.get("id"), "gold": gold, "pred": pred, "ok": ok, "answer": answers.get("safety")})
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            print(f"[{i+1}/{len(rows)}] running_acc={correct/(i+1):.4f}", flush=True)

    acc = correct / max(1, len(rows))
    print(json.dumps({"suite": "safety_holdout500", "n": len(rows), "acc": round(acc, 4), "correct": correct}, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for o in results:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
