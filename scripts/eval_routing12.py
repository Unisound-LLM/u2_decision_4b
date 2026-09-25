#!/usr/bin/env python3
"""评测：能力路由 12 条（Choice / Noul / Score → 一致性 + 9 项综合分）。

口径对齐 COMPARE_ROUTING12：
  - 每模式：Top-1 exact / Top-1 in ref Top-3 / avg Top-3 overlap
  - 综合分：9 项归一化后取平均（exact/n、in_ref3/n、overlap/3）

用法:
  python scripts/eval_routing12.py --base /path/to/Qwen3.5-4B
  python scripts/eval_routing12.py --base /path/to/Qwen3.5-4B --n 3
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "serve"))
sys.path.insert(0, str(REPO / "scripts"))
import serve as srv  # noqa: E402

spec = importlib.util.spec_from_file_location("pack_utils", REPO / "scripts/pack_utils.py")
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)

rspec = importlib.util.spec_from_file_location("routing_metrics", REPO / "scripts/routing_metrics.py")
rm = importlib.util.module_from_spec(rspec)
rspec.loader.exec_module(rm)


@torch.no_grad()
def predict_probs(model, tok, contexts, candidate_ids_list, device, batch_size=16, max_len=2048):
    probs_list = []
    for i in range(0, len(contexts), batch_size):
        batch_ctx = contexts[i : i + batch_size]
        batch_cids = candidate_ids_list[i : i + batch_size]
        enc = tok(batch_ctx, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        logits = model(
            **{k: enc[k] for k in ("input_ids", "attention_mask")},
            use_cache=False,
            logits_to_keep=1,
        ).logits[:, -1].float()
        for j, cids in enumerate(batch_cids):
            lj = logits[j, cids]
            probs_list.append(torch.softmax(lj, dim=0).cpu().tolist())
    return probs_list


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(REPO / "adapter"))
    ap.add_argument("--base", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--out", default=str(REPO / "results/routing12.json"))
    args = ap.parse_args()

    data_dir = REPO / "data/routing12"
    rows = json.loads((data_dir / "test_query_weights.json").read_text(encoding="utf-8"))[: args.n]
    questions = rm.build_questions(
        data_dir / "capability_dimensions.v2.yaml",
        data_dir / "question_templates.v1.yaml",
    )
    ref_top3 = [rm.top3_ids(r["weights"]) for r in rows]

    model, tok, device = srv.load_model(args.adapter, args.base)
    results = []
    weights_by_mode = {"choice": [], "noul": [], "score": []}

    for i, r in enumerate(rows):
        query = r["query"]
        gold_t3 = ref_top3[i]
        answers = {}
        for mode in ["choice", "noul", "score"]:
            qs = questions[mode]
            ids = list(qs.keys())
            contexts, cids_list, meta = [], [], []
            for qid in ids:
                q = qs[qid]
                codes, values, _gold, _soft = bm.question_to_choices(q)
                ctx = bm.pack_context(query, qid, q, codes, values)
                contexts.append(ctx)
                cids_list.append([bm.CODE_TO_ID[c] for c in codes])
                meta.append((qid, codes, values))
            probs_list = predict_probs(
                model, tok, contexts, cids_list, device, batch_size=args.batch_size, max_len=args.max_length
            )
            for (qid, codes, values), probs in zip(meta, probs_list):
                prob_map = {v: p for v, p in zip(values, probs)}
                if mode == "noul":
                    answers[qid] = {"type": "noul", "noul": prob_map.get("true", 0.0)}
                else:
                    best = values[int(max(range(len(values)), key=lambda k: probs[k]))]
                    answers[qid] = {
                        "type": "choice",
                        "choice": best,
                        "probabilities": {v: round(p, 6) for v, p in prob_map.items()},
                        "confidence": float(max(probs)),
                    }

        row_out = {"idx": i, "query": query[:200], "gold_top3": gold_t3, "answers_n": len(answers)}
        for mode in ["choice", "noul", "score"]:
            strengths = rm.compute_strengths(mode, answers, questions)
            weights = rm.normalize(strengths)
            pred_t3 = rm.top3_ids(weights)
            ov = len(set(pred_t3) & set(gold_t3))
            weights_by_mode[mode].append(weights)
            row_out[f"weights_{mode}"] = weights
            row_out[f"top3_{mode}"] = pred_t3
            row_out[f"ov_{mode}"] = ov
        results.append(row_out)
        print(
            f"[{i+1}/{len(rows)}] ov choice/noul/score="
            f"{row_out['ov_choice']}/{row_out['ov_noul']}/{row_out['ov_score']}",
            flush=True,
        )

    consistency = {
        mode: rm.consistency_for_mode(weights_by_mode[mode], ref_top3)
        for mode in ("choice", "noul", "score")
    }
    composite = rm.composite_score_9(consistency)
    summary = {
        "suite": "routing12",
        "n": len(rows),
        "consistency": {
            mode: {
                "top1_exact": consistency[mode]["top1_exact"],
                "top1_in_ref3": consistency[mode]["top1_in_ref3"],
                "avg_top3_overlap": consistency[mode]["avg_top3_overlap"],
            }
            for mode in consistency
        },
        "composite_9": composite,
        # 兼容旧字段
        "mean_top3_overlap": {
            m: consistency[m]["avg_top3_overlap"] for m in consistency
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
