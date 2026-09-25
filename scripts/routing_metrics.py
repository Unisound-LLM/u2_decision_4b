#!/usr/bin/env python3
"""统一评测 Jev (typesafe/jev-1.13) 能力路由：Choice / Noul / Score 三模式，与 Kev/Nimble 同一口径。

数据:
  - state: test_query_weights.json 的 query（12 个任务）
  - 参考: test_query_weights.json 的 weights（通用模型评分）
  - 问题模板: capability_dimensions.v2.yaml + question_templates.v1.yaml

输出:
  - results/<model>_capability_full_<ts>.jsonl  每条原始响应
  - results/<model>_capability_full_<ts>_summary.json  汇总（含一致性指标）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import random
from datetime import datetime
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent

# ============ 读取配置 ============
def load_yaml(path: Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)

def build_questions(cap_path: Path, tmpl_path: Path) -> dict:
    cap = load_yaml(cap_path)
    tmpl = load_yaml(tmpl_path)
    groups = cap['groups']
    dims = {}
    group_order = []
    for gid, g in groups.items():
        group_order.append(gid)
        for d in g['dimensions']:
            d['group_id'] = gid
            dims[d['id']] = d
    dim_order = list(dims.keys())

    # Choice: 5 道
    choice = {}
    choice['choice__group'] = {
        'type': 'choice',
        'instructions': tmpl['choice']['group']['instructions'],
        'criteria': {gid: groups[gid]['definition_en'] for gid in group_order},
    }
    for gid in group_order:
        qid = f'choice__group__{gid}'
        g_dims = [d for d in dim_order if dims[d]['group_id'] == gid]
        choice[qid] = {
            'type': 'choice',
            'instructions': tmpl['choice']['within_group']['instructions_template'].format(group_id=gid),
            'criteria': {},
        }
        for did in g_dims:
            d = dims[did]
            choice[qid]['criteria'][did] = f"{d['definition_en']} Boundary: {d['boundary_en']}"

    # Noul: 30 道
    noul = {}
    for did in dim_order:
        d = dims[did]
        noul[f'noul__{did}'] = {
            'type': 'noul',
            'instructions': tmpl['noul']['instructions_template'].format(
                dimension_id=did, definition_en=d['definition_en'], boundary_en=d['boundary_en']),
            'criteria': {
                'false': 'The capability is unnecessary or only incidental.',
                'true': 'The capability is required for a core requirement or an important execution step.',
            },
        }

    # Score: 30 道（type=choice，候选 0-4）
    score = {}
    for did in dim_order:
        d = dims[did]
        score[f'score__{did}'] = {
            'type': 'choice',
            'instructions': tmpl['score']['instructions_template'].format(
                dimension_id=did, definition_en=d['definition_en'], boundary_en=d['boundary_en']),
            'criteria': {},
        }
        for lv in tmpl['score']['levels']:
            lid = lv['id']
            score[f'score__{did}']['criteria'][lid] = f"{lv['name_en']}: {lv['description_en']}"

    return {'dim_order': dim_order, 'group_order': group_order, 'choice': choice, 'noul': noul, 'score': score}

# ============ Jev 调用 ============
def call_jev(state: str, questions: dict, api_key: str, model: str, timeout: int = 120) -> dict:
    url = "https://openrouter.ai/api/alpha/decisions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "state": state, "questions": questions}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}", "detail": resp.text[:500]}
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

# ============ 解析 ============
def choice_prob(answer: dict, candidate: str) -> float:
    """从 choice answer 提取某个候选的概率。"""
    if not isinstance(answer, dict):
        return 0.0
    probs = answer.get('probabilities')
    if isinstance(probs, dict):
        return float(probs.get(candidate, 0.0))
    if answer.get('choice') == candidate:
        return 1.0
    return 0.0

def noul_prob(answer: dict) -> float:
    if not isinstance(answer, dict):
        return 0.0
    if 'noul' in answer:
        v = answer['noul']
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        return float(v)
    # 兼容其它格式
    if 'probabilities' in answer:
        probs = answer['probabilities']
        if 'true' in probs:
            return float(probs['true'])
    if 'true' in answer and 'false' in answer:
        return float(answer['true'])
    return 0.0

def score_expectation(answer: dict) -> float:
    """E(level)/4，levels 0-4。"""
    if not isinstance(answer, dict):
        return 0.0
    probs = answer.get('probabilities')
    if isinstance(probs, dict):
        exp = sum(int(lv) * float(probs.get(str(lv), 0.0)) for lv in range(5))
        return exp / 4.0
    ch = answer.get('choice')
    if ch is not None:
        try:
            return int(ch) / 4.0
        except (ValueError, TypeError):
            return 0.0
    return 0.0

# ============ 合成 strengths ============
def compute_strengths(mode: str, answers: dict, questions: dict) -> dict:
    """返回 30 维 strengths。"""
    dim_order = questions['dim_order']
    group_order = questions['group_order']

    if mode == 'noul':
        strengths = {}
        for did in dim_order:
            strengths[did] = noul_prob(answers.get(f'noul__{did}'))
        return strengths

    if mode == 'score':
        strengths = {}
        for did in dim_order:
            strengths[did] = score_expectation(answers.get(f'score__{did}'))
        return strengths

    if mode == 'choice':
        # 从 choice__group 拿 P(group)
        group_prob = {}
        for gid in group_order:
            group_prob[gid] = choice_prob(answers.get('choice__group'), gid)
        # 从 choice__group__{gid} 拿 P(dim|group)
        strengths = {}
        for gid in group_order:
            cond = answers.get(f'choice__group__{gid}', {})
            # 该组维度 = cond 的 probabilities 键
            if isinstance(cond, dict) and isinstance(cond.get('probabilities'), dict):
                g_dims = list(cond['probabilities'].keys())
            else:
                # fallback: 从 criteria 键
                g_dims = list(questions['choice'][f'choice__group__{gid}']['criteria'].keys())
            for did in g_dims:
                strengths[did] = group_prob[gid] * choice_prob(cond, did)
        return strengths

    return {}

def normalize(weights: dict) -> dict:
    total = sum(weights.values())
    if total <= 0:
        n = len(weights)
        return {k: 1.0/n for k in weights}
    return {k: v / total for k, v in weights.items()}


def top3_ids(weights: dict) -> list[str]:
    return [k for k, _ in sorted(weights.items(), key=lambda x: -x[1])[:3]]


def consistency_for_mode(pred_weights_list: list[dict], ref_top3_list: list[list[str]]) -> dict:
    """单模式一致性：Top-1 exact / Top-1 in ref Top-3 / avg Top-3 overlap。"""
    n = len(pred_weights_list)
    if n == 0:
        return {"top1_exact": "0/0", "top1_in_ref3": "0/0", "avg_top3_overlap": 0.0,
                "top1_exact_n": 0, "top1_in_ref3_n": 0, "n": 0}
    top1_exact = 0
    top1_in_ref3 = 0
    overlap3 = []
    for w, ref3 in zip(pred_weights_list, ref_top3_list):
        pred3 = top3_ids(w)
        if pred3[0] == ref3[0]:
            top1_exact += 1
        if pred3[0] in ref3:
            top1_in_ref3 += 1
        overlap3.append(len(set(pred3) & set(ref3)))
    return {
        "top1_exact": f"{top1_exact}/{n}",
        "top1_in_ref3": f"{top1_in_ref3}/{n}",
        "avg_top3_overlap": round(sum(overlap3) / n, 2),
        "top1_exact_n": top1_exact,
        "top1_in_ref3_n": top1_in_ref3,
        "n": n,
    }


def _parse_frac(s) -> float:
    if isinstance(s, str) and "/" in s:
        a, b = s.split("/", 1)
        return int(a) / max(1, int(b))
    return float(s)


def composite_score_9(consistency: dict) -> dict:
    """对 9 项一致性指标归一化后取平均（与 COMPARE_ROUTING12 口径一致）。

    - Top-1 exact / Top-1 in ref Top-3：分子/分母 → [0,1]
    - avg Top-3 overlap：除以 3 → [0,1]
    - 综合分 = 9 项算术平均；并给出各模式 3 项均分
    """
    mode_avgs = {}
    all_vals = []
    for mode in ("choice", "noul", "score"):
        c = consistency.get(mode) or {}
        if not c:
            continue
        n = int(c.get("n") or 0)
        if n <= 0 and "top1_exact" in c:
            # fallback: parse "a/b"
            n = int(str(c["top1_exact"]).split("/")[-1]) if "/" in str(c.get("top1_exact", "")) else 0
        te = _parse_frac(c.get("top1_exact", 0))
        tr = _parse_frac(c.get("top1_in_ref3", 0))
        ov = float(c.get("avg_top3_overlap", 0.0)) / 3.0
        vals = [te, tr, ov]
        mode_avgs[mode] = round(sum(vals) / 3.0, 4)
        all_vals.extend(vals)
    composite = round(sum(all_vals) / len(all_vals), 4) if all_vals else 0.0
    return {
        "composite": composite,
        "by_mode": mode_avgs,
        "formula": "mean of 9: exact/n, in_ref3/n, overlap/3 for choice|noul|score",
    }


# ============ 主流程 ============
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--model', default='typesafe/jev-1.13')
    ap.add_argument('--sleep', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--start-idx', type=int, default=0, help='从第几条开始（用于断点续跑）')
    args = ap.parse_args()

    # API key
    api_key = os.environ.get('OPENROUTER_API_KEY', '')
    if not api_key:
        cur = ROOT
        for _ in range(6):
            cand = cur / '.env'
            if cand.exists():
                for line in cand.read_text().splitlines():
                    if line.startswith('OPENROUTER_API_KEY='):
                        api_key = line.split('=', 1)[1].strip().strip('"').strip("'")
                break
            cur = cur.parent
    if not api_key:
        print('错误: 未找到 OPENROUTER_API_KEY', file=sys.stderr)
        return 2

    # 配置
    cap_path = ROOT / 'capability_dimensions.v2.yaml'
    tmpl_path = ROOT / 'question_templates.v1.yaml'
    questions = build_questions(cap_path, tmpl_path)

    # 数据
    rows = json.load(open(ROOT / 'test_query_weights.json', encoding='utf-8'))
    rows = rows[:args.n]

    # 输出
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model.replace('/', '_')}_capability_full_n{len(rows)}"
    out_jsonl = out_dir / f"{tag}_{ts}.jsonl"
    out_summary = out_dir / f"{tag}_{ts}_summary.json"

    print(f"模型: {args.model}")
    print(f"条数: {len(rows)}，模式: choice/noul/score，输出: {out_jsonl}")
    print()

    results = []
    start = time.time()
    for i in range(args.start_idx, len(rows)):
        r = rows[i]
        state = r['query']
        gold_weights = r['weights']
        rec = {'idx': i, 'task_query': state, 'gold_weights': gold_weights, 'responses': {}}
        ok = True
        for mode in ['choice', 'noul', 'score']:
            resp = call_jev(state, questions[mode], api_key, args.model)
            rec['responses'][mode] = resp
            if 'error' in resp:
                ok = False
                print(f"[{i+1}/{len(rows)}] {mode} ERROR: {resp['error']} {str(resp.get('detail',''))[:80]}")
            time.sleep(args.sleep)
        if ok:
            # 计算三种模式的 weights
            for mode in ['choice', 'noul', 'score']:
                answers = rec['responses'][mode].get('answers', {})
                strengths = compute_strengths(mode, answers, questions)
                rec[f'strengths_{mode}'] = strengths
                rec[f'weights_{mode}'] = normalize(strengths)
            print(f"[{i+1}/{len(rows)}] 完成  cost={sum(rec['responses'][m].get('usage',{}).get('cost',0) for m in ['choice','noul','score']):.6f}")
        else:
            print(f"[{i+1}/{len(rows)}] 部分失败，跳过该条")
        results.append(rec)

        # 增量保存
        with open(out_jsonl, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    elapsed = time.time() - start
    print(f"\n完成 {len(results)} 条，耗时 {elapsed:.1f}s")

    # ===== 汇总 =====
    valid = [r for r in results if all('error' not in r['responses'][m] for m in ['choice','noul','score'])]
    errors = [r for r in results if r not in valid]

    # 参考 top-3（test_query_weights 每条的 top3）；valid 按 results 顺序对齐
    # 注意：valid 子集需与对应 gold 行对齐
    ref_top3_all = [top3_ids(r['weights']) for r in rows]
    # valid 条目带 idx
    ref_top3 = [ref_top3_all[r['idx']] for r in valid]

    summary = {
        'model': args.model,
        'n': len(results),
        'valid': len(valid),
        'errors': len(errors),
        'ts': ts,
        'consistency': {},
        'total_cost_usd': 0.0,
        'elapsed_sec': round(elapsed, 1),
    }
    if valid:
        for mode in ['choice', 'noul', 'score']:
            wl = [r[f'weights_{mode}'] for r in valid]
            summary['consistency'][mode] = consistency_for_mode(wl, ref_top3)
            c = summary['consistency'][mode]
            print(f"{mode}: Top1_exact={c['top1_exact']}  Top1_in_ref3={c['top1_in_ref3']}  avg_top3_overlap={c['avg_top3_overlap']}")
        summary['composite_9'] = composite_score_9(summary['consistency'])
        print(f"composite_9: {summary['composite_9']['composite']}  by_mode={summary['composite_9']['by_mode']}")
        summary['total_cost_usd'] = round(sum(r['responses'][m].get('usage',{}).get('cost',0) for r in valid for m in ['choice','noul','score']), 6)
        print(f"总成本: ${summary['total_cost_usd']}")

    with open(out_summary, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_jsonl}")
    print(f"汇总已保存: {out_summary}")
    return 0

if __name__ == '__main__':
    sys.exit(main())
