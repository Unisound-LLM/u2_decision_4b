#!/usr/bin/env python3
"""把 general(nimble/kev) + safety 统一成 candidate-logit 训练行，写入 data/mix/。

每条决策字段展开为一行（Nimble 官方同款）：
  context = 序列化后的 state+schema+requested_field 提示，或简化裸 state（safety 兼容）
这里采用 **形态 B 简化**：context 含 schema JSON + Requested field，候选 A/B/C…。
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import string
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mix"
HOLD = ROOT / "data/holdout/hashes.jsonl"

CODE_TO_ID = {c: 32 + i for i, c in enumerate(string.ascii_uppercase[:26])}  # A=32…

_ZW = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff\u2060\u180e]")
_WS = re.compile(r"\s+")


def norm(t: str) -> str:
    return _WS.sub(" ", _ZW.sub("", unicodedata.normalize("NFKC", str(t) if t is not None else ""))).strip()


def th(t: str) -> str:
    return hashlib.sha1(norm(t).encode()).hexdigest()[:16]


def load_hold() -> set[str]:
    hs = set()
    if HOLD.exists():
        for line in open(HOLD):
            hs.add(json.loads(line).get("sha1_16"))
    return hs


def state_text(state) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def infer_type(q: dict) -> str:
    t = q.get("type")
    if t in ("noul", "choice", "score", "boolean", "enum"):
        return "noul" if t in ("noul", "boolean") else ("score" if t == "score" else "choice")
    crit = q.get("criteria")
    lab = q.get("label")
    if isinstance(lab, bool) or (crit is None and "label" in q and isinstance(lab, bool)):
        return "noul"
    if isinstance(crit, list):
        return "score"
    return "choice"


def question_to_choices(q: dict) -> tuple[list[str], list[str], int, list[float] | None]:
    """返回 (codes, option_values, gold_idx, soft_or_None)。"""
    t = infer_type(q)
    if t == "noul":
        codes = ["A", "B"]
        values = ["true", "false"]
        lab = q.get("label")
        # label may be bool
        if lab is True or lab == "true" or lab == "True":
            gold = 0
        else:
            gold = 1
        return codes, values, gold, None
    if t == "score":
        crit = q.get("criteria") or []
        if isinstance(crit, dict):
            values = list(crit.keys())
        else:
            values = [str(x) for x in crit]
        if not values:
            values = ["0", "1", "2"]
        codes = list(string.ascii_uppercase[: len(values)])
        lab = q.get("label")
        if isinstance(lab, int):
            gold = lab
        else:
            gold = values.index(str(lab)) if str(lab) in values else 0
        return codes, values, gold, None
    # choice
    crit = q.get("criteria") or {}
    if isinstance(crit, dict) and crit:
        values = list(crit.keys())
    else:
        values = list(q.get("options") or ["A", "B"])
    lab = q.get("label")
    if lab in values:
        gold = values.index(lab)
    elif isinstance(lab, int):
        gold = int(lab)
    else:
        gold = 0
    # 候选最多 26；超限时保留 gold + 随机 distractor（对齐 kev/decider）
    if len(values) > 26:
        keep = {gold}
        others = [i for i in range(len(values)) if i != gold]
        random.shuffle(others)
        keep |= set(others[: 25])
        idxs = sorted(keep)
        values = [values[i] for i in idxs]
        gold = idxs.index(gold)
    codes = list(string.ascii_uppercase[: len(values)])
    soft = None
    st = q.get("soft_target")
    if isinstance(st, dict) and st:
        soft = [float(st.get(v, 0.0)) for v in values]
        s = sum(soft) or 1.0
        soft = [x / s for x in soft]
    return codes, values, gold, soft


def pack_context(state, field_name: str, q: dict, codes: list[str], values: list[str]) -> str:
    schema_field = {
        "description": q.get("instructions") or field_name,
        "choices": values,
    }
    if isinstance(q.get("criteria"), dict):
        schema_field["choice_descriptions"] = q["criteria"]
    body = {
        "context": state_text(state),
        "schema": {field_name: schema_field},
    }
    return (
        json.dumps(body, ensure_ascii=False)
        + "\n\nClassify using the schema. "
        + f"Requested field: {field_name}\n"
        + "Return only the one-letter code.\n"
        + f"Codes: " + ", ".join(f"{c}={v}" for c, v in zip(codes, values))
    )


def expand_record(r: dict, default_w: float = 1.0) -> list[dict]:
    state = r.get("state") or r.get("input", {}).get("state")
    questions = r.get("questions")
    if questions is None and "input" in r:
        questions = r["input"].get("questions")
    if not questions:
        return []
    # nimble reference target
    ref = r.get("reference") or {}
    rows = []
    for fname, q in questions.items():
        q = dict(q)
        # nimble: label often in reference.target for single decision field
        if "label" not in q and ref.get("target") is not None and fname in ("decision", list(questions)[0]):
            q["label"] = ref["target"]
            # criteria may embed options
            if q.get("type") == "choice" and isinstance(q.get("criteria"), dict):
                pass
        codes, values, gold, soft = question_to_choices(q)
        if gold < 0 or gold >= len(codes):
            continue
        if soft is None:
            soft = [0.0] * len(codes)
            soft[gold] = 1.0
        ctx = pack_context(state, fname, q, codes, values)
        if len(ctx) > 8000:
            continue
        rows.append(
            {
                "context": ctx,
                "label": values[gold] if gold < len(values) else codes[gold],
                "choices": codes,
                "candidate_ids": [CODE_TO_ID[c] for c in codes],
                "gold": gold,
                "soft_target": [round(x, 6) for x in soft],
                "loss_weight": float(r.get("loss_weight", default_w)),
                "kind": q.get("type") or "choice",
                "text_hash": th(ctx),
                "source": r.get("source") or "unknown",
                "domain": r.get("domain") or r.get("primary_src") or "general",
                "field": fname,
                "id": f"{r.get('id', 'x')}:{fname}",
            }
        )
    return rows


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    random.seed(17)
    hold = load_hold()
    OUT.mkdir(parents=True, exist_ok=True)

    buckets = [
        (ROOT / "data/general/nimble/train.jsonl", 1.2, "nimble"),
        (ROOT / "data/general/kev/train.jsonl", 1.0, "kev"),
        (ROOT / "data/safety/train.jsonl", 1.0, "safety"),
    ]
    train, seen = [], set()
    stats = Counter()
    for path, w, tag in buckets:
        rows = read_jsonl(path)
        print(f"load {tag}: {len(rows)} from {path}")
        for r in rows:
            for ex in expand_record(r, default_w=w):
                if ex["text_hash"] in hold or ex["text_hash"] in seen:
                    stats[f"drop_{tag}"] += 1
                    continue
                # also block if underlying state hash in hold
                st = r.get("state") or (r.get("input") or {}).get("state")
                if st is not None and th(state_text(st)) in hold:
                    stats[f"drop_state_{tag}"] += 1
                    continue
                seen.add(ex["text_hash"])
                train.append(ex)
                stats[f"keep_{tag}"] += 1

    random.shuffle(train)

    valid = []
    for path, w, tag in [
        (ROOT / "data/safety/valid.jsonl", 1.0, "safety_valid"),
    ]:
        for r in read_jsonl(path):
            for ex in expand_record(r, default_w=w):
                if ex["text_hash"] in seen:
                    continue
                valid.append(ex)
                if len(valid) >= 1000:
                    break
            if len(valid) >= 1000:
                break

    # 若 nimble 有部分可切 valid：从 train 抽 5% 作 valid 补充
    if len(valid) < 400 and train:
        n_v = min(400, len(train) // 20)
        valid.extend(train[:n_v])
        train = train[n_v:]

    def dump(path, rows):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(OUT / "train.jsonl", train)
    dump(OUT / "valid.jsonl", valid)
    man = {
        "n_train": len(train),
        "n_valid": len(valid),
        "source_dist": dict(Counter(r["source"] for r in train)),
        "kind_dist": dict(Counter(r["kind"] for r in train)),
        "domain_dist": dict(Counter(str(r.get("domain")) for r in train)),
        "stats": dict(stats),
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(man, indent=2, ensure_ascii=False))
    print(json.dumps(man, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
