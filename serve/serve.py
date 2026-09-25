#!/usr/bin/env python3
"""u2_decision_4b 独立推理服务（单文件）。

依赖（pip）：
  torch, transformers, peft, fastapi, uvicorn, pydantic

模型文件：
  --adapter  LoRA 目录（默认仓库根目录 ./adapter）
  --base     Qwen3.5-4B Instruct 权重目录或 HF id

协议：POST /v1/systemone（对齐 Kev）
其它：GET /health 、 GET /v1/models

用法：
  python serve/serve.py --serve --port 8000 --base /path/to/Qwen3.5-4B
  python serve/serve.py --input in.jsonl --output out.jsonl --base ...
"""
from __future__ import annotations

import argparse
import json
import string
import time
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# 协议辅助（自包含；与训练 pack 格式保持一致）
# ---------------------------------------------------------------------------

# Qwen tokenizer 中字母 A..Z 的 token id（与训练 CODE_TO_ID 一致）
CODE_TO_ID: dict[str, int] = {c: 32 + i for i, c in enumerate(string.ascii_uppercase[:26])}


def state_text(state: Any) -> str:
    """state 允许 str / list / dict（Kev 风格对话数组需序列化）。"""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def infer_type(q: dict) -> str:
    t = q.get("type")
    if t in ("noul", "boolean"):
        return "noul"
    if t == "score":
        return "score"
    if t in ("choice", "enum"):
        return "choice"
    crit = q.get("criteria")
    if isinstance(crit, list):
        return "score"
    lab = q.get("label")
    if isinstance(lab, bool):
        return "noul"
    return "choice"


def question_to_choices(q: dict) -> tuple[list[str], list[str], str]:
    """返回 (codes, option_values, qtype)。推理不依赖 label。"""
    t = infer_type(q)
    if t == "noul":
        # 训练约定：A=true, B=false
        return ["A", "B"], ["true", "false"], "noul"

    if t == "score":
        crit = q.get("criteria") or []
        if isinstance(crit, dict):
            values = [str(k) for k in crit.keys()]
        else:
            values = [str(x) for x in crit]
        if not values:
            values = ["0", "1", "2", "3", "4"]
        if len(values) > 26:
            values = values[:26]
        codes = list(string.ascii_uppercase[: len(values)])
        return codes, values, "score"

    # choice
    crit = q.get("criteria") or {}
    if isinstance(crit, dict) and crit:
        values = [str(k) for k in crit.keys()]
    else:
        values = [str(x) for x in (q.get("options") or ["A", "B"])]
    if len(values) > 26:
        values = values[:26]
    codes = list(string.ascii_uppercase[: len(values)])
    return codes, values, "choice"


def pack_context(state: Any, field_name: str, q: dict, codes: list[str], values: list[str]) -> str:
    """与训练一致的 context 串。"""
    schema_field: dict[str, Any] = {
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
        + "Codes: "
        + ", ".join(f"{c}={v}" for c, v in zip(codes, values))
    )


def build_answer(qtype: str, values: list[str], probs: list[float]) -> dict:
    prob_map = {v: float(p) for v, p in zip(values, probs)}
    if qtype == "noul":
        return {"type": "noul", "noul": float(prob_map.get("true", 0.0))}
    best_i = int(max(range(len(values)), key=lambda k: probs[k]))
    best_val = values[best_i]
    # score / choice 统一用 choice 形态（与 OpenRouter Jev 侧常见用法一致）
    return {
        "type": "choice",
        "choice": best_val,
        "probabilities": {v: round(p, 6) for v, p in prob_map.items()},
        "confidence": float(max(probs)),
    }


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

_model = None
_tok = None
_device = None
_adapter_path = None
_base_path = None


def load_model(adapter: str, base: str):
    global _model, _tok, _device, _adapter_path, _base_path
    if _model is not None:
        return _model, _tok, _device

    # peft + 部分环境的 torchao 冲突规避
    import peft.import_utils as _peft_iu
    import peft.tuners.lora.torchao as _lora_torchao

    _peft_iu.is_torchao_available = lambda: False
    _peft_iu.is_torchao_ge_v0_18_0 = lambda: False
    _lora_torchao.is_torchao_available = lambda: False

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if _tok.pad_token_id is None:
        _tok.pad_token = _tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    base_model.config.use_cache = False
    _model = PeftModel.from_pretrained(base_model, adapter).to(_device).eval()
    _adapter_path = adapter
    _base_path = base
    print(f"[u2_decision_preview] loaded adapter={adapter} base={base} device={_device}", flush=True)
    return _model, _tok, _device


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_answers(
    model,
    tok,
    device,
    state: Any,
    questions: dict,
    batch_size: int = 16,
    max_len: int = 2048,
    max_ctx_chars: int = 8000,
) -> dict:
    """questions → Jev/Kev 风格 answers。不需要 label。"""
    if not questions:
        return {}

    field_meta: list[tuple] = []
    for fname, qraw in questions.items():
        q = dict(qraw or {})
        # 推理丢弃训练专用字段
        q.pop("label", None)
        q.pop("soft_target", None)
        q.pop("src", None)

        codes, values, qtype = question_to_choices(q)
        if not codes or not values:
            continue
        ctx = pack_context(state, fname, q, codes, values)
        if len(ctx) > max_ctx_chars:
            # 超长：截断 state 再试一次
            st = state_text(state)
            cut = max(500, max_ctx_chars // 2)
            ctx = pack_context(st[:cut], fname, q, codes, values)
            if len(ctx) > max_ctx_chars:
                continue
        field_meta.append((fname, qtype, codes, values, ctx))

    if not field_meta:
        return {}

    answers: dict[str, dict] = {}
    for i in range(0, len(field_meta), batch_size):
        chunk = field_meta[i : i + batch_size]
        contexts = [m[4] for m in chunk]
        cids_list = [[CODE_TO_ID[c] for c in m[2]] for m in chunk]
        enc = tok(
            contexts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(
            **{k: enc[k] for k in ("input_ids", "attention_mask")},
            use_cache=False,
            logits_to_keep=1,
        ).logits[:, -1].float()

        for j, (fname, qtype, codes, values, _ctx) in enumerate(chunk):
            lj = logits[j, cids_list[j]]
            probs = torch.softmax(lj, dim=0).cpu().tolist()
            answers[fname] = build_answer(qtype, values, probs)
    return answers


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

try:
    from pydantic import BaseModel, Field

    class SystemOneRequest(BaseModel):
        state: Any = Field(..., description="用户原文：str 或 JSON（对话数组等）")
        questions: dict = Field(..., description="问题集，Kev/Jev 格式")
        model: str | None = Field(None, description="可选；忽略，固定 u2_decision_preview")

except Exception:  # pragma: no cover
    BaseModel = None
    SystemOneRequest = None


def run_server(args):
    from fastapi import FastAPI
    import uvicorn

    model, tok, device = load_model(args.adapter, args.base)
    app = FastAPI(
        title="mix_v5-4B SystemOne",
        version="1.1",
        description="Kev-compatible POST /v1/systemone (candidate-logit LoRA)",
    )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": "u2_decision_preview",
            "device": str(device),
        }

    @app.get("/v1/models")
    def models():
        return {
            "models": [
                {
                    "name": "u2_decision_preview",
                    "description": "安全审核、能力路由、通用决策",
                }
            ]
        }

    @app.post("/v1/systemone")
    def systemone(req: SystemOneRequest):
        t0 = time.perf_counter()
        answers = predict_answers(
            model,
            tok,
            device,
            req.state,
            req.questions,
            batch_size=args.batch_size,
            max_len=args.max_length,
        )
        return {
            "model": "u2_decision_preview",
            "answers": answers,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    print(f"[u2_decision_preview] serving http://{args.host}:{args.port}  POST /v1/systemone", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


# ---------------------------------------------------------------------------
# CLI 批量
# ---------------------------------------------------------------------------

def run_cli(args):
    model, tok, device = load_model(args.adapter, args.base)
    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    print(f"[u2_decision_preview] batch n={len(rows)}", flush=True)
    out_rows = []
    for i, r in enumerate(rows):
        state = r.get("state")
        if state is None and isinstance(r.get("input"), dict):
            state = r["input"].get("state")
        questions = r.get("questions")
        if questions is None and isinstance(r.get("input"), dict):
            questions = r["input"].get("questions")
        answers = predict_answers(
            model,
            tok,
            device,
            state,
            questions or {},
            batch_size=args.batch_size,
            max_len=args.max_length,
        )
        out_rows.append({"id": r.get("id"), "answers": answers})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for o in out_rows:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"[u2_decision_preview] wrote {args.output}", flush=True)


def main():
    here = Path(__file__).resolve().parent
    repo_root = here.parent  # .../u2_decision_4b

    ap = argparse.ArgumentParser(description="u2_decision_4b SystemOne serve")
    ap.add_argument(
        "--adapter",
        default=str(repo_root / "adapter"),
        help="LoRA adapter 目录",
    )
    ap.add_argument(
        "--base",
        default="Qwen/Qwen3.5-4B",
        help="Qwen3.5-4B Instruct 基座（本地目录或 HuggingFace id）",
    )
    ap.add_argument("--input", help="CLI：输入 jsonl")
    ap.add_argument("--output", help="CLI：输出 jsonl")
    ap.add_argument("--serve", action="store_true", help="启动 FastAPI")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=2048)
    args = ap.parse_args()

    if args.serve:
        run_server(args)
        return 0
    if args.input and args.output:
        run_cli(args)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
