# u2_decision_4b

Qwen3.5-4B 上的 **LoRA 决策适配器**（candidate-logit），对外模型名 `u2_decision_preview`。

支持能力：

- **安全审核**：文本三分类（safe / unsafe / controversial）
- **能力路由**：任务 → 能力维度权重（Choice / Noul / Score）
- **通用决策**：业务路由选择题 / 是否题 / 分档题（choice / noul / score）

协议对齐 Kev：`POST /v1/systemone`。完整接口说明见 [docs/API.md](docs/API.md)。

## 目录

```
u2_decision_4b/
├── adapter/                 # LoRA 权重（Git LFS）
├── data/
│   ├── safety_holdout500.jsonl          # 安全审核 500（新版 holdout）
│   ├── kev_deepseek_holdout.jsonl       # 通用决策 212（choice/noul/score）
│   └── routing12/                       # 能力路由 12 + 模板
├── serve/                   # HTTP 服务与 curl DEMO
├── scripts/                 # 三套测试集评测脚本
├── docs/API.md
└── requirements.txt
```



## 环境

```bash
pip install -r requirements.txt
# adapter_model.safetensors 较大，克隆需 Git LFS：
#   git lfs install && git clone https://github.com/Unisound-LLM/u2_decision_4b.git
```

基座模型请自行准备 **Qwen3.5-4B Instruct**（本地目录或 Hugging Face id，如 `Qwen/Qwen3.5-4B`）。

## 启动服务

```bash
python serve/serve.py --serve --port 8000 \
  --adapter ./adapter \
  --base /path/to/Qwen3.5-4B
```

健康检查 / 模型列表：

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models
```

冒烟 DEMO（安全 / 决策 / 路由各 1 条）：

```bash
python serve/call_demo.py
# 或
bash serve/curl_demo.sh
```



## 评测三套测试集

```bash
# 1) 安全审核 holdout500（500 条）
python scripts/eval_safety.py --base /path/to/Qwen3.5-4B

# 2) 通用决策 holdout（212 条：choice/noul/score）
python scripts/eval_decision.py --base /path/to/Qwen3.5-4B

# 3) 能力路由 12 条（三模式一致性 + 9 项综合分）
python scripts/eval_routing12.py --base /path/to/Qwen3.5-4B
```

路由口径：每模式报 Top-1 exact / Top-1 in ref Top-3 / avg Top-3 overlap；  
`composite_9` = 对 9 项归一化后取平均（exact÷n、in_ref3÷n、overlap÷3）。

可用 `--n N` 做快速冒烟；结果可加 `--out results/xxx.json`。

## 推理协议（摘要）

`POST /v1/systemone`

```json
{
  "state": "<string | object | array>",
  "questions": {
    "<field>": {
      "type": "choice | noul | score",
      "instructions": "...",
      "criteria": { "optA": "desc" }
    }
  }
}
```

返回 `answers` 中：choice/score → `{choice, probabilities, confidence}`；noul → `{noul: P(true)}`。  
不需要 `label`。详见 [docs/API.md](docs/API.md)。

## 模型说明


| 项   | 值                                        |
| --- | ---------------------------------------- |
| 基座  | Qwen3.5-4B Instruct                      |
| 适配  | LoRA（r=16, alpha=32）                     |
| 推理  | candidate-logit（字母 A/B/C… token softmax） |
| 对外名 | `u2_decision_preview`                    |


本仓库只发布 **adapter**，不含基座权重。

欢迎登陆云知声Maas平台，体验更多模型服务： [https://maas.unisound.com/](https://maas.unisound.com/)

## License

Apache-2.0（见 [LICENSE](LICENSE)）。基座模型遵循其原许可（Qwen）。
