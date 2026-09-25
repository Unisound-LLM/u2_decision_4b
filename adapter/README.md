---
base_model: Qwen/Qwen3.5-4B
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- peft
- decision
- safety
---

# u2_decision_4b adapter

LoRA adapter on **Qwen3.5-4B Instruct** for:

- content safety classification
- capability routing
- general decision (choice / noul / score)

Load with `PeftModel.from_pretrained(base, "adapter")`. See repository root README.
