---
library_name: transformers
license: apache-2.0
base_model:
- Qwen/Qwen3.5-0.8B
tags:
- decision-making
- multimodal
- structured-prediction
---

# Intern-Decision-0.8B

[Demo](https://huggingface.co/spaces/internlm/intern-decision) | [Model Weights](https://huggingface.co/collections/internlm/intern-decision) | [GitHub](https://github.com/internlm/Intern-Decision)

**Intern-Decision-0.8B** is a multimodal structured decision model fine-tuned from 
**[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)**.
It accepts a shared state, a schema of named questions, and optional images,
and returns an answer distribution for every question in one model forward pass.

## How inference works

1. Preserve the question and option order, and map each question's options to
   single-token symbols `A`, `B`, …, `Z`, `a`, …, `z`, `0`, …, `9`.
2. Render the original system prompt, state, decision schema, and a complete
   assistant JSON skeleton with one `<decision>` placeholder per field. Preserve
   the checkpoint's chat template and empty thinking block.
3. Run one causal Hugging Face forward pass. For the masked-next-token decision
   objective, read logits at the position **immediately before each placeholder**.
4. Take a softmax over only that field's allowed candidate-symbol logits, then
   apply the checkpoint's probability calibration.
5. Map symbols back to the original option values and return typed JSON answers.

This API performs structured candidate scoring. It does not call `generate()` or
sample free-form text. A request can contain multiple fields; no gold answers are
inserted into the prompt. The inference compiler uses only `state`, `questions`,
and optional `images`.

## Benchmark results

| Model | Jevbench-Easy | Jevbench-Original | Jevbench-Hard | Typed Decision | ToolACE | AG News | WildJailBreak | Average | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Jev | 100.00 | 98.61 | 72.07 | 73.35 | 91.29 | 89.57 | 96.29 | 88.74 | 0.358 | 0.095 |
| Laya | 95.83 | 72.22 | 28.83 | 35.95 | 63.87 | 92.84 | 14.84 | 57.77 | 0.804 | 0.246 |
| SemIf | 100.00 | 98.61 | 61.26 | 62.80 | 85.16 | 89.22 | 92.53 | 84.23 | 0.498 | 0.112 |
| Kev | 100.00 | 93.06 | 45.05 | 65.60 | 87.42 | 89.82 | 75.97 | 79.56 | 0.738 | 0.262 |
| JevK5 | 100.00 | 97.22 | 73.87 | 64.50 | 80.97 | 89.13 | 90.45 | 85.16 | 0.366 | 0.047 |
| Intern-Decision-0.8B | 97.92 | 80.56 | 52.25 | 77.35 | 94.52 | 88.61 | 64.48 | 79.38 | 0.530 | 0.066 |
| Intern-Decision-2B | 100.00 | 84.72 | 63.96 | 79.35 | 96.45 | 89.96 | 78.33 | 84.68 | 0.437 | 0.100 |
| Intern-Decision-4B | 100.00 | 98.61 | 73.87 | 80.55 | 96.45 | 90.82 | 89.86 | 90.02 | 0.347 | 0.065 |


## Inference latency

Measured on a single RTX 4090 with the local HF inference path. Values are
per-query end-to-end latency; they are workload and hardware dependent.

| Model | Mean | Median / P50 | P95 |
|---|---:|---:|---:|
| Jev | 109.70 ms | 106.30 ms | 146.70 ms |
| Intern-Decision-0.8B | 33.98 ms | 33.44 ms | 37.50 ms |
| Intern-Decision-2B | 33.28 ms | 33.15 ms | 33.55 ms |
| Intern-Decision-4B | 44.16 ms | 44.03 ms | 44.60 ms |

## Quick start

Use **Python 3.12+**. Install `requirements.txt` in a suitable PyTorch/CUDA
environment, then import `DecisionEngine` from the downloaded model directory:

```bash
pip install -r requirements.txt
```

```python
from inference import DecisionEngine

engine = DecisionEngine(device="cuda")  # Load once; reuse for subsequent requests.
request = {
    "state": "The customer was charged twice and asks for the extra payment back.",
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "billing": "Payments and refunds",
                "delivery": "Shipping and delivery",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "Rate the priority.",
            "criteria": ["Low", "Medium", "High"],
        },
        "refund_requested": {
            "type": "noul",
            "instructions": "Is the customer asking for a refund?",
        },
    },
}
response = engine.predict(request)  # One Python dict in, one response dict out.
print(response["answers"])
```

`predict(request)` accepts one request dictionary per call and returns a
JSON-serializable Jev-compatible response. It does not read request files or mutate
the supplied dictionary. Reuse the engine for each subsequent request.

The engine defaults to the checkpoint next to `inference.py`. To load another
local copy of this same model, use `DecisionEngine(checkpoint="./model-copy")`.
Use the inference module shipped with the selected size so its default calibration
matches. `backend="hf"` is the default and the only implemented backend. The
optional request `model` field does not switch checkpoints; the response `model`
identifies the weights actually loaded by this module.

### Request format

```json
{
  "state": "The customer was charged twice and asks for the extra payment back.",
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this request?",
      "criteria": {
        "billing": "Payments and refunds",
        "delivery": "Shipping and delivery"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "Rate the priority.",
      "criteria": ["Low", "Medium", "High"]
    },
    "refund_requested": {
      "type": "noul",
      "instructions": "Is the customer asking for a refund?"
    }
  }
}
```

- **choice**: `criteria` is an ordered object mapping option values to descriptions.
- **score**: `criteria` is a list (values become `"0"`, `"1"`, …) or an ordered
  object with finite numeric string keys.
- **noul**: a binary decision with options `no`, then `yes`. Optional criteria can
  describe these values using `no`/`yes` or `false`/`true` keys.

Supply 1–16 questions, with up to 62 options per question. Inputs exceeding
`DecisionEngine(max_length=8192)` (default 8192 tokens) are rejected without truncation.

### Images

Set the request dictionary's `images` list in the intended order:

```python
request["images"] = ["images/frame-1.png", "images/frame-2.png"]
response = engine.predict(request)
```

The checkpoint processor handles image resizing and token expansion. Relative
paths are resolved against `DecisionEngine(media_root=".")` (default: the working directory).
Supply up to eight images; image tokens count toward the input length limit.


### Response format

`answers` maps each field name to:

| Field | Meaning |
|---|---|
| `type` | `choice`, `score`, or `noul` |
| `probabilities` | Calibrated distribution over the original option values |
| `confidence` | Maximum candidate probability |
| `decision` | Highest-probability option value; lexical tie-breaking |
| `choice` | Selected value, for choice questions |
| `noul` | Probability of `yes`, for binary questions |
| `score` | Probability-weighted expected numeric value, for score questions |
| `legend` | Score values and their descriptions, for score questions |
| `source` | `local` |

The response follows the Jev envelope: `model`, `answers`, and `usage`.
It also includes `backend`, `timing`, and `calibration` as extension fields.
`usage.output_tokens` and `usage.decision_count` count scored fields, not generated
text tokens. `confidence` for a score question belongs to its most likely category;
the reported expected `score` can lie between categories.

## Calibration

The default temperature is **2.747760550703**. It was fitted separately for this checkpoint
by NLL minimization on 1,728 designated calibration cases, with 1,693 separate
validation cases. Test-suite labels were not used to select the temperature.

The script follows the demo's numerical sequence:

```text
p = softmax(candidate_logits.float())
calibrated_p = softmax(log(p) / T)
```

This is candidate probability calibration, **not a sampling temperature**. It
updates confidence, the `noul` probability, and the expected `score` while
preserving the argmax decision. For uncalibrated candidate probabilities, use
`DecisionEngine(temperature=1)`. A custom temperature must be finite and positive.

## License and acknowledgment

Intern-Decision is derived from the Qwen3.5 series. The original Qwen license is preserved
as [LICENSE-QWEN](LICENSE-QWEN). Retain the license and applicable
upstream notices when redistributing. These weights were modified by decision
tuning, and this release adds the structured inference wrapper and model card.
We thank the Qwen team for the original models and multimodal processor.
