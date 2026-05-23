# Fine-tuned adapter registry

This registry groups the recreated PEFT/LoRA adapters used by the current fine-tuned result set. Adapters are stored separately from their base models.

| Adapter | Base model in adapter config | Model class | Approx. adapter size |
| --- | --- | --- | ---: |
| Gemma 3 12B | `unsloth/gemma-3-12b-it-unsloth-bnb-4bit` | `Gemma3ForConditionalGeneration` | 179.0 MB |
| Gemma 3 27B | `unsloth/gemma-3-27b-it-unsloth-bnb-4bit` | `Gemma3ForConditionalGeneration` | 270.7 MB |
| Gemma 3 4B | `unsloth/gemma-3-4b-it-unsloth-bnb-4bit` | `Gemma3ForConditionalGeneration` | 105.4 MB |
| Gemma 4 E4B | `unsloth/gemma-4-e4b-it-unsloth-bnb-4bit` | `Gemma4ForConditionalGeneration` | 109.5 MB |
| InternVL3.5 2B | `OpenGVLab/InternVL3_5-2B-HF` | `InternVLForConditionalGeneration` | 57.8 MB |
| InternVL3.5 8B | `OpenGVLab/InternVL3_5-8B-HF` | `InternVLForConditionalGeneration` | 107.8 MB |
| Mistral Small 3.1 24B | `unsloth/mistral-small-3.1-24b-instruct-2503-unsloth-bnb-4bit` | `Mistral3ForConditionalGeneration` | 210.1 MB |
| PaliGemma 3B | `google/paligemma-3b-pt-448` | `PaliGemmaForConditionalGeneration` | 91.4 MB |
| PaliGemma2 3B | `google/paligemma2-3b-pt-448` | `PaliGemmaForConditionalGeneration` | 89.5 MB |
| Qwen2-VL 2B | `unsloth/qwen2-vl-2b-instruct-unsloth-bnb-4bit` | `Qwen2VLForConditionalGeneration` | 66.2 MB |
| Qwen2-VL 7B | `unsloth/qwen2-vl-7b-instruct-unsloth-bnb-4bit` | `Qwen2VLForConditionalGeneration` | 108.0 MB |
| Qwen2.5-VL 3B | `unsloth/qwen2.5-vl-3b-instruct-unsloth-bnb-4bit` | `Qwen2_5_VLForConditionalGeneration` | 89.4 MB |
| Qwen2.5-VL 7B | `unsloth/qwen2.5-vl-7b-instruct-unsloth-bnb-4bit` | `Qwen2_5_VLForConditionalGeneration` | 109.3 MB |
| Qwen3-VL 4B | `unsloth/qwen3-vl-4b-instruct-unsloth-bnb-4bit` | `Qwen3VLForConditionalGeneration` | 86.0 MB |
| Qwen3-VL 8B | `unsloth/qwen3-vl-8b-instruct-unsloth-bnb-4bit` | `Qwen3VLForConditionalGeneration` | 108.9 MB |
| Qwen3.5 9B | `Qwen/Qwen3.5-9B` | `Qwen3_5ForConditionalGeneration` | 116.4 MB |

## Hugging Face demo adapters

The web demo publishes a subset of these adapters as model repos:

| Web model key | Adapter repo |
| --- | --- |
| `gemma3-4b` | `elyasamri/fada-gemma3-4b-adapter` |
| `qwen2-vl-7b` | `elyasamri/fada-qwen2-vl-7b-adapter` |
| `qwen3-vl-4b` | `elyasamri/fada-qwen3-vl-4b-adapter` |
| `qwen35-9b` | `elyasamri/fada-qwen35-9b-adapter` |
| `qwen25-vl-3b` | `elyasamri/fada-qwen25-vl-3b-adapter` |
| `internvl35-2b` | `elyasamri/fada-internvl35-2b-adapter` |
| `gemma4-e2b` | `elyasamri/gemma-4-e2b-fada-adapter` |
