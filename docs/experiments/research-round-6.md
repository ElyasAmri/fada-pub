# Research Round 6: GGUF Conversion, Mobile Deployment, LoRA Fairness, Ensembling, Streamlit Integration

Date: 2026-03-20

Previous rounds covered: framework setup, batch sizes, model compatibility, medical VLM papers, Qwen3.5, eval speed, overfitting, ShareGPT format, adapter format, H100 tips, cross-model scoring, paper writing, and mobile deployment overview.

---

## 1. GGUF Conversion for Qwen2.5-VL and Qwen3-VL

### Conversion Process Overview

GGUF conversion of Qwen VL models requires **two separate runs** of `convert_hf_to_gguf.py` -- one for the language model and one for the vision encoder (mmproj file). The vision encoder is NOT embedded in the main GGUF file; it lives in a separate `mmproj-*.gguf` file.

### Exact Commands

**Step 1: Convert the language model**
```bash
python convert_hf_to_gguf.py /path/to/Qwen2.5-VL-7B-Instruct \
    --outfile Qwen2.5-VL-7B.gguf \
    --outtype f16
```

**Step 2: Generate the mmproj (vision encoder) file**
```bash
python convert_hf_to_gguf.py /path/to/Qwen2.5-VL-7B-Instruct \
    --mmproj \
    --outfile mmproj-Qwen2.5-VL-7B-f16.gguf \
    --outtype f16
```

**Step 3: Quantize the language model (optional)**
```bash
./llama-quantize Qwen2.5-VL-7B.gguf Qwen2.5-VL-7B-Q4_K_M.gguf Q4_K_M
```

**Step 4: Run inference with both files**
```bash
llama-mtmd-cli \
    -m Qwen2.5-VL-7B-Q4_K_M.gguf \
    --mmproj mmproj-Qwen2.5-VL-7B-f16.gguf \
    -p "Describe this ultrasound image." \
    --image ./test_image.png
```

### Vision Encoder Handling

- `convert_hf_to_gguf.py` does NOT automatically include vision encoder weights in the main GGUF. You must pass `--mmproj` to generate the vision encoder file separately.
- The mmproj file is typically kept at **f16 precision** -- quantized mmproj files are not well-supported yet (there is an open feature request: ggml-org/llama.cpp#18881).
- The language model can be quantized independently (Q4_K_M, Q8_0, etc.) while keeping the mmproj at f16.
- The `libmtmd` library in llama.cpp handles loading and bridging the two files at inference time.

### Qwen3-VL Specifics

- Qwen3-VL models are registered as `Qwen3VLForConditionalGeneration` in `convert_hf_to_gguf.py`.
- The same two-step process applies: separate language model and mmproj GGUF files.
- Qwen officially publishes pre-converted GGUF files on Hugging Face (e.g., `Qwen/Qwen3-VL-8B-Instruct-GGUF`, `Qwen/Qwen3-VL-4B-Instruct-GGUF`).

### Known Issues with Fine-Tuned Models

**Critical: Fine-tuned Qwen2.5-VL GGUF conversion has known problems.**

1. **"@ symbol" bug (Issue #15870)**: Users report that after merging LoRA into Qwen2.5-VL-32B, converting to GGUF, and serving via `llama-server`, the model outputs sequences of `@` symbols instead of coherent text. The original safetensors model works fine. No definitive fix documented yet.

2. **Missing vision support after conversion (Issue #13723)**: A user fine-tuned Qwen2.5-VL-7B with Unsloth, merged LoRA, converted to GGUF with Q4 quantization, but the resulting model failed to accept image input. The mmproj file must be regenerated from the merged model, not reused from the base model.

3. **Recommended workflow for fine-tuned models**:
   - Merge LoRA adapter into base model using `PeftModel.merge_and_unload()`
   - Save merged model in safetensors format
   - Verify merged model works correctly with transformers before GGUF conversion
   - Run `convert_hf_to_gguf.py` twice (with and without `--mmproj`)
   - Test with llama.cpp before quantizing further

### Quantization Recommendations

| Quantization | Size (7B) | Quality   | Use Case                     |
|-------------|-----------|-----------|------------------------------|
| f16         | ~14 GB    | Lossless  | Reference / desktop          |
| Q8_0        | ~7.5 GB   | Near-f16  | Desktop / server             |
| Q4_K_M      | ~4.5 GB   | Good      | Mobile / edge deployment     |
| Q4_0        | ~4 GB     | Acceptable| Constrained devices          |

The mmproj file should remain at **f16** (~600 MB for 7B models) regardless of language model quantization.

---

## 2. llama.cpp Mobile Apps and SDKs

### iOS Frameworks

| Framework | Type | Status | Multimodal? |
|-----------|------|--------|-------------|
| **llama.cpp Swift Package** | Official SPM | Active, maintained by ggml-org | Yes (via mtmd) |
| **SpeziLLM** | Stanford Spezi ecosystem | Production-quality, XCFramework | No (text-only currently) |
| **LocalLLMClient** | Swift package | Active (May 2025 update) | Supports llama.cpp + MLX backends |
| **SwiftLlama** | Community wrapper | Maintained | Text-only |
| **llama.swift** (alexrozanski) | Early fork | Stale (2023-era) | No |

**Recommended for FADA iOS**: The official llama.cpp Swift Package via SPM is the best option. It tracks upstream llama.cpp closely and supports the `libmtmd` multimodal library needed for vision models. Import via `https://github.com/ggml-org/llama.cpp` as a Swift Package.

### Android Frameworks

| Framework | Type | GPU Support | Multimodal? |
|-----------|------|-------------|-------------|
| **llama.cpp NDK build** | Direct C++ via CMake | Vulkan, OpenCL (Adreno) | Yes |
| **llama.rn** (React Native) | RN binding | CPU, partial GPU | Yes (text + vision) |
| **Cactus** | Cross-platform SDK | Native optimized | Text-focused |
| **MLC LLM** | Separate engine | Vulkan, OpenCL, Metal | Yes |

**Android NDK approach**: Build llama.cpp directly with CMake for Android, linking against the NDK. This gives full control over the mtmd (multimodal) library. Vulkan backend provides GPU acceleration on modern Android devices.

### Cross-Platform (React Native / Flutter)

**React Native -- llama.rn**:
- Repository: `mybigday/llama.rn`
- Actively maintained, supports both text and multimodal GGUF models
- Demonstrated running VLMs on mobile with image input
- Hugging Face published a guide ("LLM Inference on Edge") using llama.rn for React Native deployment

**Flutter -- flutter_llama**:
- Package: `flutter_llama` on pub.dev
- Supports Android, iOS, and macOS
- Wraps llama.cpp C++ code via FFI
- Less mature than llama.rn for multimodal use

**Cactus**:
- Cross-platform SDK with React Native, Flutter, and Kotlin Multiplatform bindings
- Claims sub-50ms time-to-first-token
- Full privacy (no network required)
- More focused on text generation than VLM

### Medical AI on Mobile -- Regulatory Considerations

- **HIPAA compliance**: On-device inference with llama.cpp is inherently privacy-preserving since no data leaves the device. This eliminates data leakage risks for protected health information.
- **FDA SaMD guidance**: The FDA's January 2025 Draft Guidance on AI-Enabled Device Software Functions covers lifecycle management for AI medical devices. An on-device VLM for ultrasound interpretation would likely require 510(k) clearance or De Novo classification.
- **MedAide** (arxiv 2403.00830): Research prototype for on-device medical LLM assistance using quantized models and llama.cpp.
- No production medical imaging apps using llama.cpp were found in the search results -- this remains a research frontier.

### Performance Expectations on Mobile

For a Q4_K_M Qwen2.5-VL-7B (~4.5 GB):
- **iPhone 15 Pro / 16 Pro** (8 GB RAM): Feasible but tight. Expect ~3-5 tokens/sec.
- **iPhone 16 Pro Max** (12 GB RAM): Better headroom. Vision encoder adds ~600 MB for mmproj.
- **Android flagship** (12+ GB RAM, Snapdragon 8 Gen 3): Similar performance with Vulkan backend.
- **Smaller models preferred**: Qwen3-VL-4B or Qwen2.5-VL-3B at Q4_K_M (~2.5 GB) would be more practical for mobile.

---

## 3. Comparing LoRA Fine-Tuned Models Fairly

### Is r=16, alpha=32 Fair Across 2B to 8B Models?

**Short answer**: Mostly yes, with caveats. Recent research shows that rank matters less than expected if LoRA is applied to all linear layers.

### Key Research Findings

**QLoRA paper finding**: When LoRA is applied to **all layers** (attention + MLP), there is very little statistical difference between ranks 8 and 256. The rank becomes largely irrelevant to final performance. This directly applies to your setup (r=16, all linear targets).

**"LoRA Without Regret" (Thinking Machines Lab)**: A rank-32 adapter on a 7B model matched full fine-tuning on datasets up to ~50,000 examples. Beyond that threshold, ranks of 64-128 were needed. Your dataset of ~15,000 training samples is well within the r=16 capacity.

**Layer selection matters more than rank**: When LoRA was restricted to attention-only layers, models underperformed by 5-15% compared to full fine-tuning, even at high ranks (r=64). Adding MLP adapters closed this gap almost entirely. Since your config targets all linear layers, this is already optimal.

### Trainable Parameters at r=16

The number of trainable LoRA parameters scales with the model's hidden dimension, so larger models get proportionally more trainable parameters with the same rank:

| Model | Params | Hidden Dim | Approx LoRA Params (r=16) | % of Model |
|-------|--------|-----------|--------------------------|------------|
| Qwen2-VL-2B | 2.2B | 1536 | ~8M | 0.36% |
| Qwen2.5-VL-3B | 3.7B | 2048 | ~13M | 0.35% |
| Qwen3-VL-4B | 4.4B | 2560 | ~18M | 0.41% |
| Qwen2.5-VL-7B | 8.3B | 3584 | ~34M | 0.41% |
| Qwen3-VL-8B | 8.3B | 4096 | ~40M | 0.48% |

The percentage of trainable parameters remains roughly constant (~0.35-0.48%), which means the comparison is already approximately fair. The larger models naturally get more LoRA parameters because their layers are wider.

### Alpha/Rank Relationship

Your alpha=32 with r=16 gives a scaling factor of alpha/r = 2.0, which aligns with the widely cited "alpha = 2x rank" sweet spot.

**rsLoRA (Rank-Stabilized LoRA)**: Research from Kalajdzievski (arxiv 2312.03732) shows that the standard alpha/r scaling becomes problematic at higher ranks. rsLoRA uses alpha/sqrt(r) instead, which maintains gradient stability. At r=16, the difference between alpha/r and alpha/sqrt(r) is modest (2.0 vs 8.0), but if you were to increase rank, rsLoRA would be important. PEFT supports this via `use_rslora=True`.

### Recommendations for Fair Comparison

1. **Your current config (r=16, alpha=32, all linear) is acceptable for fair comparison** across 2B-8B models. The research shows rank matters less than layer coverage.
2. **Learning rate is the critical hyperparameter**: The optimal LoRA learning rate is ~10x that of full fine-tuning (1e-4 to 5e-4), and this ratio is consistent across model sizes. Ensure all models use the same learning rate.
3. **If you want to be rigorous**: Run a small ablation on one model (e.g., 7B) comparing r=8, r=16, r=32. If r=16 and r=32 score within 1% of each other, the rank is not a confound.
4. **For the paper**: State that "all models were fine-tuned with identical LoRA configuration (r=16, alpha=32, all linear layers)" and cite the QLoRA/LoRA Without Regret findings that rank has minimal impact when all layers are targeted.

---

## 4. Ensemble/Voting Approaches for VLM Outputs

### Direct Research Evidence

**"One LLM is not Enough" (PMC, 2024)**: This study directly tested ensemble methods for medical question answering with multiple LLMs:

- **Boosting-based Weighted Majority Vote**: Assigns variable weights to different LLMs through a boosting algorithm. Achieved 96.21% on PubMedQA, outperforming every individual model.
- **Cluster-based Dynamic Model Selection**: Dynamically selects the most suitable LLM for each query based on question context clustering. Achieved highest accuracy across all three benchmarks tested.
- **Key finding**: Both ensemble methods outperformed individual LLMs across all datasets.

**"LLM Synergy for Ensemble Learning in Medical QA" (JMIR, 2025)**: Extended the ensemble approach, confirming that strategic combination of multiple LLMs leverages their diverse strengths for improved accuracy in medical QA.

### Ensemble Strategies for FADA

Given your 7+ fine-tuned Qwen VL models answering 8 clinical questions per image, several approaches are viable:

**Strategy 1: Per-Question Majority Vote**
```
For each of the 8 questions:
    Run all 7 models on the image
    Take the most common answer
    Confidence = (count of majority answer) / (total models)
```
- Simplest approach. Works well when answers are categorical.
- Our questions are free-text, so this requires an equivalence function (e.g., embedding similarity > 0.85 = same answer).

**Strategy 2: Weighted Voting by Model Accuracy**
```
For each of the 8 questions:
    Weight each model's answer by its GT accuracy on that question type
    Select answer with highest weighted sum
```
- Uses validation set performance to assign weights.
- Can be question-specific (e.g., Model A is best at Q3, Model B at Q7).

**Strategy 3: Best-Model-Per-Question Selection**
```
For each question:
    Select the single model with highest validation accuracy on that question
```
- Not a true ensemble but leverages model specialization.
- Zero additional inference cost at deployment.

**Strategy 4: LLM-as-Judge Aggregation**
```
Run all models, then pass all answers to a strong LLM (e.g., GPT-4o):
    "Given these 7 answers to the question about fetal anatomy,
     synthesize the best answer."
```
- Most sophisticated but adds latency and API cost.
- Can produce better free-text answers than any individual model.

**Strategy 5: Embedding-Space Averaging**
```
For each question:
    Embed all 7 model answers
    Compute centroid in embedding space
    Select the answer closest to centroid
```
- Robust to outlier answers.
- Naturally leverages your existing embedding similarity scoring.

### Self-Evolving VLM Approaches

**EvoQuality (arxiv 2509.25787)**: Generates pseudo-labels by performing pairwise majority voting on a VLM's own outputs. While designed for image quality assessment, the principle applies -- run the same model multiple times with temperature > 0 and vote on outputs. This is a single-model ensemble via stochastic sampling.

### Practical Considerations

- **Inference cost**: Running 7 models per image is 7x the compute. For deployment, Strategy 3 (best-per-question) or Strategy 1 with top-3 models is more practical.
- **Accuracy ceiling**: The medical QA ensemble papers show 2-4% absolute improvement over the best individual model. For your 81.1% baseline, this could push toward 83-85%.
- **Paper value**: Demonstrating ensemble gain adds a contribution even if the per-model gain is modest. It shows the models are learning complementary patterns.
- **Latency-sensitive deployment**: For the Streamlit demo or mobile app, ensembling is impractical. Pick the single best model. Ensembling is best for offline batch evaluation or when accuracy is paramount.

---

## 5. Streamlit App Integration for VLM Demo

### Current App Architecture

The existing Streamlit app (`web/app.py`) uses:
- EfficientNet-B0 classification model (legacy)
- BLIP2-based VQA model (legacy)
- Chat interface with image upload
- Session state management

The app already has the right UI structure (chat + image upload + preview panel) but needs the backend swapped from BLIP2/EfficientNet to a fine-tuned Qwen VL model.

### Integration Options Comparison

| Approach | Latency | VRAM Needed | Setup Complexity | Best For |
|----------|---------|-------------|-----------------|----------|
| **vLLM server + OpenAI API** | Low (~1-3s) | 16-24 GB | Medium | Production demo, shared GPU |
| **transformers direct load** | Medium (~3-10s) | 16-24 GB | Low | Single-user dev/demo |
| **llama.cpp server** | Low (~2-5s) | 4-8 GB (quantized) | Medium | CPU-only / low-VRAM |
| **Ollama** | Low (~2-4s) | 4-8 GB (quantized) | Very low | Quick prototype |

### Recommended Approach: vLLM Server with OpenAI-Compatible API

This is the best option for a professional demo because:
1. Decouples model serving from the web UI
2. Handles concurrent requests
3. Uses the same API format as OpenAI (easy to swap models)
4. vLLM already supports Qwen2.5-VL and Qwen3-VL natively
5. You already have vLLM experience from RCCG cluster work

**Server setup:**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/merged-qwen2.5-vl-7b \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 4096 \
    --trust-remote-code
```

**Streamlit client code pattern:**
```python
import streamlit as st
from openai import OpenAI
import base64
from io import BytesIO

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

def analyze_with_vlm(image, questions):
    # Encode image to base64
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    b64_image = base64.b64encode(buffer.getvalue()).decode()

    responses = {}
    for q in questions:
        completion = client.chat.completions.create(
            model="merged-qwen2.5-vl-7b",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                    {"type": "text", "text": q}
                ]
            }],
            max_tokens=256,
            stream=True  # For streaming display
        )
        # Collect streamed response
        answer = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                answer += chunk.choices[0].delta.content
        responses[q] = answer
    return responses
```

### Alternative: Ollama (Simplest Setup)

If you want the fastest path to a working demo:
```bash
# Import GGUF model into Ollama
ollama create fada-qwen --from /path/to/model.gguf
ollama serve
```

```python
import ollama
response = ollama.chat(
    model="fada-qwen",
    messages=[{
        "role": "user",
        "content": "Describe this ultrasound image.",
        "images": ["./test.png"]
    }]
)
```

Ollama wraps llama.cpp and handles model management, but gives less control over serving parameters.

### Minimal Changes Needed to Current App

The existing `web/components/model_loader.py` loads EfficientNet and BLIP2. To integrate VLM:

1. Add a new `load_vlm_client()` function that initializes the OpenAI client pointing at vLLM
2. Modify `web/components/tabs/analysis_tab.py` to call VLM inference instead of `analyze_ultrasound()` + `generate_response()`
3. Replace the 8-question VQA shortcut buttons with the actual FADA clinical questions
4. Stream responses using `st.write_stream()` for real-time display
5. Keep the existing chat interface -- it already supports image + text messages

### What Medical AI Demos Typically Use

Based on the search results, medical AI demos in 2025-2026 commonly use:
- **Streamlit + vLLM**: Most common for GPU-equipped demos (PyImageSearch tutorial is the canonical reference)
- **Streamlit + Ollama**: For simpler setups or CPU-only machines
- **Gradio**: Alternative to Streamlit, popular in HuggingFace Spaces
- **4-bit quantized transformers**: For 12 GB GPU machines (medical LLM projects on GitHub)

The vLLM approach is preferred for medical demos because it provides consistent latency and can serve multiple concurrent users during presentations.

---

## Sources

### GGUF Conversion
- [llama.cpp convert_hf_to_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)
- [Qwen llama.cpp documentation](https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html)
- [Qwen2.5-VL @ symbol bug (Issue #15870)](https://github.com/ggml-org/llama.cpp/issues/15870)
- [Fine-tuned Qwen2.5-VL deployment issue (Issue #13723)](https://github.com/ggml-org/llama.cpp/issues/13723)
- [mmproj quantization feature request (Issue #18881)](https://github.com/ggml-org/llama.cpp/issues/18881)
- [Qwen3-VL-8B-Instruct-GGUF (HuggingFace)](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF)
- [Run Qwen2-VL on CPU with GGUF (DEV Community)](https://dev.to/mrzaizai2k/run-qwen2-vl-on-cpu-using-gguf-model-llamacpp-bli)
- [Qwen3-VL Unsloth docs](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune)
- [llama.cpp multimodal docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)

### Mobile Deployment
- [llama.rn (React Native binding)](https://github.com/mybigday/llama.rn)
- [flutter_llama](https://pub.dev/packages/flutter_llama)
- [Cactus cross-platform SDK (InfoQ)](https://www.infoq.com/news/2025/12/cactus-on-device-inference/)
- [MLC LLM (Callstack blog)](https://www.callstack.com/blog/want-to-run-llms-on-your-device-meet-mlc)
- [LLM Inference on Edge (HuggingFace blog)](https://huggingface.co/blog/llm-inference-on-edge)
- [VLMs on mobile with llama.cpp (Medium)](https://farmaker47.medium.com/run-gemma-and-vlms-on-mobile-with-llama-cpp-dbb6e1b19a93)
- [llama.cpp iOS guide (Medium)](https://medium.com/@rashadmilton14/using-llama-cpp-on-ios-a-step-by-step-guide-to-local-ai-on-your-iphone-530e821619b6)
- [LocalLLMClient (DEV Community)](https://dev.to/tattn/localllmclient-a-swift-package-for-local-llms-using-llamacpp-and-mlx-1bcp)
- [Self-hosted Llama for regulated industries](https://www.llama.com/docs/deployment/regulated-industry-self-hosting/)
- [FDA AI/ML SaMD guidance](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-software-medical-device)

### LoRA Fairness
- [LoRA Without Regret (Thinking Machines Lab)](https://thinkingmachines.ai/blog/lora/)
- [rsLoRA paper (arxiv 2312.03732)](https://arxiv.org/abs/2312.03732)
- [rsLoRA (HuggingFace blog)](https://huggingface.co/blog/damjan-k/rslora)
- [LoRA vs Full Fine-tuning: An Illusion of Equivalence (arxiv)](https://arxiv.org/html/2410.21228v1)
- [Learning Rate Scaling across LoRA Ranks (arxiv)](https://arxiv.org/html/2602.06204v1)
- [NormAL LoRA: What is the perfect size? (EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.1074.pdf)
- [LoRA-PRO (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/ea184f920a0f0f8d8030aa1bd7ac9fd4-Paper-Conference.pdf)
- [Databricks LoRA guide](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms)
- [Sebastian Raschka LoRA tips](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
- [Unsloth LoRA hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)

### Ensemble/Voting
- [One LLM is not Enough (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10775333/)
- [LLM Synergy for Medical QA (JMIR 2025)](https://www.jmir.org/2025/1/e70080)
- [EvoQuality: Self-Evolving VLMs via Voting (arxiv 2509.25787)](https://arxiv.org/abs/2509.25787)
- [Vision-Language Models in Medical Image Analysis (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1566253525000685)
- [VLM foundation models for medical imaging (Springer)](https://link.springer.com/article/10.1007/s13534-025-00484-6)

### Streamlit Integration
- [Streamlit UI for LLaVA with vLLM (PyImageSearch)](https://pyimagesearch.com/2025/09/29/building-a-streamlit-ui-for-llava-with-openai-api-integration/)
- [vLLM Streamlit chatbot example](https://docs.vllm.ai/en/latest/examples/online_serving/streamlit_openai_chatbot_webserver/)
- [vLLM multimodal OpenAI client example](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_chat_completion_client_for_multimodal.py)
- [vLLM OpenAI-compatible server docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)
- [vLLM vs llama.cpp comparison (Red Hat)](https://developers.redhat.com/articles/2025/09/30/vllm-or-llamacpp-choosing-right-llm-inference-engine-your-use-case)
- [Streamlit + Ollama VLM (Medium)](https://medium.com/@manyi.yim/run-a-vlm-or-a-multimodal-model-with-streamlit-and-ollama-api-2401875460af)
