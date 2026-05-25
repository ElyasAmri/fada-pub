import gc
import os
import re
import threading
from typing import Any

import gradio as gr
import spaces
import torch
from peft import PeftModel
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)


HF_TOKEN = os.environ.get("HF_TOKEN") or None
DEFAULT_MODEL_KEY = os.environ.get("DEFAULT_MODEL_KEY", "gemma3-4b")
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are FADA, a fetal ultrasound annotation assistant. "
    "For every request, answer using exactly eight labeled sections and no extra commentary. "
    "Use the normalized annotation style used by the FADA scoring pipeline. "
    "Return this exact structure:\n"
    "Anatomical Structures: <comma-separated canonical structure names only>\n\n"
    "Fetal Orientation: <short canonical orientation phrase>\n\n"
    "Imaging Plane: <short canonical plane phrase>\n\n"
    "Biometric Measurements: <short canonical measurement phrase>\n\n"
    "Gestational Age: <one of 8-13 weeks, 15-20 weeks, 20-25 weeks, 25-30 weeks, 30-35 weeks, 35-38 weeks>\n\n"
    "Image Quality: <Good image quality, Medium image quality, Low image quality, or Good image quality (detailed assessment)>\n\n"
    "Normality Assessment: <short normalized assessment phrase>\n\n"
    "Clinical Recommendations: <short normalized recommendation phrase>\n"
    "Rules: use these exact labels without question numbers; leave one blank line between sections; use canonical labels rather than prose; keep each answer short; "
    "for Anatomical Structures use a comma-separated list only; for Biometric Measurements name parameter types not measurements unless the canonical label itself requires it; "
    "for Gestational Age always output one allowed range; for Clinical Recommendations prefer normalized phrases such as Prenatal routine monitoring, "
    "Remeasure NT in mid-sagittal plane, Follow up with anatomy scan at 18-22 weeks, or Prenatal monitoring with biochemistry. "
    "If uncertain, choose the closest normalized label instead of hedging.",
)

MODEL_OPTIONS: dict[str, dict[str, str]] = {
    "gemma3-4b": {
        "label": "Gemma 3 4B FADA",
        "base_model": "google/gemma-3-4b-it",
        "adapter_repo": "elyasamri/fada-gemma3-4b-adapter",
        "mode": "vision",
    },
    "qwen2-vl-7b": {
        "label": "Qwen2-VL 7B FADA",
        "base_model": "Qwen/Qwen2-VL-7B-Instruct",
        "adapter_repo": "elyasamri/fada-qwen2-vl-7b-adapter",
        "mode": "vision",
    },
    "qwen3-vl-4b": {
        "label": "Qwen3-VL 4B FADA",
        "base_model": "Qwen/Qwen3-VL-4B-Instruct",
        "adapter_repo": "elyasamri/fada-qwen3-vl-4b-adapter",
        "mode": "vision",
    },
    "qwen35-9b": {
        "label": "Qwen3.5 9B FADA",
        "base_model": "Qwen/Qwen3.5-9B",
        "adapter_repo": "elyasamri/fada-qwen35-9b-adapter",
        "mode": "text",
    },
    "qwen25-vl-3b": {
        "label": "Qwen2.5-VL 3B FADA",
        "base_model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "adapter_repo": "elyasamri/fada-qwen25-vl-3b-adapter",
        "mode": "vision",
    },
    "internvl35-2b": {
        "label": "InternVL3.5 2B FADA",
        "base_model": "OpenGVLab/InternVL3_5-2B-HF",
        "adapter_repo": "elyasamri/fada-internvl35-2b-adapter",
        "mode": "vision",
    },
    "gemma4-e2b": {
        "label": "Gemma 4 E2B FADA",
        "base_model": "google/gemma-4-E2B-it",
        "adapter_repo": "elyasamri/gemma-4-e2b-fada-adapter",
        "mode": "vision",
    },
}

QUESTION_LABELS = {
    1: "Anatomical Structures:",
    2: "Fetal Orientation:",
    3: "Imaging Plane:",
    4: "Biometric Measurements:",
    5: "Gestational Age:",
    6: "Image Quality:",
    7: "Normality Assessment:",
    8: "Clinical Recommendations:",
}

_load_lock = threading.Lock()
_loaded_key: str | None = None
_loaded_model: Any = None
_loaded_processor: Any = None


def _quantization_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def _clear_loaded_model() -> None:
    global _loaded_key, _loaded_model, _loaded_processor
    _loaded_key = None
    _loaded_model = None
    _loaded_processor = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_processor(config: dict[str, str]) -> Any:
    if config["mode"] == "text":
        return AutoTokenizer.from_pretrained(
            config["base_model"], token=HF_TOKEN, trust_remote_code=True
        )

    source = config["adapter_repo"] or config["base_model"]
    try:
        return AutoProcessor.from_pretrained(source, token=HF_TOKEN, trust_remote_code=True)
    except Exception:
        return AutoProcessor.from_pretrained(
            config["base_model"], token=HF_TOKEN, trust_remote_code=True
        )


def _load_model(model_key: str) -> tuple[Any, Any, dict[str, str]]:
    global _loaded_key, _loaded_model, _loaded_processor

    if model_key not in MODEL_OPTIONS:
        raise gr.Error(f"Unknown model: {model_key}")

    with _load_lock:
        if _loaded_key == model_key and _loaded_model is not None and _loaded_processor is not None:
            return _loaded_model, _loaded_processor, MODEL_OPTIONS[model_key]

        _clear_loaded_model()
        config = MODEL_OPTIONS[model_key]
        processor = _load_processor(config)
        model_cls = (
            AutoModelForCausalLM
            if config["mode"] == "text"
            else AutoModelForImageTextToText
        )
        base_model = model_cls.from_pretrained(
            config["base_model"],
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16,
            quantization_config=_quantization_config(),
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(
            base_model,
            config["adapter_repo"],
            token=HF_TOKEN,
            is_trainable=False,
        )
        model.eval()

        _loaded_key = model_key
        _loaded_model = model
        _loaded_processor = processor
        return model, processor, config


def _first_model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _strip_repeated_label(text: str, question_index: int) -> str:
    label = QUESTION_LABELS[question_index]
    plain_label = label.rstrip(":")
    numbered_label = f"Q{question_index}: {label}"
    numbered_plain_label = f"Q{question_index}: {plain_label}"
    for prefix in (numbered_label, numbered_plain_label, label, plain_label):
        if text.lower().startswith(prefix.lower()):
            return text[len(prefix):].lstrip(" :")
    return text


def _capitalize_phrase(text: str) -> str:
    text = text.strip().strip(".")
    if not text:
        return text
    return text[0].upper() + text[1:]


def _normalize_q1(answer: str) -> str:
    parts = [part.strip() for part in answer.split(",") if part.strip()]
    if not parts:
        return answer.strip()
    return ", ".join(_capitalize_phrase(part) for part in parts)


def _normalize_q4(answer: str) -> str:
    lower = answer.lower()
    if not answer.strip() or lower in {"n/a", "na", "none"}:
        return "No measurable parameters"
    if "not measurable" in lower and "center" in lower:
        return "NT not measurable (not centered)"
    if "not measurable" in lower and ("mid" in lower or "sagittal" in lower):
        return "NT not measurable (not mid-sagittal)"
    if ("nt" in lower or "nuchal translucency" in lower) and re.search(r"\d", answer):
        return "NT measurement (specific value)"
    if "crl" in lower and "nt" in lower and "nasal" in lower:
        return "CRL, NT, and nasal bone length"
    if "crl" in lower and "nt" in lower:
        return "CRL and NT"
    if "crl" in lower or "crown-rump" in lower:
        return "CRL"
    if "bpd" in lower and "hc" in lower:
        return "BPD and HC"
    if "femur" in lower:
        return "Femur length"
    if "cervical" in lower and "amniotic" in lower:
        return "Cervical length and amniotic fluid"
    if "cervical" in lower and "ac" in lower:
        return "Cervical length and AC"
    if "cervical" in lower:
        return "Cervical length"
    if "amniotic" in lower and "ac" in lower:
        return "Amniotic fluid pocket and AC"
    if "amniotic" in lower:
        return "Amniotic fluid pocket"
    if "angle of progression" in lower:
        return "Angle of progression"
    if "aortic" in lower:
        return "Aortic transverse diameter"
    if "cardiac" in lower or "heart chamber" in lower:
        return "Cardiac chamber dimensions"
    if "cerebell" in lower and "cisterna" in lower and "nuchal fold" in lower:
        return "Cerebellar, cisterna magna, and nuchal fold"
    if "cerebell" in lower and "cisterna" in lower:
        return "Cerebellar and cisterna magna"
    if "lateral ventricle" in lower:
        return "Lateral ventricle dimensions"
    if "nt" in lower or "nuchal translucency" in lower:
        return "NT thickness"
    return _capitalize_phrase(answer)


def _weeks_to_bin(weeks: int) -> str:
    if weeks <= 13:
        return "8-13 weeks"
    if weeks <= 20:
        return "15-20 weeks"
    if weeks <= 25:
        return "20-25 weeks"
    if weeks <= 30:
        return "25-30 weeks"
    if weeks <= 35:
        return "30-35 weeks"
    return "35-38 weeks"


def _normalize_q5(answer: str) -> str:
    lower = answer.lower()
    for option in (
        "8-13 weeks",
        "15-20 weeks",
        "20-25 weeks",
        "25-30 weeks",
        "30-35 weeks",
        "35-38 weeks",
    ):
        if option in lower:
            return option
    match = re.search(r"(\d{1,2})\s*-\s*(\d{1,2})", lower)
    if match:
        midpoint = (int(match.group(1)) + int(match.group(2))) // 2
        return _weeks_to_bin(midpoint)
    match = re.search(r"\b(\d{1,2})\s*(?:weeks?|w)\b", lower)
    if match:
        return _weeks_to_bin(int(match.group(1)))
    return "20-25 weeks"


def _normalize_q6(answer: str) -> str:
    lower = answer.lower()
    if "detailed" in lower:
        return "Good image quality (detailed assessment)"
    if any(word in lower for word in ("low", "poor", "bad", "dark", "shadow", "artifact")):
        return "Low image quality"
    if any(word in lower for word in ("medium", "acceptable", "moderate", "adequate")):
        return "Medium image quality"
    return "Good image quality"


def _normalize_q7(answer: str) -> str:
    lower = answer.lower()
    if "subchor" in lower or "hematoma" in lower:
        return "Subchorionic hematoma"
    if "spina bifida" in lower:
        return "Abnormal spina bifida"
    if "vsd" in lower:
        return "Abnormal VSD"
    if "lateral ventricle" in lower or "ventricle" in lower and "dilat" in lower:
        return "Abnormal dilated lateral ventricle"
    if "cystic hygroma" in lower:
        return "Increased NT (cystic hygroma)"
    if "biochem" in lower or "down syndrome" in lower:
        return "Increased NT (requires biochemistry)"
    if "nt" in lower and "mark" in lower:
        return "Abnormal NT thickening (marked)"
    if "nt" in lower and "mild" in lower:
        return "Abnormal NT thickening (mild)"
    if "nt" in lower and "abnormal" in lower:
        return "Abnormal NT thickening"
    if "placenta previa" in lower:
        return "Normal closed cervix with placenta previa"
    if "low-lying placenta" in lower or "low lying placenta" in lower:
        return "Abnormal low-lying placenta"
    if "closed cervix" in lower:
        return "Normal closed cervix"
    if "intracranial" in lower and "normal" in lower:
        return "Normal intracranial anatomy"
    if "within normal" in lower:
        return "Within normal limits"
    if "normal delivery" in lower and "low" in lower:
        return "Normal for gestational age, guarded prognosis"
    if "normal delivery" in lower or "favorable" in lower:
        return "Normal for gestational age, favorable prognosis"
    if "normal for" in lower or "gestational age" in lower:
        return "Normal for gestational age"
    if "normal" in lower:
        return "Normal"
    return _capitalize_phrase(answer)


def _normalize_q8(answer: str) -> str:
    lower = answer.lower()
    if "biochem" in lower:
        return "Prenatal monitoring with biochemistry"
    if "anatomy scan" in lower or "18-22" in lower:
        return "Follow up with anatomy scan at 18-22 weeks"
    if "remeasure" in lower and "anomal" in lower:
        return "Remeasure NT with complete anomaly scan"
    if "remeasure" in lower or "mid-sagittal" in lower and "nt" in lower:
        return "Remeasure NT in mid-sagittal plane"
    if "clinical assessment" in lower and "follow up" in lower:
        return "Clinical assessment and follow up"
    if "delivery" in lower and "clinical assessment" in lower:
        return "Clinical assessment for delivery mode"
    if "follow up" in lower and ("1 month" in lower or "4 week" in lower):
        return "Follow up after 1 month"
    if "follow up" in lower and "antenatal" in lower:
        return "Follow up with antenatal care"
    if "follow up" in lower and "probe" in lower:
        return "Follow up (use high-power probe)"
    if "follow up" in lower:
        return "Follow up"
    if "probe" in lower or "high power" in lower or "bright" in lower:
        return "Prenatal monitoring (use high-power probe)"
    if "prenatal" in lower and "nt" in lower:
        return "Prenatal monitoring with NT remeasurement"
    if "prenatal" in lower or "perinatal" in lower or "routine monitoring" in lower:
        return "Prenatal routine monitoring"
    return _capitalize_phrase(answer)


def _format_annotation_output(raw_text: str) -> str:
    lines = [line.strip(" -*\t") for line in raw_text.splitlines() if line.strip()]
    parsed: dict[int, str] = {}
    label_to_index = {label.rstrip(":").lower(): index for index, label in QUESTION_LABELS.items()}

    for line in lines:
        match = re.match(r"^Q([1-8])\s*:\s*(.*)$", line, re.IGNORECASE)
        if match:
            question_index = int(match.group(1))
            answer = _strip_repeated_label(match.group(2).strip(), question_index)
            parsed[question_index] = answer
            continue

        match = re.match(r"^([^:]+):\s*(.*)$", line)
        if not match:
            continue
        label = match.group(1).strip().lower()
        question_index = label_to_index.get(label)
        if question_index is None:
            continue
        answer = _strip_repeated_label(match.group(2).strip(), question_index)
        parsed[question_index] = answer

    if not parsed:
        return raw_text.strip()

    for question_index in range(1, 9):
        parsed.setdefault(question_index, "")

    parsed[1] = _normalize_q1(parsed[1])
    parsed[2] = _capitalize_phrase(parsed[2])
    parsed[3] = _capitalize_phrase(parsed[3])
    parsed[4] = _normalize_q4(parsed[4])
    parsed[5] = _normalize_q5(parsed[5])
    parsed[6] = _normalize_q6(parsed[6])
    parsed[7] = _normalize_q7(parsed[7])
    parsed[8] = _normalize_q8(parsed[8])

    return "\n\n".join(
        f"{QUESTION_LABELS[question_index]} {parsed[question_index].strip()}".rstrip()
        for question_index in range(1, 9)
    )


def _apply_chat_template(processor: Any, messages: list[dict[str, Any]]) -> str:
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        parts: list[str] = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                text = " ".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            else:
                text = str(content)
            parts.append(f"{message['role'].title()}: {text}")
        parts.append("Assistant:")
        return "\n\n".join(parts)


def _decode_generated_text(processor: Any, output_ids: torch.Tensor, input_len: int) -> str:
    tokenizer = getattr(processor, "tokenizer", processor)
    raw = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=False)
    for marker in ("<channel|>", "<|assistant|>", "Assistant:"):
        if marker in raw:
            raw = raw.split(marker, 1)[1]
    for marker in ("<turn|>", "<|im_end|>", "<|endoftext|>", "<end_of_turn>"):
        raw = raw.replace(marker, "")
    return _format_annotation_output(raw.strip())


def _prepare_text_only_inputs(processor: Any, prompt: str, device: torch.device) -> dict[str, Any]:
    tokenizer = getattr(processor, "tokenizer", processor)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = _apply_chat_template(tokenizer, messages)
    return tokenizer(text, return_tensors="pt").to(device)


def _prepare_vision_inputs(
    processor: Any,
    image: Image.Image | None,
    prompt: str,
    device: torch.device,
) -> dict[str, Any]:
    user_content: list[dict[str, object]] = []
    processor_kwargs: dict[str, object] = {}

    if image is not None:
        user_content.append({"type": "image"})
        processor_kwargs["images"] = [image.convert("RGB")]

    user_content.append({"type": "text", "text": prompt})
    text = _apply_chat_template(
        processor,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    inputs = processor(text=text, return_tensors="pt", **processor_kwargs)
    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            if key == "pixel_values":
                inputs[key] = value.to(device=device, dtype=torch.bfloat16)
            else:
                inputs[key] = value.to(device=device)
    return inputs


@spaces.GPU(duration=600)
def analyze_image(
    model_key: str,
    image: Image.Image | None,
    question: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    prompt = question.strip()
    if not prompt:
        raise gr.Error("Enter a question or instruction.")

    model, processor, config = _load_model(model_key)
    device = _first_model_device(model)

    if config["mode"] == "text":
        if image is not None:
            prompt = f"{prompt}\n\nNote: an image was attached, but this selected model is text-only."
        inputs = _prepare_text_only_inputs(processor, prompt, device)
    else:
        inputs = _prepare_vision_inputs(processor, image, prompt, device)

    input_len = int(inputs["input_ids"].shape[1])
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=pad_token_id,
        )

    return _decode_generated_text(processor, output_ids, input_len)


with gr.Blocks(title="FADA Multi-Model ZeroGPU") as demo:
    gr.Markdown(
        "# FADA Multi-Model ZeroGPU\n"
        "Hugging Face Space for FADA web chat. Select a fine-tuned adapter-backed model, "
        "upload an ultrasound image when the selected model supports vision, and ask a question."
    )

    with gr.Row():
        model_input = gr.Dropdown(
            choices=[(value["label"], key) for key, value in MODEL_OPTIONS.items()],
            value=DEFAULT_MODEL_KEY if DEFAULT_MODEL_KEY in MODEL_OPTIONS else "gemma3-4b",
            label="Model",
        )
        max_tokens_input = gr.Slider(
            minimum=64,
            maximum=1024,
            value=256,
            step=32,
            label="Max new tokens",
        )
        temperature_input = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.2,
            step=0.1,
            label="Temperature",
        )

    with gr.Row():
        image_input = gr.Image(
            type="pil",
            label="Ultrasound image",
            sources=["upload", "clipboard"],
        )
        response_output = gr.Textbox(label="Response", lines=18)

    question_input = gr.Textbox(
        label="Question",
        lines=3,
        value="Describe the visible anatomy and key observations in this ultrasound image.",
    )

    submit_button = gr.Button("Analyze", variant="primary")
    submit_button.click(
        fn=analyze_image,
        inputs=[
            model_input,
            image_input,
            question_input,
            max_tokens_input,
            temperature_input,
        ],
        outputs=response_output,
    )


if __name__ == "__main__":
    demo.queue(max_size=8).launch()
