"""
Benchmark training script: 100 images, 1 epoch, Qwen2.5-VL-7B with Unsloth.
Designed to compare A100 vs H100 fine-tuning speed.

Usage:
    python bench_train.py --data-root /path/to/Fetal\ Ultrasound --train-data /path/to/gt_train.jsonl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_SAMPLES = 100
NUM_EPOCHS = 1
BATCH_SIZE = 4
MAX_SEQ_LENGTH = 4096


class VLMDataset(Dataset):
    def __init__(self, jsonl_path, processor, max_samples=100, max_length=4096, data_root=None):
        self.processor = processor
        self.max_length = max_length
        self.data_root = Path(data_root) if data_root else None
        self.samples = []
        self._response_template_ids = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        self._end_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line))
        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = sample["messages"]
        image_paths = sample.get("images", [])

        images = []
        for img_path in image_paths:
            if self.data_root and not os.path.isabs(img_path):
                full_path = (self.data_root / img_path).resolve()
                img_path = str(full_path)
            try:
                images.append(Image.open(img_path).convert("RGB"))
            except FileNotFoundError:
                images.append(Image.new("RGB", (224, 224), color="gray"))

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(
            text=text, images=images if images else None,
            return_tensors="pt", padding=False,
            truncation=True, max_length=self.max_length,
        )
        keep_2d = {"image_grid_thw", "video_grid_thw"}
        squeezed = {}
        for k, v in inputs.items():
            if k in keep_2d:
                squeezed[k] = v.squeeze(0) if v.dim() == 3 else v
            else:
                squeezed[k] = v.squeeze(0)
        inputs = squeezed

        input_ids = inputs["input_ids"]
        labels = torch.full_like(input_ids, -100)
        if self._response_template_ids:
            template_len = len(self._response_template_ids)
            ids_list = input_ids.tolist()
            i = 0
            while i <= len(ids_list) - template_len:
                if ids_list[i:i + template_len] == self._response_template_ids:
                    j = i + template_len
                    while j < len(ids_list):
                        labels[j] = input_ids[j]
                        if ids_list[j] == self._end_token_id:
                            break
                        j += 1
                    i = j + 1
                else:
                    i += 1
        inputs["labels"] = labels
        return inputs


def collate_fn(batch):
    concat_keys = {"pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}
    pad_keys = {"input_ids", "attention_mask", "labels"}
    collated = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            if key in concat_keys:
                collated[key] = torch.cat(values, dim=0)
            elif key in pad_keys:
                max_len = max(v.shape[-1] for v in values)
                pad_value = -100 if key == "labels" else 0
                padded = []
                for v in values:
                    pad_size = max_len - v.shape[-1]
                    if pad_size > 0:
                        padded.append(torch.nn.functional.pad(v, (0, pad_size), value=pad_value))
                    else:
                        padded.append(v)
                collated[key] = torch.stack(padded)
            else:
                shapes = {v.shape for v in values}
                collated[key] = torch.stack(values) if len(shapes) == 1 else torch.cat(values, dim=0)
        else:
            collated[key] = values
    return collated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="/tmp/bench_output")
    args = parser.parse_args()

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    print(f"Benchmark: {MODEL_ID}, {MAX_SAMPLES} images, {NUM_EPOCHS} epoch, batch_size={BATCH_SIZE}")

    from unsloth import FastModel

    print("Loading model...")
    t0 = time.time()
    model, processor = FastModel.from_pretrained(
        MODEL_ID, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True, dtype=torch.bfloat16,
    )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    model = FastModel.get_peft_model(
        model, r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear",
    )

    train_dataset = VLMDataset(
        args.train_data, processor, max_samples=MAX_SAMPLES,
        max_length=MAX_SEQ_LENGTH, data_root=args.data_root,
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=True,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        dataloader_num_workers=4,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=processor,
        data_collator=collate_fn,
    )

    torch.cuda.reset_peak_memory_stats()
    print("Starting benchmark training...")
    train_start = time.time()
    trainer.train()
    train_time = time.time() - train_start
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    total_steps = len(trainer.state.log_history) - 1
    final_loss = None
    for entry in reversed(trainer.state.log_history):
        if "loss" in entry:
            final_loss = entry["loss"]
            break

    results = {
        "gpu": gpu_name,
        "gpu_memory_gb": round(gpu_mem, 1),
        "model": MODEL_ID,
        "num_samples": MAX_SAMPLES,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "total_steps": total_steps,
        "model_load_seconds": round(load_time, 1),
        "training_seconds": round(train_time, 1),
        "peak_vram_gb": round(peak_mem, 2),
        "final_loss": final_loss,
        "samples_per_second": round(MAX_SAMPLES * NUM_EPOCHS / train_time, 3),
        "steps_per_second": round(total_steps / train_time, 3) if total_steps > 0 else 0,
    }

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    out_path = Path(args.output_dir) / "bench_results.json"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
