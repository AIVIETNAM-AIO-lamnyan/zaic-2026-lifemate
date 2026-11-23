"""Whole-trajectory SFT for the LifeMate tool-calling agent.

Run from repository root: bash scripts/finetune.sh
"""
from __future__ import annotations

import sys
import json
import random
from pathlib import Path
from typing import Any

from ..lifemate_config import load_config
from .agent import build_system_prompt, build_tools_spec
from .dataset import open_dataset

FUZZY_KEYS = {"content", "message", "query", "keyword", "keywords", "text", "body", "note", "description"}


def get_data(data_root: str | Path, limit: int | None = None):
    dataset = open_dataset(data_root, "train")
    return (dataset.samples[:limit] if limit is not None else dataset.samples), dataset.catalog_tools


def _call(call: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {
        "name": call["name"], "arguments": call["arguments"],
    }}


def to_chat(row: dict[str, Any], catalog: dict[str, dict[str, Any]]):
    """Use the same system prompt, tool schema and user transcripts as inference."""
    context = row.get("context") or {}
    tools = build_tools_spec(context.get("installed_apps", {}), catalog)
    messages = [{"role": "system", "content": build_system_prompt(context)}]
    for turn_index, turn in enumerate(row.get("history") or []):
        if turn.get("role") == "user":
            messages.append({"role": "user", "content": turn.get("audio_text") or ""})
        elif turn.get("role") == "assistant":
            calls, results = turn.get("tool_calls") or [], turn.get("tool_results") or []
            if len(results) > len(calls):
                raise ValueError(f"{row['id']}: invalid history results")
            # History is context; inference represents all calls in one assistant message.
            message = {"role": "assistant", "content": turn.get("text") or ""}
            if calls:
                message["tool_calls"] = [_call(call, f"call_hist_{turn_index}_{i}") for i, call in enumerate(calls)]
            messages.append(message)
            for i, result in enumerate(results):
                value = json.dumps(result, ensure_ascii=False)
                if len(value) > 2000:
                    value = value[:2000] + "...(truncated)"
                messages.append({"role": "tool", "tool_call_id": f"call_hist_{turn_index}_{i}", "content": value})
    messages.append({"role": "user", "content": row.get("audio_text") or ""})
    target_start = len(messages)
    label = row.get("label") or {}
    calls, results = label.get("tool_calls") or [], label.get("tool_results") or []
    if len(calls) != len(results):
        raise ValueError(f"{row['id']}: tool_calls/tool_results mismatch")
    allowed = {tool["function"]["name"] for tool in tools}
    for i, (call, result) in enumerate(zip(calls, results)):
        if call["name"] not in allowed:
            raise ValueError(f"{row['id']}: unavailable tool {call['name']}")
        call_id = f"call_train_{i}"
        messages.append({"role": "assistant", "content": "", "tool_calls": [_call(call, call_id)]})
        if i + 1 < len(calls):
            messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result, ensure_ascii=False)})
    return messages, tools, target_start


def _render(tokenizer, messages, tools, generation, thinking):
    return tokenizer.apply_chat_template(
        messages, tools=tools or None, tokenize=False,
        add_generation_prompt=generation, enable_thinking=thinking,
    )


def encode_trajectory(row, catalog, tokenizer, max_seq_length=16384, fuzzy_weight=0.3, thinking=True):
    """Mask all non-assistant tokens; verify every assistant prefix against inference."""
    if not (row.get("label") or {}).get("tool_calls"):
        return None  # No assistant answer is provided for no-tool examples.
    messages, tools, target_start = to_chat(row, catalog)
    text = _render(tokenizer, messages, tools, False, thinking)
    spans, fuzzy_spans = [], []
    for i, message in enumerate(messages):
        if i < target_start or message["role"] != "assistant":
            continue
        prefix = _render(tokenizer, messages[:i], tools, True, thinking)
        complete = _render(tokenizer, messages[:i + 1], tools, False, thinking)
        if not complete.startswith(prefix) or not text.startswith(complete):
            raise ValueError(f"{row['id']}: training/inference prefix mismatch at turn {i}")
        spans.append((len(prefix), len(complete)))
        for call in message.get("tool_calls") or []:
            arguments = call["function"]["arguments"]
            for key, value in arguments.items():
                if key not in FUZZY_KEYS or not isinstance(value, str) or not value:
                    continue
                marker = f"<parameter={key}>\n"
                marker_pos = text.find(marker, len(prefix), len(complete))
                if marker_pos >= 0:
                    pos = marker_pos + len(marker)
                    if text.startswith(value, pos):
                        fuzzy_spans.append((pos, pos + len(value)))
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if len(ids) > max_seq_length:
        return None  # Never truncate a tool call or its inference prefix.
    labels, weights = [-100] * len(ids), [0.0] * len(ids)
    for i, (start, end) in enumerate(offsets):
        if end > start and any(start >= a and end <= b for a, b in spans):
            labels[i] = ids[i]
            weights[i] = fuzzy_weight if any(start >= a and end <= b for a, b in fuzzy_spans) else 1.0
    if not any(label != -100 for label in labels) or labels[0] != -100:
        raise ValueError(f"{row['id']}: invalid assistant token mask")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels, "loss_weights": weights}


class WeightedLossTrainerMixin:
    """Compute weighted CE for every assistant target in the microbatch."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        import torch
        from torch.nn import functional as F

        labels = inputs.pop("labels")
        weights = inputs.pop("loss_weights")
        # Union positions across rows so one model forward covers the batch.
        # Each row still masks its own non-assistant targets with -100.
        target_mask = labels[:, 1:] != -100
        logit_positions = torch.nonzero(target_mask.any(dim=0), as_tuple=True)[0]
        if logit_positions.numel() == 0:
            raise ValueError("Batch contains no assistant labels")
        outputs = model(**inputs, logits_to_keep=logit_positions)
        logits = outputs.logits
        if logits.shape[:2] != (labels.shape[0], logit_positions.numel()):
            raise RuntimeError("Model did not honor sparse logits_to_keep")
        target_positions = logit_positions + 1
        targets = labels.index_select(1, target_positions)
        token_weights = weights.index_select(1, target_positions)
        denominator = num_items_in_batch if num_items_in_batch is not None else target_mask.sum()
        denominator = torch.as_tensor(denominator, device=logits.device).clamp_min(1)
        loss = logits.new_zeros((), dtype=torch.float32)
        chunk_size = getattr(self.args, "loss_chunk_size", 128)
        for start in range(0, logit_positions.numel(), chunk_size):
            end = start + chunk_size
            chunk = logits[:, start:end].float()
            ce = F.cross_entropy(
                chunk.reshape(-1, chunk.shape[-1]),
                targets[:, start:end].reshape(-1),
                reduction="none",
                ignore_index=-100,
            ).reshape_as(targets[:, start:end])
            loss = loss + (ce * token_weights[:, start:end]).sum() / denominator
        return (loss, outputs) if return_outputs else loss


def run_sft(config: dict[str, Any]) -> None:
    settings = config["finetune"]
    if not 0 < settings["fuzzy_weight"] <= 1:
        raise ValueError("finetune.fuzzy_weight must be in (0, 1]")
    if not 0 < settings["eval_ratio"] < 1:
        raise ValueError("finetune.eval_ratio must be in (0, 1)")
    for key in ("max_seq_length", "lora_r", "lora_alpha", "epochs",
                "early_stopping_patience", "per_device_train_batch_size",
                "gradient_accumulation_steps", "per_device_eval_batch_size", "save_total_limit"):
        if settings[key] <= 0:
            raise ValueError(f"finetune.{key} must be positive")
    if settings["loss_chunk_size"] <= 0:
        raise ValueError("finetune.loss_chunk_size must be positive")
    if settings["learning_rate"] <= 0 or not 0 <= settings["lora_dropout"] < 1:
        raise ValueError("finetune.learning_rate must be positive and lora_dropout in [0, 1)")
    if not settings["lora_target_modules"]:
        raise ValueError("finetune.lora_target_modules must not be empty")
    if settings["sample_limit"] < 0 or settings["warmup_steps"] < 0:
        raise ValueError("finetune.sample_limit and warmup_steps must be nonnegative")
    if settings["quantization"] not in ("4bit", "bf16"):
        raise ValueError("finetune.quantization must be 4bit or bf16")
    if settings["max_seq_length"] > config["vllm"]["max_model_len"]:
        raise ValueError("finetune.max_seq_length exceeds vllm.max_model_len")
    model_config_path = Path(config["paths"]["llm_model"]) / "config.json"
    if settings["quantization"] == "4bit" and model_config_path.is_file():
        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
        if model_config.get("quantization_config"):
            raise ValueError("Checkpoint is already quantized; use finetune.quantization = 'bf16' or an unquantized base model")
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Unsloth fine-tuning requires a CUDA GPU")
    random.seed(settings["seed"])
    set_seed(settings["seed"])
    limit = settings["sample_limit"] or None
    rows, catalog = get_data(config["paths"]["data_root"], limit)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["paths"]["llm_model"],
        max_seq_length=settings["max_seq_length"],
        dtype=None, load_in_4bit=settings["quantization"] == "4bit",
    )
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer  # Qwen3.5 multimodal returns a Processor.
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for offset masking")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model, r=settings["lora_r"], lora_alpha=settings["lora_alpha"],
        lora_dropout=settings["lora_dropout"], bias=settings["lora_bias"],
        target_modules=settings["lora_target_modules"],
        use_gradient_checkpointing=settings["gradient_checkpointing"], random_state=settings["seed"],
    )
    prepared, no_target, too_long = [], 0, 0
    for row in rows:
        item = encode_trajectory(
            row, catalog, tokenizer, settings["max_seq_length"],
            settings["fuzzy_weight"], config["prediction"]["agent_thinking_mode"],
        )
        if item is None:
            if (row.get("label") or {}).get("tool_calls"):
                too_long += 1
            else:
                no_target += 1
        else:
            prepared.append(item)
    if len(prepared) < 2:
        raise ValueError("At least two labeled trajectories are needed")
    print(f"[sft] samples={len(prepared)} no_target={no_target} too_long={too_long}")
    random.Random(settings["seed"]).shuffle(prepared)
    eval_count = max(1, round(len(prepared) * settings["eval_ratio"]))
    train_rows, eval_rows = prepared[eval_count:], prepared[:eval_count]
    if not train_rows:
        raise ValueError("Training split is empty")

    def collate(features):
        width = max(len(item["input_ids"]) for item in features)
        pads = {"input_ids": tokenizer.pad_token_id, "attention_mask": 0, "labels": -100, "loss_weights": 0.0}
        return {key: torch.tensor([item[key] + [pad] * (width - len(item[key])) for item in features])
                for key, pad in pads.items()}

    class WeightedLossTrainer(WeightedLossTrainerMixin, Trainer):
        pass

    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    effective_batch = settings["per_device_train_batch_size"] * settings["gradient_accumulation_steps"]
    steps_per_epoch = max(1, (len(train_rows) + effective_batch - 1) // effective_batch)
    training_args = TrainingArguments(
        output_dir=str(output), num_train_epochs=settings["epochs"],
        per_device_train_batch_size=settings["per_device_train_batch_size"],
        per_device_eval_batch_size=settings["per_device_eval_batch_size"],
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        learning_rate=settings["learning_rate"],
        lr_scheduler_type=settings["lr_scheduler_type"],
        warmup_steps=settings["warmup_steps"],
        optim=settings["optimizer"], bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
        eval_strategy="steps", eval_steps=steps_per_epoch,
        save_strategy="steps", save_steps=steps_per_epoch,
        save_total_limit=settings["save_total_limit"],
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, logging_steps=max(1, steps_per_epoch // 5),
        remove_unused_columns=False, report_to="none", seed=settings["seed"],
    )
    trainer = WeightedLossTrainer(
        model=model, args=training_args,
        train_dataset=Dataset.from_list(train_rows), eval_dataset=Dataset.from_list(eval_rows),
        data_collator=collate,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=settings["early_stopping_patience"])],
    )
    trainer.args.loss_chunk_size = settings["loss_chunk_size"]
    trainer.train(resume_from_checkpoint=settings["resume_from_checkpoint"] or None)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    print(f"[sft] Adapter saved to {output}")


def main() -> None:
    if len(sys.argv) != 1:
        raise SystemExit("Finetune reads config.toml; run without CLI arguments")
    run_sft(load_config())


if __name__ == "__main__":
    main()
