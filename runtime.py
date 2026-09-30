"""Load and run local chat / vision models with transformers."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .discovery import LocalModel

LOGGER = logging.getLogger("ComfyUI-LocalQwen")

VISION_MODEL_TYPES = {
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_omni_moe",
    "qwen2_5_omni",
    "mistral3",
    "llava",
}


@dataclass
class LoadedModel:
    key: str
    model: Any
    processor: Any
    info: LocalModel
    quant: str
    device: str


class ModelCache:
    def __init__(self) -> None:
        self.current: LoadedModel | None = None

    def clear(self) -> None:
        if self.current is None:
            return
        LOGGER.info("Unloading %s", self.current.info.label)
        try:
            del self.current.model
        except Exception:
            pass
        self.current = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        try:
            import comfy.model_management as mm

            mm.soft_empty_cache()
        except Exception:
            pass

    def get(self, key: str) -> LoadedModel | None:
        if self.current is not None and self.current.key == key:
            return self.current
        return None

    def put(self, loaded: LoadedModel) -> LoadedModel:
        if self.current is not None and self.current.key != loaded.key:
            self.clear()
        self.current = loaded
        return loaded


CACHE = ModelCache()


def _torch():
    import torch

    return torch


def _pick_attn() -> str:
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


def _free_vram_bytes() -> int:
    torch = _torch()
    if not torch.cuda.is_available():
        return 0
    free, _total = torch.cuda.mem_get_info()
    return int(free)


def pick_quant(info: LocalModel, requested: str) -> str:
    requested = (requested or "auto").lower()
    if requested != "auto":
        return requested
    torch = _torch()
    if not torch.cuda.is_available():
        return "4bit"
    # Leave headroom for activations and whatever ComfyUI already holds.
    budget = max(_free_vram_bytes() - 6 * 1024 ** 3, int(_free_vram_bytes() * 0.65))
    size = info.size_bytes
    if size <= budget:
        return "bf16"
    if size / 2.0 <= budget:
        return "8bit"
    return "4bit"


def _quant_config(quant: str):
    torch = _torch()
    compute = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if quant not in {"4bit", "8bit"}:
        return None, compute
    from transformers import BitsAndBytesConfig

    if quant == "4bit":
        return (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute,
            ),
            compute,
        )
    return BitsAndBytesConfig(load_in_8bit=True), compute


def _load_processor(path: str):
    from transformers import AutoProcessor, AutoTokenizer

    try:
        return AutoProcessor.from_pretrained(path, trust_remote_code=True)
    except Exception:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)


def _model_class(info: LocalModel):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    model_type = (info.model_type or "").lower()
    if model_type in {"qwen3_omni_moe", "qwen2_5_omni"}:
        try:
            from transformers import Qwen3OmniMoeThinkerForConditionalGeneration

            if model_type == "qwen3_omni_moe":
                return Qwen3OmniMoeThinkerForConditionalGeneration
        except Exception:
            pass
        return AutoModelForImageTextToText
    if info.vision or model_type in VISION_MODEL_TYPES:
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def load_model(info: LocalModel, quant: str) -> LoadedModel:
    quant = pick_quant(info, quant)
    key = f"{info.path}|{quant}"
    cached = CACHE.get(key)
    if cached is not None:
        return cached

    CACHE.clear()
    torch = _torch()
    LOGGER.info("Loading %s (%s, %.1f GB, quant=%s)", info.label, info.model_type, info.size_gb, quant)

    qconf, dtype = _quant_config(quant)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": _pick_attn(),
        "low_cpu_mem_usage": True,
    }
    if qconf is not None:
        kwargs["quantization_config"] = qconf
        kwargs["device_map"] = "auto"
    else:
        kwargs["dtype"] = dtype
        kwargs["device_map"] = "cuda:0" if torch.cuda.is_available() else "cpu"

    cls = _model_class(info)
    try:
        model = cls.from_pretrained(info.path, **kwargs)
    except TypeError:
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = dtype
        model = cls.from_pretrained(info.path, **kwargs)
    except Exception:
        if kwargs.get("attn_implementation") == "flash_attention_2":
            kwargs["attn_implementation"] = "sdpa"
            model = cls.from_pretrained(info.path, **kwargs)
        else:
            raise
    model.eval()
    processor = _load_processor(info.path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return CACHE.put(
        LoadedModel(key=key, model=model, processor=processor, info=info, quant=quant, device=device)
    )


def _content_for_user(text: str, images: list, vision: bool) -> Any:
    if images:
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": text})
        return content
    if vision:
        return [{"type": "text", "text": text}]
    return text


def _apply_chat(processor, messages: list[dict], enable_thinking: bool):
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    func = processor.apply_chat_template
    if enable_thinking:
        try:
            return func(messages, enable_thinking=True, **kwargs)
        except TypeError:
            pass
    return func(messages, **kwargs)


def _move_inputs(inputs, model):
    torch = _torch()
    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    model_dtype = getattr(model, "dtype", None)
    moved = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
            continue
        tensor = value.to(device)
        if model_dtype is not None and tensor.is_floating_point() and not getattr(model, "is_loaded_in_4bit", False) and not getattr(model, "is_loaded_in_8bit", False):
            tensor = tensor.to(model_dtype)
        moved[key] = tensor
    return moved


def _decode_new_tokens(processor, generated, prompt_len: int) -> str:
    new_tokens = generated[0, prompt_len:]
    tokenizer = getattr(processor, "tokenizer", processor)
    text = processor.batch_decode(
        [new_tokens],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def generate(
    info: LocalModel,
    prompt: str,
    images: list | None,
    system_prompt: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 0,
    keep_model_loaded: bool = True,
    quantization: str = "auto",
    enable_thinking: bool = False,
) -> str:
    torch = _torch()
    images = list(images or [])
    if images and not info.vision:
        raise RuntimeError(
            f"模型 {info.label} 是纯文本模型，不能识图。请改选 Qwen3-VL / Omni，或断开 image 输入。"
        )
    if not (prompt or "").strip() and not images:
        raise RuntimeError("请输入 prompt，或连接一张图片。")

    loaded = load_model(info, quantization)
    processor = loaded.processor
    model = loaded.model

    messages: list[dict] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append(
        {
            "role": "user",
            "content": _content_for_user(
                prompt.strip() or "请描述这张图片。",
                images,
                info.vision,
            ),
        }
    )

    if not hasattr(processor, "apply_chat_template"):
        tokenizer = getattr(processor, "tokenizer", processor)
        encoded = tokenizer(prompt, return_tensors="pt")
        inputs = encoded
    else:
        inputs = _apply_chat(processor, messages, enable_thinking=enable_thinking)
    inputs = _move_inputs(inputs, model)
    prompt_len = inputs["input_ids"].shape[-1]

    do_sample = float(temperature) > 0
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
    }
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id
    elif eos_id is not None:
        gen_kwargs["pad_token_id"] = eos_id
    if do_sample:
        gen_kwargs["temperature"] = max(float(temperature), 1e-5)
        gen_kwargs["top_p"] = float(top_p)

    if seed is not None and int(seed) >= 0:
        torch.manual_seed(int(seed) & 0xFFFFFFFF)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed) & 0xFFFFFFFF)

    with torch.inference_mode():
        output = model.generate(**inputs, **gen_kwargs)
    text = _decode_new_tokens(processor, output, prompt_len)

    if not keep_model_loaded:
        CACHE.clear()
    return text


def unload() -> str:
    had = CACHE.current.info.label if CACHE.current is not None else ""
    CACHE.clear()
    return f"unloaded {had}" if had else "no model loaded"
