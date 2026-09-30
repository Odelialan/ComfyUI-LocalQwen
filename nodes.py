"""ComfyUI nodes that call local Qwen / chat models."""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from .discovery import PLACEHOLDER, model_choices, preferred_default
from .presets import DEFAULT_PROMPT, PRESET_NAMES, build_user_text, compose_user_turn

LOGGER = logging.getLogger("ComfyUI-LocalQwen")


def _tensor_to_pils(image) -> list[Image.Image]:
    if image is None:
        return []
    array = image
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    array = np.asarray(array)
    if array.ndim == 3:
        array = array[None, ...]
    pils: list[Image.Image] = []
    for frame in array[:8]:
        if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
            frame = np.transpose(frame, (1, 2, 0))
        pixels = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        if pixels.ndim == 3 and pixels.shape[-1] == 1:
            pixels = pixels[..., 0]
        pils.append(Image.fromarray(pixels))
    return pils


class LocalQwen:
    """Call a local Qwen VL / Instruct model for captioning, prompting, or translation."""

    @classmethod
    def INPUT_TYPES(cls):
        labels, _mapping = model_choices()
        default_model = preferred_default(labels)
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": DEFAULT_PROMPT,
                        "tooltip": "任务指令（如 QWEN_PROMPT）。与 user_input 一起进入带图的 user 轮。",
                    },
                ),
                "model": (labels, {"default": default_model}),
                "task": (PRESET_NAMES, {"default": "custom"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "user_input": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "用户提示词。非空时拼进 user 轮："
                            "「用户输入：…」+ 换行 + prompt，与图片同轮，不走 system。"
                        ),
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "真正的 system 角色（如扩图方向约束）。不要把用户提示词放这里。",
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 16, "max": 8192, "step": 16},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "quantization": (
                    ["auto", "bf16", "8bit", "4bit"],
                    {"default": "auto"},
                ),
                "keep_model_loaded": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "enable_thinking": (
                    "BOOLEAN",
                    {"default": False},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "LocalQwen"
    DESCRIPTION = (
        "调用本机 Qwen3-VL / Qwen3 Instruct 等模型做识图提示词、翻译和其它小任务。"
        "vision 场景：user_input（用户提示词）+ prompt（指令）进入带图的 user 轮；"
        "system_prompt 仅作真正的 system 角色。"
        "把 Hugging Face 目录放到 ComfyUI/models/LLM/ 后刷新节点即可出现。"
    )

    def run(
        self,
        prompt,
        model,
        task,
        image=None,
        user_input="",
        system_prompt="",
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        seed=0,
        quantization="auto",
        keep_model_loaded=True,
        enable_thinking=False,
    ):
        from .runtime import generate

        labels, mapping = model_choices()
        if model == PLACEHOLDER or model not in mapping:
            available = ", ".join(labels)
            raise RuntimeError(
                "没有可用的本地对话模型。"
                f"当前扫描结果: {available}。"
                "请把完整 HF 模型目录放到 ComfyUI/models/LLM/ 后刷新节点。"
            )
        info = mapping[model]
        preset_system, instruction = build_user_text(task, prompt)
        user_in = (user_input or "").strip()
        # When user_input is set, put intent + instruction in the same user turn
        # as the image. Do not stuff user intent into the system role.
        if user_in:
            user_text = compose_user_turn(user_in, instruction)
            system = (system_prompt or "").strip()
        else:
            user_text = instruction
            system = (system_prompt or "").strip() or preset_system
        images = _tensor_to_pils(image)
        LOGGER.info(
            "LocalQwen model=%s task=%s vision=%s images=%s user_input=%s",
            info.label,
            task,
            info.vision,
            len(images),
            bool(user_in),
        )
        text = generate(
            info=info,
            prompt=user_text,
            images=images,
            system_prompt=system,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            seed=int(seed),
            keep_model_loaded=bool(keep_model_loaded),
            quantization=str(quantization),
            enable_thinking=bool(enable_thinking),
        )
        return (text,)


class LocalQwenUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": (["unload"],),
            },
            "optional": {
                "passthrough": ("STRING", {"forceInput": True, "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "LocalQwen"
    OUTPUT_NODE = True
    DESCRIPTION = "卸载 Local Qwen 缓存的模型，把显存还给 ComfyUI。"

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")

    def run(self, trigger, passthrough=""):
        from .runtime import unload

        status = unload()
        LOGGER.info("LocalQwen %s (%s)", status, trigger)
        return (passthrough or status,)


NODE_CLASS_MAPPINGS = {
    "LocalQwen": LocalQwen,
    "LocalQwenUnload": LocalQwenUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LocalQwen": "Local Qwen",
    "LocalQwenUnload": "Local Qwen Unload",
}
