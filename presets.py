"""Task presets for Local Qwen nodes."""

from __future__ import annotations

PRESETS: dict[str, dict[str, str]] = {
    "custom": {
        "label": "custom",
        "system": "",
        "prefix": "",
    },
    "vision_prompt_zh": {
        "label": "vision_prompt_zh",
        "system": "你是一位极具观察力的视觉分析专家。",
        "prefix": (
            "请先判断这张图片的主体类型（是人物、建筑、风景还是静物产品），"
            "然后生成一段高质量的中文提示词，适合用于图像生成。"
            "只输出提示词，不要解释。\n"
        ),
    },
    "vision_prompt_en": {
        "label": "vision_prompt_en",
        "system": "You are an expert image-prompt engineer.",
        "prefix": (
            "Analyze the image and write a high-quality English prompt for image generation. "
            "Include subject, composition, lighting, materials, camera, and style. "
            "Output the prompt only.\n"
        ),
    },
    "caption_zh": {
        "label": "caption_zh",
        "system": "你是严谨的中文图像描述助手。",
        "prefix": "请用中文详细描述这张图片的内容、主体、场景、风格和关键细节。只输出描述。\n",
    },
    "caption_en": {
        "label": "caption_en",
        "system": "You are a precise image captioner.",
        "prefix": "Describe this image in detailed English. Output the caption only.\n",
    },
    "translate_zh": {
        "label": "translate_zh",
        "system": "你是专业翻译。只输出译文，不要解释，不要加引号。",
        "prefix": "请将以下内容翻译成流畅自然的中文：\n\n",
    },
    "translate_en": {
        "label": "translate_en",
        "system": "You are a professional translator. Output only the translation.",
        "prefix": "Translate the following into natural English:\n\n",
    },
    "tags": {
        "label": "tags",
        "system": "You extract concise image tags.",
        "prefix": (
            "Extract comma-separated tags for this image "
            "(subject, appearance, clothing, scene, lighting, style). "
            "Output tags only.\n"
        ),
    },
}

PRESET_NAMES = list(PRESETS.keys())

DEFAULT_PROMPT = (
    "你是一位极具观察力的视觉分析专家。请先**判断**这张图片的主体类型"
    "（是人物、建筑、风景还是静物产品），然后生成一段高质量的中文提示词。"
)


def build_user_text(task: str, prompt: str) -> tuple[str, str]:
    """Return (system_prompt, user_text) for the selected task."""
    preset = PRESETS.get(task) or PRESETS["custom"]
    system = preset.get("system") or ""
    prefix = preset.get("prefix") or ""
    user = (prompt or "").strip()
    if task != "custom" and user:
        user_text = f"{prefix}{user}".strip()
    elif task != "custom":
        user_text = prefix.strip() or user
    else:
        user_text = user
    return system, user_text


def compose_user_turn(user_input: str, instruction: str) -> str:
    """Build the VL user-turn text: 用户输入 first, then the instruction.

    Image tokens are attached separately by the runtime; this is only the text
    block that sits next to the image in the user role.
    """
    user = (user_input or "").strip()
    instruction = (instruction or "").strip()
    if user and instruction:
        return f"用户输入：{user}\n\n{instruction}"
    if user:
        return f"用户输入：{user}"
    return instruction
