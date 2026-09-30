"""Discover local Hugging Face chat / vision models on this machine."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SKIP_MODEL_TYPES = {
    "florence2",
    "t5",
    "mt5",
    "umt5",
    "t5encoder",
    "clip",
    "clip_text_model",
    "clip_vision_model",
    "vit",
    "siglip",
    "bert",
    "xlm-roberta",
}

SKIP_ARCH_MARKERS = (
    "T5For",
    "T5Encoder",
    "CLIPModel",
    "CLIPText",
    "CLIPVision",
    "Florence2",
    "ViTFor",
    "BertFor",
)

VISION_MARKERS = (
    "vl",
    "omni",
    "llava",
    "internvl",
    "minicpm-v",
    "mistral3",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_5",
)

NODE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = NODE_DIR / "local_paths.json"

PLACEHOLDER = "(no local chat models found)"


@dataclass(frozen=True)
class LocalModel:
    label: str
    path: str
    model_type: str
    vision: bool
    size_bytes: int

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _has_weights(path: Path) -> bool:
    if any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin")):
        return True
    if any(path.glob("model-*.safetensors")):
        return True
    return False


def _weight_bytes(path: Path) -> int:
    total = 0
    for pattern in ("*.safetensors", "pytorch_model*.bin", "*.bin"):
        for file in path.glob(pattern):
            try:
                total += file.stat().st_size
            except OSError:
                pass
    return total


def _is_chat_model(config: dict) -> bool:
    model_type = str(config.get("model_type") or "").lower()
    if model_type in SKIP_MODEL_TYPES:
        return False
    architectures = " ".join(config.get("architectures") or [])
    if any(marker in architectures for marker in SKIP_ARCH_MARKERS):
        return False
    if "qwen" in model_type or model_type in {
        "llama",
        "mistral",
        "mistral3",
        "gemma",
        "gemma2",
        "gemma3",
        "phi3",
        "phi",
        "qwen3_moe",
        "qwen2_moe",
    }:
        return True
    return any(
        marker in architectures
        for marker in (
            "ForCausalLM",
            "VLForConditionalGeneration",
            "Omni",
            "ForImageTextToText",
            "Mistral3ForConditionalGeneration",
            "LlavaForConditionalGeneration",
        )
    )


def _is_vision(config: dict) -> bool:
    model_type = str(config.get("model_type") or "").lower()
    if any(marker in model_type for marker in VISION_MARKERS):
        return True
    architectures = " ".join(config.get("architectures") or []).lower()
    return any(
        marker in architectures
        for marker in ("vlfor", "omni", "llava", "imagetext", "mistral3forconditional")
    )


def _hf_cache_roots() -> list[Path]:
    roots: list[Path] = []
    env_keys = ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "HUGGINGFACE_CACHE")
    for key in env_keys:
        value = os.environ.get(key, "").strip()
        if value:
            roots.append(Path(value))
    roots.extend(
        [
            Path.home() / "autodl-tmp" / "flux-training-service" / "storage" / "hf-cache",
            Path.home() / "autodl-tmp" / ".cache" / "huggingface",
            Path.home() / ".cache" / "huggingface",
        ]
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = str(root.resolve()) if root.exists() else str(root)
        except OSError:
            resolved = str(root)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(root)
    return unique


def _load_extra_dirs() -> list[Path]:
    dirs: list[Path] = []
    data = _read_json(CONFIG_PATH)
    for item in data.get("extra_model_dirs") or []:
        if item:
            dirs.append(Path(os.path.expanduser(str(item))))
    return dirs


def _comfy_model_dirs() -> list[Path]:
    dirs: list[Path] = []
    try:
        import folder_paths

        models_dir = Path(folder_paths.models_dir)
        dirs.append(models_dir / "LLM")
        try:
            for item in folder_paths.get_folder_paths("LLM"):
                dirs.append(Path(item))
        except Exception:
            pass
        dirs.append(models_dir / "text_encoders")
        dirs.append(models_dir / "diffusers")
    except Exception:
        fallback = Path.home() / "autodl-tmp" / "ComfyUI" / "models"
        dirs.extend([fallback / "LLM", fallback / "text_encoders"])
    return dirs


def _candidate_dirs() -> list[Path]:
    dirs = []
    dirs.extend(_comfy_model_dirs())
    dirs.extend(_load_extra_dirs())
    dirs.extend(_hf_cache_roots())
    unique: list[Path] = []
    seen: set[str] = set()
    for item in dirs:
        try:
            key = str(item.resolve()) if item.exists() else str(item)
        except OSError:
            key = str(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _iter_direct_model_dirs(root: Path) -> Iterable[Path]:
    if not root.exists() or not root.is_dir():
        return
    if (root / "config.json").exists():
        yield root
        return
    try:
        children = list(root.iterdir())
    except OSError:
        return
    for child in children:
        if not child.is_dir() or child.name.startswith("."):
            continue
        if (child / "config.json").exists():
            yield child
            continue
        # One extra level: models/LLM/Qwen/Qwen3-14B
        try:
            grandchildren = list(child.iterdir())
        except OSError:
            continue
        for grandchild in grandchildren:
            if grandchild.is_dir() and (grandchild / "config.json").exists():
                yield grandchild


def _iter_hf_snapshots(root: Path) -> Iterable[Path]:
    hub = root / "hub" if (root / "hub").is_dir() else root
    if not hub.exists():
        return
    try:
        model_dirs = list(hub.glob("models--*"))
    except OSError:
        return
    for model_dir in model_dirs:
        snapshots = model_dir / "snapshots"
        if not snapshots.is_dir():
            continue
        chosen: Path | None = None
        ref = model_dir / "refs" / "main"
        if ref.is_file():
            try:
                snap_id = ref.read_text(encoding="utf-8").strip()
                candidate = snapshots / snap_id
                if candidate.is_dir():
                    chosen = candidate
            except OSError:
                chosen = None
        if chosen is None:
            try:
                snaps = [p for p in snapshots.iterdir() if p.is_dir()]
            except OSError:
                continue
            if not snaps:
                continue
            chosen = max(snaps, key=lambda p: p.stat().st_mtime)
        yield chosen


def _label_for(path: Path, model_type: str, vision: bool, used: set[str]) -> str:
    name = path.name
    if name.startswith("models--"):
        name = name.replace("models--", "").replace("--", "/")
    parent = path.parent
    if parent.name == "snapshots":
        repo = parent.parent.name.replace("models--", "").replace("--", "/")
        name = repo
    kind = "vl" if vision else "text"
    label = f"{name} [{kind}]"
    if label in used:
        label = f"{name} [{kind} | {path.parent.name}]"
    suffix = 2
    base = label
    while label in used:
        label = f"{base} #{suffix}"
        suffix += 1
    used.add(label)
    return label


def discover_models() -> list[LocalModel]:
    found: list[LocalModel] = []
    seen_paths: set[str] = set()
    used_labels: set[str] = set()

    def consider(path: Path) -> None:
        config_file = path / "config.json"
        if not config_file.exists() or not _has_weights(path):
            return
        try:
            resolved = str(path.resolve())
        except OSError:
            resolved = str(path)
        if resolved in seen_paths:
            return
        config = _read_json(config_file)
        if not _is_chat_model(config):
            return
        seen_paths.add(resolved)
        vision = _is_vision(config)
        model_type = str(config.get("model_type") or "unknown")
        found.append(
            LocalModel(
                label=_label_for(path, model_type, vision, used_labels),
                path=resolved,
                model_type=model_type,
                vision=vision,
                size_bytes=_weight_bytes(path),
            )
        )

    for root in _candidate_dirs():
        has_hub = (root / "hub").is_dir() or root.name == "hub"
        has_model_repos = False
        try:
            has_model_repos = next(root.glob("models--*"), None) is not None
        except OSError:
            has_model_repos = False
        if has_hub or has_model_repos:
            for snap in _iter_hf_snapshots(root):
                consider(snap)
        for model_dir in _iter_direct_model_dirs(root):
            consider(model_dir)

    found.sort(key=lambda item: (not item.vision, item.label.lower()))
    return _dedupe(found)


def _repo_key(model: LocalModel) -> tuple[str, int]:
    path = Path(model.path)
    name = path.name.lower()
    if path.parent.name == "snapshots":
        name = path.parent.parent.name.replace("models--", "").split("--")[-1].lower()
    return (name, model.size_bytes)


def _dedupe(models: list[LocalModel]) -> list[LocalModel]:
    preferred: dict[tuple[str, int], LocalModel] = {}
    order: list[tuple[str, int]] = []

    def score(model: LocalModel) -> int:
        path = model.path.replace("\\", "/").lower()
        value = 0
        if "/comfyui/models/" in path:
            value += 3
        if "/llm/" in path:
            value += 1
        if "/snapshots/" in path:
            value -= 1
        return value

    for model in models:
        key = _repo_key(model)
        if key not in preferred:
            preferred[key] = model
            order.append(key)
            continue
        if score(model) > score(preferred[key]):
            preferred[key] = model
    return [preferred[key] for key in order]


def model_choices() -> tuple[list[str], dict[str, LocalModel]]:
    models = discover_models()
    mapping = {item.label: item for item in models}
    labels = list(mapping.keys()) or [PLACEHOLDER]
    return labels, mapping


def preferred_default(labels: list[str]) -> str:
    if not labels or labels[0] == PLACEHOLDER:
        return PLACEHOLDER
    for needle in (
        "Qwen3-VL-8B-Instruct",
        "Qwen3-VL-4B",
        "Qwen3-VL-2B",
        "Qwen3-8B",
        "Qwen3-4B",
    ):
        for label in labels:
            if needle in label:
                return label
    return labels[0]
