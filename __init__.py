import os

import folder_paths

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

_llm_dir = os.path.join(folder_paths.models_dir, "LLM")
os.makedirs(_llm_dir, exist_ok=True)
folder_paths.add_model_folder_path("LLM", _llm_dir)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
