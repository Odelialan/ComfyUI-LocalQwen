# ComfyUI-LocalQwen

在 ComfyUI 中直接调用本机已下载的 Qwen3-VL / Qwen3 Instruct 等本地对话模型，完成识图提示词、图像描述、翻译、标签提取等小任务。无需任何在线 API，模型自动发现、显存自动管理。

作者：**OdeliaLan**（GitHub: [Odelialan](https://github.com/Odelialan)）

## 功能特性

- **本地模型自动发现**：自动扫描 `ComfyUI/models/LLM/`、Hugging Face 缓存目录等位置，找到所有可用的本地对话 / 视觉模型，在下拉框中直接选择，无需手动填写路径。
- **视觉 / 文本双支持**：Qwen3-VL、Qwen2.5-VL、Qwen3-Omni 等视觉模型可接收图片输入；Qwen3 等 Instruct 模型可做纯文本任务。纯文本模型误接图片会明确报错提示。
- **任务预设**：内置 `vision_prompt_zh/en`（识图写提示词）、`caption_zh/en`（图像描述）、`translate_zh/en`（翻译）、`tags`（标签提取）等预设，也可选 `custom` 完全自定义。
- **灵活的对话结构**：`user_input`（用户提示词）与指令拼进带图的 user 轮；`system_prompt` 仅作真正的 system 角色，不混淆。
- **显存友好**：`auto` 量化按空闲显存自动选择 bf16 / 8bit / 4bit；`keep_model_loaded` 控制是否常驻；配套 `Local Qwen Unload` 节点一键卸载显存还给 ComfyUI。
- **可复现采样**：支持 seed、temperature、top_p；`enable_thinking` 控制思考模式输出（自动剥离 `<think>` 段）。

## 安装

把本仓库克隆到 ComfyUI 的 `custom_nodes` 目录：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Odelialan/ComfyUI-LocalQwen.git
```

依赖复用 ComfyUI 自带环境中的 `transformers` / `accelerate` / `bitsandbytes` / `torch`，无需额外安装；如装了 `flash-attn` 会自动优先使用。重启 ComfyUI 后在节点列表搜索 **Local Qwen** 即可。

## 模型准备（外部权重，需自行下载）

本插件不附带任何模型权重，需要把完整的 Hugging Face 模型目录（含 `config.json` 与权重文件）放到本地。推荐下载：

| 模型 | 用途 | 下载地址 |
| --- | --- | --- |
| Qwen3-VL-8B-Instruct（推荐） | 识图 + 文本 | [Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct) |
| Qwen3-VL-4B / 2B | 显存紧张的识图 | [Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct) |
| Qwen3-8B / 4B（Instruct） | 纯文本任务 | [Hugging Face](https://huggingface.co/Qwen/Qwen3-8B) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-8B) |

放置方式（任选其一，重启或刷新节点后生效）：

1. 放到 `ComfyUI/models/LLM/<模型名>/`（推荐，插件启动时会自动创建该目录）。
2. 编辑本插件目录下的 `local_paths.json`，把其它目录加进 `extra_model_dirs`：
   ```json
   {
     "extra_model_dirs": ["/path/to/your/models"]
   }
   ```
3. 若本机已有 Hugging Face 缓存（`HF_HOME`、`~/.cache/huggingface` 等），插件会自动扫描 `models--*/snapshots/` 下的模型，无需复制。

## 节点说明

### Local Qwen

| 输入 | 说明 |
| --- | --- |
| `prompt` | 任务指令（选预设时可叠加在预设之后） |
| `model` | 自动发现的本机模型列表，默认优先选择 Qwen3-VL-8B |
| `task` | 任务预设（识图 / 描述 / 翻译 / 标签 / custom） |
| `image` | 可选图片输入（仅视觉模型可用） |
| `user_input` | 用户提示词，非空时与指令、图片同进 user 轮 |
| `system_prompt` | 真正的 system 角色（如扩图方向约束） |
| `max_new_tokens` / `temperature` / `top_p` / `seed` | 生成参数 |
| `quantization` | `auto` 按空闲显存自动选 bf16/8bit/4bit |
| `keep_model_loaded` | 是否常驻显存，关闭则推理后立即卸载 |
| `enable_thinking` | 思考模式开关（输出自动剥离 `<think>` 段） |

返回 `STRING`（模型输出文本）。

### Local Qwen Unload

执行节点即卸载 Local Qwen 缓存的模型并清空显存缓存；`passthrough` 输入会把文本原样传下去，方便串在工作流末端。

## 版本记录

### v1.0.0（2026-09-30）

首次公开发布。本版本内容：

- 本地模型自动发现（`discovery.py`）：扫描 ComfyUI 模型目录、`local_paths.json` 扩展目录和 Hugging Face 缓存；按架构过滤掉 CLIP / T5 / BERT 等非对话模型；区分视觉 / 文本模型；HF snapshot 去重并优先选择 ComfyUI 目录中的副本。
- 节点实现（`nodes.py`）：`Local Qwen` 主节点 + `Local Qwen Unload` 卸载节点；图片张量转 PIL（自动处理 CHW/HWC 布局）；`user_input` + 指令拼合同一 user 轮、`system_prompt` 独立成 system 角色的对话结构。
- 任务预设（`presets.py`）：中英文识图提示词、图像描述、翻译、标签提取共 7 个预设，及 `compose_user_turn` 用户轮拼接规则。
- 推理运行时（`runtime.py`）：transformers 加载（自动选 flash-attention-2 / sdpa）、按空闲显存自动量化（bf16 / 8bit / 4bit）、单模型缓存与显存释放、思考模式输出剥离、seed 可复现采样。
- 模型路径扫描全部改为基于用户主目录的相对路径，配置文件 `local_paths.json` 提供干净的默认值，仓库内不含任何机器专属路径。

## License

MIT
