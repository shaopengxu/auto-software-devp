"""
doc_utils.py —— 文档读取工具函数（共用模块）

供 generate_leader.py、generate_design.py、check_requirement.py 等脚本导入使用。
"""

import os
import glob

# 需求文档目录（各调用方保持一致）
REQUIREMENT_DIR = "requirement"


def read_file(filepath: str) -> str:
    """读取单个文本文件，返回带文件名标头的内容块。"""
    if not os.path.exists(filepath):
        return f"[文件不存在: {filepath}]"
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return f"=== 文件: {filepath} ===\n{content}\n=== 文件结束: {filepath} ===\n"
    except Exception as e:
        return f"[读取文件失败: {filepath}, 原因: {e}]"


def read_requirement_docs() -> str:
    """读取 requirement/ 目录下所有需求文档。

    - .md / .txt 文件：直接读取内容注入 prompt，确保 LLM 一定能读到
    - 其他格式（pdf、docx 等）：仅列出相对路径，由 OpenCode agent 自行读取
    """
    TEXT_EXTS = {".md", ".txt"}
    pattern = os.path.join(REQUIREMENT_DIR, "*")
    files = sorted(f for f in glob.glob(pattern) if os.path.isfile(f))
    if not files:
        return "[未找到需求文档]"

    text_parts = []
    other_paths = []
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in TEXT_EXTS:
            text_parts.append(read_file(f))
        else:
            other_paths.append(f)

    result = "\n".join(text_parts) if text_parts else ""
    if other_paths:
        paths_str = "\n".join(f"- {p}" for p in other_paths)
        result += f"\n\n【以下需求文件为非文本格式，请直接读取】\n{paths_str}"
    return result.strip()


def read_module_design_docs(exclude: list[str] | None = None) -> str:
    """读取所有 design_module_*.md 文档，拼接成带文件名标头的字符串。"""
    pattern = "design_module_*.md"
    files = sorted(glob.glob(pattern))
    if exclude:
        files = [f for f in files if f not in exclude]
    if not files:
        return "[未找到模块设计文档]"
    return "\n".join(read_file(f) for f in files)
