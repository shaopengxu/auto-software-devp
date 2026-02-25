"""
需求文档模块拆分 Agent

三步骤流程：
1. 多模型轮换，连续调用5次生成模块拆分设计文档 requirement_leader_[1-5].md
2. 多模型（doc_reviewer）轮询对每份文档打分（m个model × 5个文档 = m×5次）
3. 统计总分，选出最高分文档，合并建议，调用 LLM 优化，最终生成 requirement_leader.md

使用方式：
    python generate_leader.py
"""

import os
import re
import json
import yaml
from ask_llm import OpenCodeClient
from doc_utils import read_file, read_requirement_docs


# ============================================================
# 配置加载
# ============================================================
def load_config(config_path='agents_config.yaml'):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: 配置文件 {config_path} 不存在")
        return {}

# ============================================================
# 步骤 1: 生成5份模块拆分文档
# ============================================================
def step1_generate(client, doc_writer_models):
    print("\n" + "="*60)
    print("【步骤1】 生成模块拆分文档（共5次，轮换模型）")
    print("="*60)

    # 提前读取需求文档，注入到 prompt（Python 侧读文件，不依赖 LLM 的文件读取能力）
    req_content = read_requirement_docs()

    for i in range(1, 6):
        current_model = doc_writer_models[(i - 1) % len(doc_writer_models)]
        model_display = current_model or "默认模型"
        output_file = f"requirement_leader_{i}.md"

        print(f"\n[{i}/5] 正在生成 {output_file}（模型: {model_display}）...")

        prompt = f"""你是一位经验丰富的系统架构师。请阅读以下需求文档，理解完整的业务全貌后，进行模块拆分设计。

【需求文档】
{req_content}

【拆分原则】
1. 高内聚低耦合：每个模块只负责一组高度相关的业务，模块间依赖最小化；
2. 边界清晰：每个模块需有明确的职责边界，不存在职责重叠；
3. 依赖单向：尽量保证模块间的依赖方向单向，避免循环依赖；
4. 拆分维度：如果业务存在多个维度，优先选择最核心的维度作第一级模块，再按其他维度在模块内细分；也可以直接按AXB多维度拆分到最细粒度（如A1B1、A1B2、A2B1等）；
5. 降低复杂度：拆分结果应能显著降低代码实现复杂度和业务理解复杂度。

【输出格式】
注意：你生成的文档名为 {output_file}，请将内容保存到该文件。文档结构如下：

# 模块拆分设计

## 1. 拆分说明
简要说明选择该拆分方案的理由和核心思路。

## 2. 模块列表
逐一列出每个模块，每个模块包含：
- 模块名称
- 职责范围（该模块负责哪些业务，不负责哪些）
- 核心功能点
- 对外暴露的主要操作/接口概述

## 3. 模块总览（统领章节）
- 模块依赖关系图（文字描述或 ASCII 图）
- 核心业务流：描述主要业务场景下，请求如何在各模块之间流转
- 模块间调用约定：调用方向、数据传递方式

注意：文档第一行注明你是哪个大模型。"""

        client.chat(prompt, model=current_model)
        print(f"[{i}/5] {output_file} 已生成。")

# ============================================================
# 步骤 2: 多模型对每份文档打分，收集得分和建议
# ============================================================
def step2_review(client, doc_reviewer_models):
    """
    返回格式：
    {
      1: {"total_score": int, "suggestions": [str, ...]},
      2: {...},
      ...
      5: {...}
    }
    """
    print("\n" + "="*60)
    print(f"【步骤2】 多模型评审打分（{len(doc_reviewer_models)} 个模型 × 5 份文档 = {len(doc_reviewer_models)*5} 次）")
    print("="*60)

    scores = {i: {"total_score": 0, "suggestions": []} for i in range(1, 6)}

    total_calls = len(doc_reviewer_models) * 5
    call_count = 0

    # 提前读取需求文档，注入到 prompt
    req_content = read_requirement_docs()

    for reviewer_model in doc_reviewer_models:
        reviewer_display = reviewer_model or "默认模型"
        for i in range(1, 6):
            call_count += 1
            doc_name = f"requirement_leader_{i}.md"
            doc_content = read_file(doc_name)
            print(f"\n[{call_count}/{total_calls}] 模型 {reviewer_display} 正在审核 {doc_name}...")

            prompt = f"""你是一位资深系统架构师，请对以下模块拆分设计文档进行严格审核。

【需求文档】
{req_content}

【待审核模块拆分设计文档】（文件名: {doc_name}）
{doc_content}

请从以下维度逐项审核：

【审核维度】
1. 需求覆盖度：拆分的模块是否覆盖了所有需求，有没有遗漏的业务场景；
2. 模块边界合理性：各模块职责是否清晰、边界是否有歧义或重叠；
3. 内聚性：每个模块内部的功能是否高度相关，有无不该放在一起的功能；
4. 耦合性：模块间依赖是否过多、是否存在循环依赖，依赖方向是否合理；
5. 可扩展性：该拆分方案对于未来业务扩展是否友好；
6. 统领章节质量：模块依赖关系图是否清晰、业务流转描述是否完整、调用约定是否合理；
7. 整体可理解性：整份文档是否易于开发者理解和落地实现。

请给出具体的意见和建议（指出问题所在，并提出改进方向），并综合以上维度打分（0～100分）。

返回格式（只返回此 JSON，不要有其他内容）：
{{"suggestions":"详细的意见与建议", "score":"分数"}}"""

            response = client.chat(prompt, model=reviewer_model)

            if response:
                parsed = parse_review_response(response)
                score = parsed.get("score", 0)
                suggestion = parsed.get("suggestions", "")
                scores[i]["total_score"] += score
                if suggestion:
                    scores[i]["suggestions"].append(f"[{reviewer_display}] {suggestion}")
                print(f"  → 得分: {score}，已累计到 {doc_name}")
            else:
                print(f"  → ⚠️ 未获得有效响应，跳过")

    return scores

def parse_review_response(response: str) -> dict:
    """从 LLM 响应中提取 JSON 格式的 score 和 suggestions"""
    # 尝试直接解析
    try:
        # 先尝试找 JSON 块
        match = re.search(r'\{.*?\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            score_raw = data.get("score", 0)
            score = int(str(score_raw).strip())
            return {"score": score, "suggestions": data.get("suggestions", "")}
    except Exception:
        pass

    # 正则从文本中提取分数
    score_match = re.search(r'"score"\s*:\s*"?(\d+)"?', response)
    suggestion_match = re.search(r'"suggestions"\s*:\s*"([^"]*)"', response, re.DOTALL)
    score = int(score_match.group(1)) if score_match else 0
    suggestion = suggestion_match.group(1) if suggestion_match else response.strip()[:500]
    return {"score": score, "suggestions": suggestion}

# ============================================================
# 步骤 3: 统计最高分，合并建议，优化生成最终文档
# ============================================================
def step3_optimize(client, scores, doc_writer_models):
    print("\n" + "="*60)
    print("【步骤3】 统计得分，选出最佳文档并优化")
    print("="*60)

    # 打印得分汇总
    print("\n各文档得分汇总：")
    for i in range(1, 6):
        print(f"  requirement_leader_{i}.md → 总分: {scores[i]['total_score']}")

    # 找出最高分
    best_index = max(range(1, 6), key=lambda i: scores[i]["total_score"])
    best_score = scores[best_index]["total_score"]
    print(f"\n🏆 最高分文档: requirement_leader_{best_index}.md（总分: {best_score}）")

    # 合并所有建议
    all_suggestions = "\n\n".join(scores[best_index]["suggestions"])
    if not all_suggestions.strip():
        all_suggestions = "（无具体建议，请根据需求文档进行通用优化）"

    # 使用 doc_writer 第一个模型来做最终优化
    optimize_model = doc_writer_models[0]
    optimize_model_display = optimize_model or "默认模型"
    print(f"\n正在调用 {optimize_model_display} 对 requirement_leader_{best_index}.md 进行优化...")

    # 读取需求文档和最佳候选文档内容，注入到 prompt
    req_content = read_requirement_docs()
    best_doc_content = read_file(f"requirement_leader_{best_index}.md")

    prompt = f"""你是一位资深系统架构师。

注意：你需要将最终优化后的文档保存为 requirement_leader.md。

以下是当前评分最高的模块拆分设计文档（requirement_leader_{best_index}.md），以及它所基于的需求文档。

【需求文档】
{req_content}

【待优化模块拆分设计文档】（文件名: requirement_leader_{best_index}.md）
{best_doc_content}

【优化要求】
1. 确保优化后的文档覆盖以上所有需求文档中的业务场景；
2. 参考以下审核意见，逐条评估是否合理，合理的采纳，不合理的忽略并说明理由；
3. 保留原文档中已经合理的部分，不要推倒重来；
4. 优化后的文档结构应包含：拆分说明、模块列表（含职责边界和核心功能）、模块总览（统领章节，含依赖关系、业务流转、调用约定）；
5. 确保最终文档清晰、完整、可直接指导开发人员进行详细设计。

【审核意见（来自多位审核者，仅供参考，请自行判断取舍）】
{all_suggestions}

请将最终优化后的文档保存为 requirement_leader.md。"""

    client.chat(prompt, model=optimize_model)
    print("\n✅ 最终文档 requirement_leader.md 已生成！")

# ============================================================
# 主流程
# ============================================================
def main():
    config = load_config()

    doc_writer_models = config.get('agents', {}).get('doc_writer', {}).get('models', [])
    if not doc_writer_models:
        print("警告: 配置文件中没有找到 doc_writer 的 models，使用默认模型")
        doc_writer_models = [None]

    doc_reviewer_models = config.get('agents', {}).get('doc_reviewer', {}).get('models', [])
    if not doc_reviewer_models:
        print("警告: 配置文件中没有找到 doc_reviewer 的 models，使用默认模型")
        doc_reviewer_models = [None]

    print(f"📋 写文档模型: {doc_writer_models}")
    print(f"📋 审核模型: {doc_reviewer_models}")
    print(f"📊 预计打分次数: {len(doc_reviewer_models)} × 5 = {len(doc_reviewer_models) * 5}")

    client = OpenCodeClient()

    # 步骤 1: 生成
    step1_generate(client, doc_writer_models)

    # 步骤 2: 评审打分
    scores = step2_review(client, doc_reviewer_models)

    # 步骤 3: 择优优化
    step3_optimize(client, scores, doc_writer_models)

    print("\n🎉 全流程完成！最终文档已保存为 requirement_leader.md\n")


if __name__ == "__main__":
    main()
