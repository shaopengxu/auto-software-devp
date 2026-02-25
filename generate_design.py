"""
总体设计 + 模块详细设计 Agent

流程：
1. 生成5份总体设计文档 design_overall_[1-5].md（轮换 doc_writer 模型）
2. 多模型轮询对5份文档打分
3. 选出最高分，合并建议，优化生成 design_overall.md
4. 从 requirement_leader.md 获取模块列表
5. 按模块依次生成详细设计文档 design_module_${module}.md
6. 对每个模块文档进行多模型审核→优化循环（上限 REVIEW_OPTIMIZE_FACTOR 块模型数次）
7.1 接口对齐（ALIGN_ROUNDS 轮）
7.2-7.4 全局审核→优化循环（上限 REVIEW_OPTIMIZE_FACTOR 块模型数次）

使用方式：
    python generate_design.py
"""

import os
import re
import sys
import json
import time
import logging
import yaml
from datetime import datetime
from ask_llm import OpenCodeClient

# ============================================================
# 可配置常量
# ============================================================

LEADER_FILE    = "requirement_leader.md"
OVERALL_FILE   = "design_overall.md"

# 步骤1：生成候选总体设计文档的份数
GENERATE_CANDIDATES = 5

# 步骤7.1：接口对齐循环轮数
ALIGN_ROUNDS = 5

# 步骤6 / 7.2-7.4：审核-优化循环调用上限 = REVIEW_OPTIMIZE_FACTOR 块 reviewer 模型数
# 公式：(len(doc_reviewer_models) + 1) * REVIEW_OPTIMIZE_FACTOR
REVIEW_OPTIMIZE_FACTOR = 5

# ============================================================
# 日志配置
# ============================================================

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ============================================================
# 工具函数 - LLM 调用包装（带日志）
# ============================================================

_llm_call_index = 0  # 全局调用计数器

def llm_call(client, prompt: str, model=None, step_desc: str = "") -> str:
    """
    统一的 LLM 调用入口，自动记录：
    - 步骤描述、调用序号
    - 完整 prompt
    - 完整 response
    - 耗时
    """
    global _llm_call_index
    _llm_call_index += 1
    idx = _llm_call_index
    model_display = model or "默认模型"

    logger.info("=" * 60)
    logger.info(f"📤 第 {idx} 次 LLM 调用 [模型: {model_display}] — {step_desc}")
    logger.info(f"📝 Prompt:\n{prompt}")
    logger.info("-" * 60)

    t0 = time.time()
    response = client.chat(prompt, model=model)
    elapsed = time.time() - t0

    if response:
        logger.info(f"📥 Response（耗时 {elapsed:.1f}s）:\n{response}")
    else:
        logger.warning(f"⚠️ LLM 返回为空（耗时 {elapsed:.1f}s）")
    logger.info("=" * 60)
    return response or ""


# ============================================================
# 工具函数 - 配置加载
# ============================================================

def load_config(config_path='agents_config.yaml'):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: 配置文件 {config_path} 不存在")
        return {}


from doc_utils import read_file, read_requirement_docs, read_module_design_docs


# ============================================================
# 工具函数 - 结构化响应解析
# ============================================================

def parse_review_response(response: str) -> dict:
    """
    解析审核响应，期望 LLM 返回结构化 JSON：
    {
        "satisfied": true/false,
        "issues": ["问题1", "问题2", ...],
        "suggestions": "汇总建议文字",
        "score": 85
    }
    返回解析后的 dict，解析失败时给出默认值。
    """
    if not response:
        return {"satisfied": False, "issues": [], "suggestions": "", "score": 0}

    # 尝试提取 JSON 块
    try:
        # 匹配最外层 {...}
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            raw = match.group()
            raw = raw.replace("'", '"')  # 兼容单引号
            data = json.loads(raw)
            score_raw = data.get("score", 0)
            score = int(str(score_raw).strip()) if str(score_raw).strip().isdigit() else 0
            issues = data.get("issues", [])
            if isinstance(issues, str):
                issues = [issues] if issues.strip() else []
            return {
                "satisfied": bool(data.get("satisfied", False)),
                "issues": issues,
                "suggestions": data.get("suggestions", ""),
                "score": score,
            }
    except Exception:
        pass

    # 正则兜底：尝试提取 score 和 satisfied
    score_match = re.search(r'"score"\s*:\s*"?(\d+)"?', response)
    satisfied_match = re.search(r'"satisfied"\s*:\s*(true|false)', response, re.IGNORECASE)
    score = int(score_match.group(1)) if score_match else 0
    satisfied = satisfied_match and satisfied_match.group(1).lower() == "true"
    return {
        "satisfied": satisfied,
        "issues": [],
        "suggestions": response.strip()[:800],
        "score": score,
    }


def is_all_satisfied(review_parsed_list: list[dict]) -> bool:
    """判断所有审核结果是否全部满意（satisfied=true 且 issues 列表为空）。"""
    return all(
        r.get("satisfied", False) and len(r.get("issues", [])) == 0
        for r in review_parsed_list
    )


def merge_suggestions_from_parsed(parsed_list: list[dict], reviewer_labels: list[str]) -> str:
    """从结构化审核结果中合并 issues 和 suggestions，过滤掉满意的返回。"""
    parts = []
    for label, parsed in zip(reviewer_labels, parsed_list):
        if parsed.get("satisfied") and not parsed.get("issues"):
            continue
        issues = parsed.get("issues", [])
        suggestions = parsed.get("suggestions", "")
        if issues:
            issues_str = "\n".join(f"  - {iss}" for iss in issues)
            parts.append(f"[{label}] 具体问题：\n{issues_str}")
        if suggestions:
            parts.append(f"[{label}] 综合建议：{suggestions}")
    return "\n\n".join(parts)


REVIEW_OUTPUT_FORMAT = """\
请仅返回以下 JSON，不要有其他内容：
{
    "satisfied": true 或 false,
    "issues": ["具体问题1（指出位置+原因）", "具体问题2", ...],
    "suggestions": "综合改进建议（如无问题可为空字符串）",
    "score": 0到100的整数
}
说明：
- satisfied 为 true 当且仅当文档完全满足要求，issues 列表为空；
- 有任何问题时 satisfied 必须为 false，并在 issues 中逐条列出；
- score 反映整体质量，满分100。"""


SCORE_OUTPUT_FORMAT = """\
请仅返回以下 JSON，不要有其他内容：
{
    "issues": ["具体问题1（指出位置+原因）", "具体问题2", ...],
    "suggestions": "综合改进建议",
    "score": 0到100的整数
}
说明：
- issues 中逐条列出所有不合理或可优化的地方；如无问题 issues 为空列表；
- score 反映整体质量，满分100。"""


# ============================================================
# 步骤 1: 生成5份总体设计文档
# ============================================================

def step1_generate_overall(client, doc_writer_models):
    logger.info("\n" + "=" * 60)
    logger.info(f"【步骤1】 生成 {GENERATE_CANDIDATES} 份总体设计文档（轮换 doc_writer 模型）")
    logger.info("=" * 60)

    req_content = read_requirement_docs()
    leader_content = read_file(LEADER_FILE)

    for i in range(1, GENERATE_CANDIDATES + 1):
        model = doc_writer_models[(i - 1) % len(doc_writer_models)]
        model_display = model or "默认模型"
        doc_name = f"design_overall_{i}.md"
        logger.info(f"\n[{i}/{GENERATE_CANDIDATES}] 生成 {doc_name}（模型: {model_display}）...")

        prompt = f"""你是一位资深的软件架构师。
请仔细阅读以下需求文档和模块拆分设计文档，充分理解业务全貌后，生成总体设计文档 {doc_name}。

注意：你生成的文档名为 {doc_name}，请将内容保存到该文件。

【输入文档】
{req_content}

{leader_content}

【设计要求】文档必须包含以下章节，且每个章节需有具体内容而非模糊占位：

1. 总体业务架构
   - 列出所有模块及其职责层次
   - 用 ASCII 或文字描述模块间依赖关系图（明确标注依赖方向）

2. 业务驱动模式
   - 逐一列出每个模块的驱动方式：UI交互、流程驱动、定时任务或事件驱动
   - 说明选择该驱动方式的理由

3. 推荐设计模式
   - 分析整个业务架构是否适合某种设计模式（如 CQRS、Event Sourcing、Saga、Repository 等）
   - 如适用，说明应用方式和收益；如没有，说明原因

4. 业务流时序图
   - 针对最核心的 2-3 个主要业务场景分别画出多模块时序图（可用文字时序图）
   - 包含参与者、消息、返回值

5. 技术栈
   - 后端语言/框架选型及理由
   - 中间件选型（消息队列、缓存等）及理由

6. 总体技术架构
   - 分层架构图（如表现层、应用层、领域层、基础设施层）
   - 各层的职责和边界

7. 数据库选型
   - 选型结论及理由：结合业务特点分析为什么选这种数据库
   - 如需多种数据库方案，说明各自分工

注意：文档第一行注明你是哪个大模型。"""

        llm_call(client, prompt, model=model, step_desc=f"步骤1: 生成 {doc_name}")
        logger.info(f"[{i}/{GENERATE_CANDIDATES}] {doc_name} 生成完毕。")


# ============================================================
# 步骤 2: 多模型对5份总体设计文档打分
# ============================================================

def step2_review_overall(client, doc_reviewer_models) -> dict:
    logger.info("\n" + "=" * 60)
    logger.info(f"【步骤2】 多模型评审总体设计文档（{len(doc_reviewer_models)} 模型 × {GENERATE_CANDIDATES} = {len(doc_reviewer_models)*GENERATE_CANDIDATES} 次）")
    logger.info("=" * 60)

    req_content = read_requirement_docs()
    leader_content = read_file(LEADER_FILE)

    scores = {i: {"total_score": 0, "issue_count": 0, "suggestions_by_model": [], "labels": []} for i in range(1, GENERATE_CANDIDATES + 1)}
    total_calls = len(doc_reviewer_models) * GENERATE_CANDIDATES
    call_count = 0

    for reviewer_model in doc_reviewer_models:
        reviewer_display = reviewer_model or "默认模型"
        for i in range(1, GENERATE_CANDIDATES + 1):
            call_count += 1
            doc_name = f"design_overall_{i}.md"
            doc_content = read_file(doc_name)
            logger.info(f"\n[{call_count}/{total_calls}] 模型 {reviewer_display} 审核 {doc_name}...")

            prompt = f"""你是一位资深的软件架构师。请对以下总体设计文档进行严格审核。

【待审核文档】文件名: {doc_name}
该文档基于以下需求文档和模块拆分设计生成。

【需求文档】
{req_content}

【模块拆分设计文档】（文件名: {LEADER_FILE}）
{leader_content}

【待审核总体设计文档】（文件名: {doc_name}）
{doc_content}

【审核维度】
1. 架构合理性：总体业务架构是否能支撑需求中所有业务场景，模块划分是否清晰合理；
2. 业务驱动模式：每个模块的驱动方式是否合理，是否有更优的方式；
3. 设计模式：推荐的设计模式是否最优，是否有更适合的模式；
4. 时序图质量：时序图是否准确反映业务流，是否深入清晰；
5. 技术栈选型：技术栈是否最适合这个业务的规模和特点；
6. 架构分层：技术架构分层是否合理清晰，层与层之间职责边界是否明确；
7. 数据库选型：数据库选型是否匹配业务数据结构和查询模式。

{SCORE_OUTPUT_FORMAT}"""

            response = llm_call(client, prompt, model=reviewer_model, step_desc=f"步骤2: 审核 {doc_name}")
            parsed = parse_review_response(response)
            score = parsed["score"]
            issues = parsed["issues"]
            suggestions = parsed["suggestions"]

            scores[i]["total_score"] += score
            scores[i]["issue_count"] += len(issues)
            if issues or suggestions:
                scores[i]["suggestions_by_model"].append(parsed)
                scores[i]["labels"].append(reviewer_display)
            logger.info(f"  → 得分: {score}，问题数: {len(issues)}")

    return scores


# ============================================================
# 步骤 3: 择优并优化生成 design_overall.md
# ============================================================

def step3_optimize_overall(client, scores, doc_writer_models):
    logger.info("\n" + "=" * 60)
    logger.info("【步骤3】 统计总分，择优优化生成 design_overall.md")
    logger.info("=" * 60)

    logger.info("\n各文档得分汇总：")
    for i in range(1, GENERATE_CANDIDATES + 1):
        logger.info(f"  design_overall_{i}.md → 总分: {scores[i]['total_score']}，总问题数: {scores[i]['issue_count']}")

    best_index = max(range(1, GENERATE_CANDIDATES + 1), key=lambda i: (scores[i]["total_score"], -scores[i]["issue_count"]))
    best_score = scores[best_index]["total_score"]
    logger.info(f"\n🏆 最高分: design_overall_{best_index}.md（总分: {best_score}，问题数: {scores[best_index]['issue_count']}）")

    all_suggestions = merge_suggestions_from_parsed(
        scores[best_index]["suggestions_by_model"],
        scores[best_index]["labels"]
    )
    if not all_suggestions:
        all_suggestions = "（无具体建议，请根据需求文档进行通用优化）"

    optimize_model = doc_writer_models[0]
    doc_content = read_file(f"design_overall_{best_index}.md")
    req_content = read_requirement_docs()
    leader_content = read_file(LEADER_FILE)

    logger.info(f"\n正在用模型 {optimize_model or '默认模型'} 优化 design_overall_{best_index}.md ...")

    prompt = f"""你是一位资深的软件架构师。

注意：你最终需要将优化后的内容保存到文件 {OVERALL_FILE}。

以下是当前评分最高的总体设计文档（design_overall_{best_index}.md），以及它所基于的需求文档和模块拆分设计文档。

【需求文档】
{req_content}

【模块拆分设计文档】（文件名: {LEADER_FILE}）
{leader_content}

【待优化总体设计文档】（文件名: design_overall_{best_index}.md）
{doc_content}

【审核意见（来自多位审核者，请自行判断取舍）】
{all_suggestions}

【优化要求】
1. 在保留原文档优点的基础上，逐条评估审核意见是否合理，合理的采纳，不合理的忽略并说明理由；
2. 确保最终文档覆盖所有需求文档中的业务场景；
3. 保留原有的完整章节结构：总体业务架构、驱动模式、设计模式、时序图、技术栈、技术架构、数据库选型；
4. 最终文档必须完整、具体，能直接指导模块设计和开发实现；
5. 将优化后的文档内容保存为 {OVERALL_FILE}。"""

    llm_call(client, prompt, model=optimize_model, step_desc=f"步骤3: 优化生成 {OVERALL_FILE}")
    logger.info(f"✅ {OVERALL_FILE} 已生成！")


# ============================================================
# 步骤 4: 获取模块列表
# ============================================================

def step4_get_modules(client, doc_writer_models) -> list[str]:
    logger.info("\n" + "=" * 60)
    logger.info("【步骤4】 从 requirement_leader.md 获取模块列表")
    logger.info("=" * 60)

    model = doc_writer_models[0]
    leader_content = read_file(LEADER_FILE)

    prompt = f"""请阅读以下文档，提取其中的模块列表。

{leader_content}

请仅返回以下 JSON 格式的模块列表，不要有其他内容：
["模块1", "模块2", "模块3"]"""

    response = llm_call(client, prompt, model=model, step_desc="步骤4: 获取模块列表")
    logger.info(f"LLM 返回模块列表原文：{response}")

    modules = []
    if response:
        try:
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                raw = match.group().replace("'", '"')
                modules = json.loads(raw)
        except Exception:
            pass

        if not modules:
            modules = re.findall(r'["「『【]([^"「』】\n]+)["」』】]', response)

    if not modules:
        logger.warning("⚠️ 无法解析模块列表，请检查 requirement_leader.md 文件。")
        return []

    logger.info(f"✅ 解析到 {len(modules)} 个模块: {modules}")
    return modules


# ============================================================
# 步骤 5: 按模块生成详细设计文档
# ============================================================

def step5_generate_module_docs(client, modules, doc_writer_models):
    logger.info("\n" + "=" * 60)
    logger.info(f"【步骤5】 按模块生成详细设计文档（共 {len(modules)} 个模块）")
    logger.info("=" * 60)

    req_content = read_requirement_docs()
    leader_content = read_file(LEADER_FILE)
    overall_content = read_file(OVERALL_FILE)

    for idx, module in enumerate(modules, 1):
        model = doc_writer_models[(idx - 1) % len(doc_writer_models)]
        doc_name = f"design_module_{module}.md"
        logger.info(f"\n[{idx}/{len(modules)}] 生成 {doc_name}（模型: {model or '默认模型'}）...")

        # 注入已生成的其他模块接口摘要，帮助 LLM 了解上下文
        existing_modules_context = _build_existing_modules_summary(modules[:idx-1])

        prompt = f"""你是一位资深软件架构师。
请阅读以下文档，为模块「{module}」生成详细设计文档。

注意：你生成的文档名为 {doc_name}，请将内容保存到该文件。

【需求文档】
{req_content}

【模块拆分设计文档】（文件名: {LEADER_FILE}）
{leader_content}

【总体设计文档】（文件名: {OVERALL_FILE}）
{overall_content}

{existing_modules_context}

【设计要求】文档必须包含以下章节，内容必须详实到能直接写代码的程度：

1. 实体与实体关系
   - 定义该模块的所有实体和字段（包含字段类型、是否必填、说明）
   - 用文字描述实体关系图（ER图）
   - 完整的建表 DDL：包含字段定义、主键（业务主键或系统流水号）、外键、常用查询所需的索引

2. 业务逻辑设计
   - 逐一描述每个业务操作的具体流程和规则
   - 对于复杂流程，提供时序图或流程图
   - 包含异常处理逻辑和边界条件

3. 设计模式（如适用）
   - 分析该模块是否适用某种设计模式
   - 如适用，说明应用方式和收益；如不适用，说明原因

4. 对外提供的接口定义
   - 列出该模块对外提供的所有接口
   - 每个接口需包含：接口名称、参数列表（含类型）、返回值、业务语义
   - 用伪代码写出核心实现逻辑

5. 对其他模块的依赖
   - 列出该模块依赖其他哪些模块
   - 每项依赖需列出：依赖模块名称、需要的接口名称、参数和返回值

注意：文档第一行注明你是哪个大模型。"""

        llm_call(client, prompt, model=model, step_desc=f"步骤5: 生成 {doc_name}")
        logger.info(f"[{idx}/{len(modules)}] {doc_name} 生成完毕。")


def _extract_sections(content: str, keywords: list[str]) -> str:
    """
    从 Markdown 文本中提取匹配关键词的章节。
    识别同级别的标题（# / ## / ### 等），从匹配标题开始提取到下一个同级标题结束。
    """
    lines = content.splitlines()
    extracted = []
    inside = False
    current_level = 0

    for line in lines:
        heading_match = re.match(r'^(#{1,6})\s+(.*)', line)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2)
            if any(kw in title for kw in keywords):
                inside = True
                current_level = level
                extracted.append(line)
            elif inside:
                if level <= current_level:
                    inside = False
                else:
                    extracted.append(line)
        elif inside:
            extracted.append(line)

    return "\n".join(extracted).strip()


def _build_existing_modules_summary(done_modules: list[str]) -> str:
    """
    读取已生成的模块文档，仅提取“对外接口定义”和“实体定义”两个章节，
    压缩注入到后续 prompt 的上下文大小，同时保留文件名标头。
    """
    if not done_modules:
        return ""

    # 提取这两类章节，关键词覆盖文档模板里可能出现的标题变体
    INTERFACE_KEYWORDS = ["对外主动接口", "对外提供的接口", "对外接口", "接口定义", "API"]
    ENTITY_KEYWORDS = ["实体与实体关系", "实体关系", "实体定义", "数据模型"]

    parts = ["【已生成的其他模块接口与实体摘要（供参考，避免定义冲突）】"]

    for m in done_modules:
        fname = f"design_module_{m}.md"
        if not os.path.exists(fname):
            continue
        try:
            with open(fname, 'r', encoding='utf-8') as f:
                raw = f.read()
        except Exception:
            continue

        interface_section = _extract_sections(raw, INTERFACE_KEYWORDS)
        entity_section = _extract_sections(raw, ENTITY_KEYWORDS)

        if not interface_section and not entity_section:
            # 两个章节都没提取到，降级为截取前 500 字符作为兄底
            fallback = raw.strip()[:500]
            parts.append(
                f"--- 模块: {m}（文件: {fname}，未能精确提取章节，以下为文档摘要）---\n{fallback}"
            )
        else:
            section_text = ""
            if entity_section:
                section_text += f"### 实体定义（节选自 {fname}）\n{entity_section}\n\n"
            if interface_section:
                section_text += f"### 对外接口定义（节选自 {fname}）\n{interface_section}\n"
            parts.append(f"--- 模块: {m}（文件: {fname}）---\n{section_text}")

    return "\n\n".join(parts) if len(parts) > 1 else ""


# ============================================================
# 步骤 6: 对每个模块文档进行多模型审核→优化循环
# ============================================================

def step6_review_optimize_module(client, modules, doc_writer_models, doc_reviewer_models):
    logger.info("\n" + "=" * 60)
    logger.info("【步骤6】 对每个模块文档进行多模型审核→优化循环")
    logger.info("=" * 60)

    max_calls_per_module = (len(doc_reviewer_models) + 1) * REVIEW_OPTIMIZE_FACTOR
    req_content = read_requirement_docs()
    leader_content = read_file(LEADER_FILE)
    overall_content = read_file(OVERALL_FILE)

    for module in modules:
        doc_name = f"design_module_{module}.md"
        logger.info(f"\n--- 开始审核模块: {module} ({doc_name})，上限 {max_calls_per_module} 次调用 ---")

        call_count = 0

        while call_count < max_calls_per_module:
            doc_content = read_file(doc_name)

            # 6.1 多模型审核
            review_parsed_list = []
            reviewer_labels = []
            for reviewer_model in doc_reviewer_models:
                if call_count >= max_calls_per_module:
                    break
                call_count += 1
                reviewer_display = reviewer_model or "默认模型"
                logger.info(f"  [{call_count}/{max_calls_per_module}] 模型 {reviewer_display} 审核 {doc_name}...")

                prompt = f"""你是一位资深软件架构师。请严格审核模块「{module}」的详细设计文档。

【需求文档】
{req_content}

【模块拆分设计文档】（文件名: {LEADER_FILE}）
{leader_content}

【总体设计文档】（文件名: {OVERALL_FILE}）
{overall_content}

【待审核模块设计文档】（文件名: {doc_name}）
{doc_content}

【审核维度】
1. 需求覆盖度：文档是否覆盖了该模块的所有需求，是否与需求一致，前后是否有矛盾；
2. 可开发性：文档是否详实，开发者能否直接根据文档写出代码而无需追问；
3. 设计模式：如有提及设计模式，是否合理，是否有更优的选择；
4. 实体设计：实体和实体关系的定义是否合理、准确、完整。DDL 表结构是否完善（包含字段、主键、索引）；
5. 接口定义：对外接口和伪代码实现是否合理、准确、完整；
6. 模块依赖：对其他模块的依赖和需要的接口是否合理、准确、完整。

{REVIEW_OUTPUT_FORMAT}"""

                response = llm_call(client, prompt, model=reviewer_model, step_desc=f"步骤6: 审核 {doc_name}")
                parsed = parse_review_response(response)
                review_parsed_list.append(parsed)
                reviewer_labels.append(reviewer_display)
                logger.info(f"    → satisfied={parsed['satisfied']}，问题数: {len(parsed['issues'])}，得分: {parsed['score']}")

            # 6.2 判断是否全部满意
            if is_all_satisfied(review_parsed_list):
                logger.info(f"  ✅ 所有模型对 {doc_name} 满意，跳过优化。")
                break

            suggestions = merge_suggestions_from_parsed(review_parsed_list, reviewer_labels)
            if not suggestions:
                logger.info("  ✅ 无有效建议，结束优化循环。")
                break

            if call_count >= max_calls_per_module:
                logger.warning("  ⚠️ 已达调用上限，停止。")
                break

            # 优化
            call_count += 1
            writer_model = doc_writer_models[0]
            logger.info(f"  [{call_count}/{max_calls_per_module}] 优化 {doc_name}...")

            doc_content = read_file(doc_name)
            optimize_prompt = f"""你是一位资深软件架构师。模块「{module}」的详细设计文档需要根据审核意见进行优化。

注意：你需要将优化后的内容直接更新到文件 {doc_name}。

【需求文档】
{req_content}

【总体设计文档】（文件名: {OVERALL_FILE}）
{overall_content}

【当前模块设计文档】（文件名: {doc_name}）
{doc_content}

【审核意见】
{suggestions}

【优化要求】
1. 逐条评估意见是否合理，合理的采纳，不合理的忽略；
2. 确保文档覆盖该模块的所有需求；
3. 实体、接口、伪代码需保持完整和准确；
4. 确保其他模块的依赖接口描述具体；
5. 将优化后的内容保存到 {doc_name}。"""

            llm_call(client, optimize_prompt, model=writer_model, step_desc=f"步骤6: 优化 {doc_name}")

        logger.info(f"  → 模块 {module} 审核优化完成（共 {call_count} 次调用）。")


# ============================================================
# 步骤 7.1: 接口对齐（连续调用5次）
# ============================================================

def step71_align_interfaces(client, doc_writer_models):
    logger.info("\n" + "=" * 60)
    logger.info(f"【步骤7.1】 跨模块接口对齐（连续调用 {ALIGN_ROUNDS} 轮，累积优化）")
    logger.info("=" * 60)

    previous_round_summary = ""  # 上一轮 LLM 报告的改动摘要

    for i in range(1, ALIGN_ROUNDS + 1):
        model = doc_writer_models[(i - 1) % len(doc_writer_models)]
        model_display = model or "默认模型"
        logger.info(f"\n[{i}/{ALIGN_ROUNDS}] 接口对齐第 {i} 轮（模型: {model_display}）...")

        # 每轮重新读取文件，读到的是上一轮已改动后的最新内容
        overall_content = read_file(OVERALL_FILE)
        module_docs_content = read_module_design_docs()

        # 构建"上轮改动记录"上下文，让 LLM 知道哪些已经处理过
        prev_context = ""
        if previous_round_summary:
            prev_context = f"""
【上一轮（第 {i-1} 轮）已完成的对齐改动记录】
以下是上一轮对齐操作的摘要，本轮请在此基础上继续推进，不要重复处理已完成的项，
也不要撤销上一轮已经对齐的接口定义：
{previous_round_summary}

"""

        prompt = f"""你是一位资深软件架构师。请对所有模块设计文档进行跨模块接口对齐（第 {i} 轮，共 {ALIGN_ROUNDS} 轮）。
{prev_context}
【当前总体设计文档】（文件名: {OVERALL_FILE}）
{overall_content}

【当前所有模块设计文档（已包含上轮对齐结果）】
{module_docs_content}

【本轮对齐工作】
1. 逐模块扫描"对外依赖接口"的所有描述
   - 找到每个模块中"依赖其他模块的接口"的内容
   - 对应到被依赖模块的实际接口定义

2. 接口对齐规则
   - 如模块A依赖模块B的某个接口，且模块B已定义该接口：
     在模块A中直接引用模块B的接口定义，并对齐参数（字段名、类型等）
   - 如模块B尚未定义该接口：
     在模块B中新增该接口定义（含伪代码），再在模块A中引用

3. 接口复用原则
   - 尽量将功能相似的接口泛化为通用接口，不要为每种场景定义独立的特殊接口
   - 跨模块引用同一实体时确保实体名称和字段定义一致

4. 直接将对齐后的内容更新到对应的 design_module_*.md 文档（每次只更新有变化的文档）。

5. 本轮结束后，请在回复末尾用如下格式列出本轮所做的改动摘要（供下一轮参考）：
【本轮对齐摘要】
- 模块X.对外接口Y: 对齐了参数Z（原类型→新类型）
- 模块B: 新增接口defineXxx()，参数…，返回…
- （如无新改动请写"本轮无新改动"）"""

        response = llm_call(client, prompt, model=model, step_desc=f"步骤7.1: 接口对齐第 {i} 轮")

        # 从 response 中提取本轮改动摘要，传给下一轮
        summary_match = re.search(r'【本轮对齐摘要】([\s\S]*?)(?:$|【)', response)
        if summary_match:
            previous_round_summary = summary_match.group(1).strip()
        else:
            # 如果 LLM 没按格式输出，截取 response 末尾 500 字符作为摘要
            previous_round_summary = response.strip()[-500:] if response else ""

        logger.info(f"[{i}/{ALIGN_ROUNDS}] 完成，本轮摘要: {previous_round_summary[:100]}...")



# ============================================================
# 步骤 7.2-7.4: 全局审核→优化循环
# ============================================================

def step72_review_optimize_all(client, doc_writer_models, doc_reviewer_models):
    logger.info("\n" + "=" * 60)
    logger.info("【步骤7.2-7.4】 全局审核→优化循环")
    logger.info("=" * 60)

    max_calls = (len(doc_reviewer_models) + 1) * REVIEW_OPTIMIZE_FACTOR
    call_count = 0
    req_content = read_requirement_docs()
    leader_content = read_file(LEADER_FILE)

    while call_count < max_calls:
        overall_content = read_file(OVERALL_FILE)
        module_docs_content = read_module_design_docs()

        # 7.2 多模型全局审核
        review_parsed_list = []
        reviewer_labels = []
        for reviewer_model in doc_reviewer_models:
            if call_count >= max_calls:
                break
            call_count += 1
            reviewer_display = reviewer_model or "默认模型"
            logger.info(f"\n[{call_count}/{max_calls}] 模型 {reviewer_display} 全局审核所有设计文档...")

            prompt = f"""你是一位资深软件架构师。请对所有设计文档进行全局审核。

【需求文档】
{req_content}

【模块拆分设计文档】（文件名: {LEADER_FILE}）
{leader_content}

【总体设计文档】（文件名: {OVERALL_FILE}）
{overall_content}

【所有模块设计文档】
{module_docs_content}

【审核维度】
1. 需求全覆盖：所有需求中的业务场景是否在设计文档中有对应覆盖，是否与需求一致，前后是否有矛盾；
2. 可开发性：设计文档是否详实，开发者能否直接写出代码而无需追问；
3. 设计模式合理性：各模块提到的设计模式是否最适合；
4. 实体完备性：实体定义和表结构是否准确、完整，查询模式是否合理；
5. 接口完备性：对外接口和伪代码实现是否准确完整；
6. 模块依赖完备性：各模块间的依赖和接口依赖是否准确、完整；
7. 跨模块一致性：各模块间跨引用的同一实体和接口名称是否一致。

{REVIEW_OUTPUT_FORMAT}"""

            response = llm_call(client, prompt, model=reviewer_model, step_desc=f"步骤7.2: 全局审核")
            parsed = parse_review_response(response)
            review_parsed_list.append(parsed)
            reviewer_labels.append(reviewer_display)
            logger.info(f"  → satisfied={parsed['satisfied']}，问题数: {len(parsed['issues'])}，得分: {parsed['score']}")

        # 7.3 判断
        if is_all_satisfied(review_parsed_list):
            logger.info("\n✅ 所有审核模型对所有设计文档满意，全局流程完成！")
            break

        suggestions = merge_suggestions_from_parsed(review_parsed_list, reviewer_labels)
        if not suggestions:
            logger.info("\n✅ 无有效全局建议，结束优化循环。")
            break

        if call_count >= max_calls:
            logger.warning(f"\n⚠️ 已达全局调用上限 {max_calls}，停止。")
            break

        # 7.4 优化
        call_count += 1
        writer_model = doc_writer_models[0]
        logger.info(f"\n[{call_count}/{max_calls}] 根据全局审核建议优化所有设计文档...")

        overall_content = read_file(OVERALL_FILE)
        module_docs_content = read_module_design_docs()

        optimize_prompt = f"""你是一位资深软件架构师。请根据全局审核意见优化所有设计文档。

注意：你需要将优化后的内容直接更新到对应的文件（{OVERALL_FILE} 和 design_module_*.md）。

【需求文档】
{req_content}

【模块拆分设计文档】（文件名: {LEADER_FILE}）
{leader_content}

【当前总体设计文档】（文件名: {OVERALL_FILE}）
{overall_content}

【当前所有模块设计文档】
{module_docs_content}

【全局审核意见】
{suggestions}

【优化要求】
1. 逐条评估意见是否合理，合理的采纳，不合理的忽略；
2. 保留现有文档已合理的内容，不需推倒重来；
3. 确保更新后各文档之间的接口引用和实体定义保持一致；
4. 将更新后的内容分别保存到对应文件。"""

        llm_call(client, optimize_prompt, model=writer_model, step_desc="步骤7.4: 全局优化")
        logger.info(f"[{call_count}/{max_calls}] 优化完毕。")

    logger.info(f"\n全局审核优化共调用 {call_count} 次。")


# ============================================================
# 主流程
# ============================================================

def main():
    config = load_config()

    doc_writer_models = config.get('agents', {}).get('doc_writer', {}).get('models', [])
    if not doc_writer_models:
        logger.warning("配置文件中没有找到 doc_writer 的 models，使用默认模型")
        doc_writer_models = [None]

    doc_reviewer_models = config.get('agents', {}).get('doc_reviewer', {}).get('models', [])
    if not doc_reviewer_models:
        logger.warning("配置文件中没有找到 doc_reviewer 的 models，使用默认模型")
        doc_reviewer_models = [None]

    logger.info(f"📋 写文档模型: {doc_writer_models}")
    logger.info(f"📋 审核模型:   {doc_reviewer_models}")
    logger.info(f"📊 候选总体设计文档数: {GENERATE_CANDIDATES}")
    logger.info(f"📊 接口对齐轮数: {ALIGN_ROUNDS}")
    logger.info(f"📊 审核-优化循环因子: {REVIEW_OPTIMIZE_FACTOR}")

    # 添加带时间戳的文件日志 handler
    log_filename = f"generate_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)
    logger.info(f"📔 日志文件: {log_filename}")

    client = OpenCodeClient()

    # 步骤 1: 生成总体设计文档
    step1_generate_overall(client, doc_writer_models)

    # 步骤 2: 多模型评审
    scores = step2_review_overall(client, doc_reviewer_models)

    # 步骤 3: 择优优化
    step3_optimize_overall(client, scores, doc_writer_models)

    # 步骤 4: 获取模块列表
    modules = step4_get_modules(client, doc_writer_models)
    if not modules:
        logger.error("❌ 未能获取模块列表，流程终止。")
        return

    # 步骤 5: 生成模块详细设计文档
    step5_generate_module_docs(client, modules, doc_writer_models)

    # 步骤 6: 审核→优化每个模块
    step6_review_optimize_module(client, modules, doc_writer_models, doc_reviewer_models)

    # 步骤 7.1: 接口对齐
    step71_align_interfaces(client, doc_writer_models)

    # 步骤 7.2-7.4: 全局审核→优化
    step72_review_optimize_all(client, doc_writer_models, doc_reviewer_models)

    logger.info("\n🎉 全部设计文档生成流程完成！")
    logger.info(f"  · {OVERALL_FILE}   —— 总体设计文档")
    logger.info("  · design_module_*.md  —— 各模块详细设计文档")
    logger.info(f"  · {log_filename}  —— 完整调用日志")


if __name__ == "__main__":
    main()
