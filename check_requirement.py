import os
import sys
import yaml
import argparse
from ask_llm import OpenCodeClient
from doc_utils import read_file, read_requirement_docs



def load_config(config_path='agents_config.yaml'):
    if not os.path.exists(config_path):
        print(f"Error: 配置文件 {config_path} 不存在")
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def process_single_file(client, doc_writer_models, doc_reviewer_models):

    # 提前读取需求文档，注入到 prompt（Python 侧读文件，不依赖 LLM 的文件读取能力）
    req_content = read_requirement_docs()

    # ==========================================
    # 任务 1: 连续调用5次，轮换不同模型，每个都用新会话
    # ==========================================
    for i in range(1, 6):
        # 轮流切换不同的 model
        current_model = doc_writer_models[(i - 1) % len(doc_writer_models)]
        model_display = current_model if current_model else "默认模型"
        print(f"\n[{i}/5] 开始检查需求文档 (使用模型: {model_display})...")

        output_file_name = f"requirement_insufficient_{i}.md"

        prompt = f"""你是一位经验丰富的业务分析师和需求工程师。
请阅读以下需求文档，该需求文档的最终目的是指导开发者直接写出业务代码，因此必须达到下列标准。

【需求文档】
{req_content}

【审核标准】
1. 需求详实、语义清晰：每个需求点的语义非常清楚、不存在多种解读。证验方式：假设自己是开发者，每个需求点是否能直接动手写代码而无需追问；
2. 前后一致性：文档内各需求之间没有矛盾、有歧义或语义模糊的描述；
3. 数据来源明确：所有涉及的数据字段和实体都有明确的来源说明——是数据库表中的某字段、还是通过某接口从外部系统获取，需讲清具体怎么获取；
4. 计算逻辑具体化：所有需要计算的逻辑都必须给出明确的公式或计算步骤，不能只写"按照一定规则计算"、"合理分配"这类模糊表述；
5. 公式元素可源性：文档中所有公式的每个元素都必须能从前面已定义的公式结果或明确指定的数据源中获得。

【审核方式】
- 逐条检查每个需求模块和需求点，不要泛泛地给出整体评价；
- 对于每个不满足标准的点，必须：
  (a) 指出具体在哪个需求模块/需求点上不满足；
  (b) 说明不满足的原因；
  (c) 给出具体的补充建议或建议开发者向业务方求确认的问题。

【输出要求】
- 如果需求文档完全满足以上所有标准，则返回：需求文档清晰明了，可用于开发
- 如果存在不足，请按以下格式生成文档 {output_file_name}：

  # 需求文档审核报告（第 {i} 次，{model_display}）

  ## 一、审核总结
  （简要说明整体质量）

  ## 二、具体问题清单
  （按标准分类，每个问题说明位置 + 原因 + 建议）

  ## 三、需要进一步确认的问题（可选）
  （列出需要业务方确认的模糊点）

注意：文档第一行说明你是哪个大模型。"""

        # 不传 session_id，使得每次都创建新会话
        client.chat(prompt, model=current_model)

    # ==========================================
    # 任务 2: 审核前5次生成的检查报告，合并总结出最终意见
    # ==========================================
    reviewer_model = doc_reviewer_models[0]
    reviewer_model_display = reviewer_model if reviewer_model else "默认模型"
    print(f"\n[汇总阶段] 开始总结所有的审查结果 (使用模型: {reviewer_model_display})...")

    # 读取5份审核报告内容，注入到 prompt
    report_content = "\n".join(
        read_file(f"requirement_insufficient_{j}.md") for j in range(1, 6)
    )

    summary_prompt = f"""你是一位资深的需求质量专家。

背景：以下是五位来自不同大模型的审核者对需求文档所做的全面审查报告。

【需求文档】
{req_content}

【五份审核报告】
{report_content}

请完成以下整合工作：

【整合规则】
1. 与需求文档对照验证：每个意见必须确实指向需求文档中存在的问题，不准添加需求中未提及的内容；
2. 去重合并：将各审核者提到的相同或相似问题合并为一条，避免重复；
3. 过滤不合理点：如果某条意见超出需求文档的范围、或者是主观偏见而非客观问题，删除并说明删除理由；
4. 按重要性排序：影响开发正确性的问题排在前面，相对次要的优化建议排在后面。

【输出格式】
请生成文档 requirement_insufficient.md，结构如下：

# 需求文档审核总结报告

## 一、总体评价
（简要说明需求文档的整体质量和主要瓶颈）

## 二、待改进问题清单
按问题类型分组列出，每个问题包含：
  - 问题位置：具体指出在哪个文档/模块/需求点
  - 问题描述：具体是什么问题
  - 改进建议：建议如何补充或修改

## 三、需业务方进一步确认的问题（可选）
（列出需求文档中存在描述模糊、需要业务方确认才能继续的问题）"""

    # 使用新会话调用
    client.chat(summary_prompt, model=reviewer_model)

def main():
   
    # 加载配置
    config = load_config()
    
    # 获取 doc_writer 配置的模型
    doc_writer_models = config.get('agents', {}).get('doc_writer', {}).get('models', [])
    if not doc_writer_models:
        print("警告: 配置文件中没有找到 doc_writer 的 models，使用默认模型")
        doc_writer_models = [None]
        
    # 获取 doc_reviewer 配置的模型
    doc_reviewer_models = config.get('agents', {}).get('doc_reviewer', {}).get('models', [])
    if not doc_reviewer_models:
        print("警告: 配置文件中没有找到 doc_reviewer 的 models，使用默认模型")
        doc_reviewer_models = [None]
        
    client = OpenCodeClient()
    
    process_single_file(client, doc_writer_models, doc_reviewer_models)

if __name__ == "__main__":
    main()
