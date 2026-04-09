"""
evaluate.py — 评估函数

用大模型给 skill 输出打分（OpenAI 接口规范）。
相当于 autoresearch 里固定不动的 evaluate_bpb()，
评分标准永远不变，只有 SKILL.md 在变。
"""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1"),
)
MODEL = os.environ.get("MODEL", "Pro/MiniMaxAI/MiniMax-M2.5")

log = logging.getLogger("skill-autoresearch")


def evaluate_single(output: str, test_case: dict, checklist: list[dict]) -> dict:
    """
    用模型对单个输出按 checklist 打分。
    返回 {scores: {...}, reason: str, score: float}。
    """
    checklist_text = "\n".join(
        f"{i+1}. [{item['id']}] {item['description']}"
        for i, item in enumerate(checklist)
    )

    score_keys = "\n    ".join(
        f'"{item["id"]}": true/false'
        for item in checklist
    )

    prompt = f"""你是一个严格的内容质量评估员。请根据以下 checklist 评估这段小红书文案。

## 原始输入（用户需求）
{test_case['input']}

## 必须出现的关键信息
{json.dumps(test_case['key_info'], ensure_ascii=False)}

## 待评估的文案输出
{output}

## 评估 Checklist
{checklist_text}

## 评估规则
- 对每一条 checklist 项，判断输出是否满足：true（满足）或 false（不满足）
- 判断要严格：按照每条 checklist 的描述逐一核对，有疑问时从严判断
- 只返回 JSON，格式如下，不要任何解释：
{{
  "scores": {{
    {score_keys}
  }},
  "reason": "一句话说明最主要的问题（如果全部通过则说明亮点）"
}}"""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"thinking": {"type": "disabled"}},
    )

    raw = response.choices[0].message.content.strip()
    log.debug("  [%s] 评估原始返回：%s", test_case["id"], raw)

    # 剥离可能的 markdown 代码块 ```json ... ```
    cleaned = raw
    if "```" in cleaned:
        # 取第一个 ``` 之后、最后一个 ``` 之前的内容
        fence_start = cleaned.find("```")
        # 跳过 ```json 或 ``` 这一行
        newline_after_fence = cleaned.find("\n", fence_start)
        if newline_after_fence != -1:
            fence_end = cleaned.rfind("```")
            if fence_end > newline_after_fence:
                cleaned = cleaned[newline_after_fence + 1:fence_end].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    try:
        result = json.loads(cleaned[start:end])
    except json.JSONDecodeError as e:
        print(f"\n  ⚠️  JSON 解析失败：{e}")
        print(f"  原始返回内容：\n{raw}\n")
        raise

    # 加权算分
    total_weight = sum(item["weight"] for item in checklist)
    earned = sum(
        item["weight"]
        for item in checklist
        if result["scores"].get(item["id"], False)
    )
    result["score"] = round(earned / total_weight * 100, 1)
    return result


def evaluate_skill(skill_prompt: str, test_cases: list[dict], checklist: list[dict]) -> dict:
    """
    用所有测试用例评估一个 skill 版本，返回平均分和详情。

    返回：
        {
            "avg_score": float,       # 0-100，越高越好（对应 val_bpb 越低越好）
            "case_scores": [...],     # 每个用例的得分和原因
            "weakest_items": [...]    # 失分最多的 checklist 项
        }
    """
    case_scores = []
    item_fail_counts = {item["id"]: 0 for item in checklist}

    print(f"  📋 开始评估（共 {len(test_cases)} 个用例）…")

    for i, tc in enumerate(test_cases):
        print(f"    [{i+1}/{len(test_cases)}] {tc['id']}  生成中…", end="", flush=True)
        log.debug("  ── 用例 [%d/%d] %s", i + 1, len(test_cases), tc["id"])
        log.debug("  输入：%s", tc["input"])

        # 1. 让 skill 生成输出
        output_response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": skill_prompt},
                {"role": "user",   "content": tc["input"]}
            ],
            extra_body={"thinking": {"type": "disabled"}},
        )
        output = output_response.choices[0].message.content
        log.debug("  [%s] skill 输出（共 %d 字符）：\n%s", tc["id"], len(output), output)

        print("  打分中…", end="", flush=True)

        # 2. 评估输出
        eval_result = evaluate_single(output, tc, checklist)
        eval_result["case_id"] = tc["id"]
        eval_result["output_preview"] = output[:120] + "..." if len(output) > 120 else output
        case_scores.append(eval_result)

        # 统计失分项
        for item_id, passed in eval_result["scores"].items():
            if not passed:
                item_fail_counts[item_id] = item_fail_counts.get(item_id, 0) + 1

        # 打分详情
        passed_items  = [k for k, v in eval_result["scores"].items() if v]
        failed_items  = [k for k, v in eval_result["scores"].items() if not v]
        score = eval_result["score"]
        mark = "✅" if score >= 80 else ("⚠️ " if score >= 50 else "❌")
        print(f"  {mark} {score:.0f}分  ✅{passed_items}  ❌{failed_items}")
        log.info(
            "    [%d/%d] %s: %.0f分  ✅%s  ❌%s  — %s",
            i + 1, len(test_cases), tc["id"],
            eval_result["score"],
            passed_items, failed_items,
            eval_result["reason"],
        )

    avg_score = sum(r["score"] for r in case_scores) / len(case_scores)

    weakest_items = sorted(item_fail_counts.items(), key=lambda x: x[1], reverse=True)
    weakest_items = [(k, v) for k, v in weakest_items if v > 0]

    log.info("  平均分：%.1f / 100  |  失分项：%s", avg_score, weakest_items)

    return {
        "avg_score":    round(avg_score, 2),
        "case_scores":  case_scores,
        "weakest_items": weakest_items,
    }
