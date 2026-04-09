"""
optimize_skill.py — Skill 目录自动优化

每轮优化都保存完整的 skill 目录快照，便于回溯对比：
  results/{name}_{timestamp}/
    v0/           ← 原始版本
    v1_keep/      ← 第1轮，保留
    v2_discard/   ← 第2轮，丢弃
    ...
    run.log
    run.tsv

用法：
  python optimize_skill.py \\
      --skill skills/xiaohongshu-copywriter \\
      --test-cases test_cases/xiaohongshu.json \\
      --max-iterations 20
"""

import argparse
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from evaluate import evaluate_skill

load_dotenv(Path(__file__).parent / ".env")

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1"),
)
MODEL = os.environ.get("MODEL", "Pro/MiniMaxAI/MiniMax-M2.5")

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

log = logging.getLogger("optimize-skill")


# ────────────────────────────────────────────
# 0. 日志
# ────────────────────────────────────────────

def setup_logging(log_path: Path):
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    fmt_console = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fmt_file    = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_console)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_file)

    log.addHandler(ch)
    log.addHandler(fh)


# ────────────────────────────────────────────
# 1. Skill 目录读写 & 快照
# ────────────────────────────────────────────

def read_skill_md(skill_dir: Path) -> str:
    return (skill_dir / "SKILL.md").read_text(encoding="utf-8")


def write_skill_md(skill_dir: Path, content: str):
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")


def snapshot_skill(skill_dir: Path, snapshot_dir: Path):
    """将 skill 目录完整拷贝到 snapshot_dir。"""
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    shutil.copytree(skill_dir, snapshot_dir)


def restore_skill(skill_dir: Path, snapshot_dir: Path):
    """从 snapshot_dir 恢复 skill 目录。"""
    if skill_dir.exists():
        shutil.rmtree(skill_dir)
    shutil.copytree(snapshot_dir, skill_dir)


# ────────────────────────────────────────────
# 2. AI 提出改动
# ────────────────────────────────────────────

def propose_improvement(
    current_skill: str,
    eval_result: dict,
    iteration: int,
    previous_attempts: list[dict],
) -> tuple[str, str]:
    """
    分析评估结果，提出针对性的 SKILL.md 改动。
    返回 (改动后完整 SKILL.md, 改动描述)。
    """
    prev_text = ""
    if previous_attempts:
        prev_text = "\n\n## 之前已尝试过的改动（避免重复）\n"
        for p in previous_attempts[-5:]:
            status = "✅ 保留" if p["kept"] else "❌ 撤销"
            prev_text += f"- 第{p['iteration']}轮 {status}（{p['score']}分）：{p['description']}\n"

    weakest_text = ""
    if eval_result["weakest_items"]:
        weakest_text = "\n## 失分最多的方面\n"
        for item_id, fail_count in eval_result["weakest_items"]:
            weakest_text += f"- {item_id}：在 {fail_count} 个测试用例中未通过\n"

    cases_text = "\n## 各测试用例详情\n"
    for cs in eval_result["case_scores"]:
        failed = [k for k, v in cs["scores"].items() if not v]
        cases_text += f"- {cs['case_id']}（{cs['score']}分）失分项：{failed}  原因：{cs['reason']}\n"

    prompt = f"""你是一名专业的 AI Prompt 工程师，正在用 autoresearch 方法优化一个 skill 的提示词。

## 当前任务
这是第 {iteration} 轮优化。当前平均分：{eval_result['avg_score']}/100。
目标：提出一个针对性的小改动，让平均分提升。
{weakest_text}
{cases_text}
{prev_text}

## 优化原则（非常重要）
1. **每次只改一件事**：小而精准的改动比大改动更可靠
2. **改动要有针对性**：直接针对失分最多的方面
3. **简洁优于复杂**：删除无用内容也是好改动
4. **禁止重复**：不要尝试之前已经失败的改动方向

## 当前 SKILL.md 内容
```markdown
{current_skill}
```

## 你的任务
1. 分析失分原因，确定最值得改进的一个方向
2. 对 SKILL.md 做一个小而精准的改动
3. 返回**完整的改动后 SKILL.md**（不是 diff，是完整内容）

请在最后用如下格式说明你做了什么改动（用于记录）：
<!-- CHANGE_DESCRIPTION: 一句话描述改动 -->

直接输出完整的 SKILL.md 内容，以 `---` 开头。"""

    log.debug("=" * 60)
    log.debug("[第 %d 轮] 优化 prompt（共 %d 字符）：\n%s", iteration, len(prompt), prompt)

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"thinking": {"type": "disabled"}},
    )

    new_skill = response.choices[0].message.content

    if "<!-- CHANGE_DESCRIPTION:" in new_skill:
        desc_start = new_skill.find("<!-- CHANGE_DESCRIPTION:") + len("<!-- CHANGE_DESCRIPTION:")
        desc_end   = new_skill.find("-->", desc_start)
        description = new_skill[desc_start:desc_end].strip()
        new_skill   = new_skill[:new_skill.find("<!-- CHANGE_DESCRIPTION:")].strip()
    else:
        description = f"第{iteration}轮改动"

    log.debug("[第 %d 轮] 改动描述：%s", iteration, description)
    log.debug("[第 %d 轮] 新 SKILL.md（共 %d 字符）：\n%s", iteration, len(new_skill), new_skill)

    return new_skill, description


# ────────────────────────────────────────────
# 3. TSV 记录
# ────────────────────────────────────────────

def init_tsv(run_dir: Path) -> Path:
    path = run_dir / "run.tsv"
    path.write_text("iteration\tscore\tstatus\tdescription\tweakest_items\n", encoding="utf-8")
    return path


def append_tsv(tsv_path: Path, iteration: int, score: float,
               status: str, description: str, weakest_items: list):
    weakest_str = ",".join(f"{k}({v})" for k, v in weakest_items[:3])
    with open(tsv_path, "a", encoding="utf-8") as f:
        f.write(f"{iteration}\t{score}\t{status}\t{description}\t{weakest_str}\n")


# ────────────────────────────────────────────
# 4. 主循环
# ────────────────────────────────────────────

def run(skill_dir: Path, test_config: dict, max_iterations: int = 20):
    skill_name = test_config["skill_name"]
    checklist  = test_config["checklist"]
    test_cases = test_config["test_cases"]

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_dir   = RESULTS_DIR / f"{skill_name}_{timestamp}"
    run_dir.mkdir(parents=True)

    setup_logging(run_dir / "run.log")
    tsv_path = init_tsv(run_dir)

    log.info("=" * 60)
    log.info("  🔬 optimize_skill 启动")
    log.info("  Skill     : %s", skill_name)
    log.info("  Skill 目录 : %s", skill_dir)
    log.info("  结果目录   : %s", run_dir)
    log.info("  测试用例   : %d 个", len(test_cases))
    log.info("  最大迭代   : %d 轮", max_iterations)
    log.info("  模型       : %s", MODEL)
    log.info("=" * 60)

    # ── v0：保存原始版本快照 ──
    snapshot_skill(skill_dir, run_dir / "v0")
    log.info("  已保存原始版本 → v0/")

    initial_skill = read_skill_md(skill_dir)
    log.debug("初始 SKILL.md（共 %d 字符）：\n%s", len(initial_skill), initial_skill)

    # ── 第 0 轮：建立基准线 ──
    log.info("")
    log.info("📊 第 0 轮：建立基准线...")
    baseline   = evaluate_skill(initial_skill, test_cases, checklist)
    best_score = baseline["avg_score"]
    last_eval  = baseline

    log.info("  基准分  : %.1f / 100", best_score)
    log.info("  失分项  : %s", baseline["weakest_items"])
    log.debug("基准评估详情：\n%s", json.dumps(baseline, ensure_ascii=False, indent=2))
    append_tsv(tsv_path, 0, best_score, "baseline", "原始 SKILL.md", baseline["weakest_items"])

    previous_attempts: list[dict] = []

    # ── 迭代循环 ──
    for i in range(1, max_iterations + 1):
        log.info("")
        log.info("─" * 60)
        log.info("🧪 第 %d 轮 | 当前最佳: %.1f / 100", i, best_score)

        # 备份当前状态（用于回滚）
        backup_dir = run_dir / f"_backup_iter{i}"
        snapshot_skill(skill_dir, backup_dir)

        # AI 提出改动
        log.info("  💡 AI 分析中，提出改动...")
        try:
            new_skill_md, description = propose_improvement(
                current_skill=read_skill_md(skill_dir),
                eval_result=last_eval,
                iteration=i,
                previous_attempts=previous_attempts,
            )
        except Exception as e:
            log.error("  ❌ AI 改动失败：%s", e, exc_info=True)
            shutil.rmtree(backup_dir)
            continue

        log.info("  改动描述：%s", description)
        write_skill_md(skill_dir, new_skill_md)

        # 评估新版本
        log.info("  📊 评估新版本...")
        try:
            new_eval = evaluate_skill(new_skill_md, test_cases, checklist)
        except Exception as e:
            log.error("  ❌ 评估失败：%s，回滚", e, exc_info=True)
            restore_skill(skill_dir, backup_dir)
            shutil.rmtree(backup_dir)
            append_tsv(tsv_path, i, 0.0, "crash", description, [])
            continue

        new_score = new_eval["avg_score"]
        log.debug("[第 %d 轮] 评估详情：\n%s", i, json.dumps(new_eval, ensure_ascii=False, indent=2))
        log.info("  新分数: %.1f / 100  （当前最佳: %.1f）", new_score, best_score)

        # 决策：保留 or 回滚
        if new_score > best_score:
            best_score = new_score
            last_eval  = new_eval
            status = "keep"
            kept   = True
            snapshot_skill(skill_dir, run_dir / f"v{i}_keep")
            log.info("  ✅ 改进！保留 → v%d_keep/  新最佳: %.1f / 100", i, best_score)
        else:
            restore_skill(skill_dir, backup_dir)
            last_eval = new_eval
            status = "discard"
            kept   = False
            # 仍保存一份快照（方便分析为何没改进）
            snapshot_skill(skill_dir, run_dir / f"v{i}_discard")
            log.info("  ❌ 未改进（%.1f ≤ %.1f），回滚 → v%d_discard/", new_score, best_score, i)

        shutil.rmtree(backup_dir)
        append_tsv(tsv_path, i, new_score, status, description, new_eval["weakest_items"])
        previous_attempts.append({
            "iteration":   i,
            "score":       new_score,
            "kept":        kept,
            "description": description,
        })

    # ── 最终报告 ──
    improvement = best_score - baseline["avg_score"]
    kept_count  = sum(1 for a in previous_attempts if a["kept"])

    log.info("")
    log.info("=" * 60)
    log.info("  🏆 优化完成！")
    log.info("  基准分 : %.1f / 100", baseline["avg_score"])
    log.info("  最终分 : %.1f / 100", best_score)
    log.info("  提升   : +%.1f 分", improvement)
    log.info("  保留   : %d / %d 次改动", kept_count, max_iterations)
    log.info("  结果   : %s", run_dir)
    log.info("=" * 60)
    log.debug("全部实验记录：\n%s", json.dumps(previous_attempts, ensure_ascii=False, indent=2))


# ────────────────────────────────────────────
# 5. 入口
# ────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="optimize_skill — 自动优化 skill 目录")
    parser.add_argument("--skill",          required=True,        help="skill 目录路径（含 SKILL.md）")
    parser.add_argument("--test-cases",     required=True,        help="测试用例 JSON 文件路径")
    parser.add_argument("--max-iterations", type=int, default=20, help="最大迭代次数（默认 20）")
    args = parser.parse_args()

    skill_dir = Path(args.skill).resolve()
    if not (skill_dir / "SKILL.md").exists():
        print(f"❌ 找不到 SKILL.md：{skill_dir}")
        exit(1)

    run(
        skill_dir=skill_dir,
        test_config=json.load(open(args.test_cases, encoding="utf-8")),
        max_iterations=args.max_iterations,
    )
