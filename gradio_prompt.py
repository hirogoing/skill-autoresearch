"""
gradio_prompt.py — Prompt 自动优化监控界面

职责：
  1. 右侧配置面板：填写参数，点击"开始"
  2. 启动 optimize_prompt.py 子进程（实际优化在子进程里跑）
  3. gr.Timer 每 3 秒轮询 results/ 下的输出文件，刷新界面

不再内嵌任何优化逻辑，命令行方式 `python optimize_prompt.py ...` 也照常工作。

启动：
  python gradio_prompt.py   →  http://localhost:7863
"""

import os
import sys
import time
from pathlib import Path

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
load_dotenv(_HERE / ".env")

RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
DEFAULT_MODEL   = os.getenv("MODEL", "doubao-seed-2-0-pro-260215")


# ──────────────────────────────────────────────
# 文件读取辅助
# ──────────────────────────────────────────────

def _status_html(text: str, color: str = "#555") -> str:
    return (
        f'<div style="padding:8px 12px;border-radius:6px;'
        f'background:#f5f5f5;color:{color};font-size:0.9em">{text}</div>'
    )


def _find_new_run_dir(dirs_before: set) -> str:
    """找 results/ 下新出现的目录；找不到就用最新的。"""
    if not RESULTS_DIR.exists():
        return ""
    all_dirs = {p for p in RESULTS_DIR.iterdir() if p.is_dir()}
    new_dirs  = all_dirs - dirs_before
    candidates = new_dirs if new_dirs else all_dirs
    if not candidates:
        return ""
    return str(sorted(candidates, key=lambda p: p.stat().st_mtime)[-1])


def _read_log(run_dir: Path) -> str:
    p = run_dir / "run.log"
    if not p.exists():
        return "等待日志输出…"
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        return f"读取日志失败：{e}"


def _read_plot_df(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "run.tsv"
    empty = pd.DataFrame({"轮次": pd.Series(dtype="int"), "分数": pd.Series(dtype="float")})
    if not p.exists():
        return empty
    try:
        df = pd.read_csv(p, sep="\t")
        out = df[["iteration", "score"]].rename(
            columns={"iteration": "轮次", "score": "分数"}
        )
        out["轮次"] = out["轮次"].astype(int)
        return out
    except Exception:
        return empty


def _read_history(run_dir: Path) -> list[list]:
    p = run_dir / "run.tsv"
    if not p.exists():
        return []
    try:
        df = pd.read_csv(p, sep="\t")
        rows = []
        for _, row in df.iterrows():
            rows.append([
                int(row["iteration"]),
                f"{float(row['score']):.1f}",
                str(row["status"]),
                str(row["description"])[:50],
            ])
        return rows
    except Exception:
        return []


def _read_prompts(run_dir: Path) -> tuple[str, str]:
    """返回 (原始 v0.md, 当前最优 prompt)。"""
    v0 = run_dir / "v0.md"
    orig = v0.read_text(encoding="utf-8") if v0.exists() else ""

    keep_files = sorted(run_dir.glob("v*_keep.md"), key=lambda p: p.stat().st_mtime)
    best = keep_files[-1].read_text(encoding="utf-8") if keep_files else orig
    return orig, best


# ──────────────────────────────────────────────
# 事件处理
# ──────────────────────────────────────────────

def on_start(prompt_path, test_cases_path, max_iter, model, api_key, base_url):
    """点击"开始"：校验参数 → 启动子进程 → 等待 run_dir 出现。"""
    import subprocess

    prompt_path    = prompt_path.strip()
    test_cases_path = test_cases_path.strip()

    if not Path(prompt_path).exists():
        return None, "", _status_html(f"❌ 找不到 Prompt 文件：{prompt_path}", "#dc2626")
    if not Path(test_cases_path).exists():
        return None, "", _status_html(f"❌ 找不到测试用例：{test_cases_path}", "#dc2626")

    dirs_before = {p for p in RESULTS_DIR.iterdir() if p.is_dir()} if RESULTS_DIR.exists() else set()

    env = os.environ.copy()
    if api_key.strip():
        env["OPENAI_API_KEY"] = api_key.strip()
    if base_url.strip():
        env["OPENAI_BASE_URL"] = base_url.strip()
    if model.strip():
        env["MODEL"] = model.strip()

    cmd = [
        sys.executable, str(_HERE / "optimize_prompt.py"),
        "--prompt",         prompt_path,
        "--test-cases",     test_cases_path,
        "--max-iterations", str(int(max_iter)),
    ]
    proc = subprocess.Popen(cmd, cwd=str(_HERE), env=env)

    # 等 run_dir 被子进程创建出来
    for _ in range(10):
        time.sleep(0.5)
        run_dir_str = _find_new_run_dir(dirs_before)
        if run_dir_str:
            break
    else:
        run_dir_str = ""

    status = _status_html(
        f"🔄 子进程已启动（PID {proc.pid}）| 结果目录：{Path(run_dir_str).relative_to(_HERE) if run_dir_str else '创建中…'}",
        "#2563eb",
    )
    return proc, run_dir_str, status


def on_stop(proc_state):
    """立即终止子进程。"""
    if proc_state is not None:
        try:
            proc_state.terminate()
        except Exception:
            pass
    return _status_html("🛑 已发送停止信号", "#dc2626")


def poll_files(run_dir_str: str, proc_state):
    """
    gr.Timer 每 3 秒触发：读取本地文件，刷新所有展示组件。
    返回顺序：log, plot_df, history, orig_prompt, best_prompt, status
    """
    empty_df = pd.DataFrame({"轮次": pd.Series(dtype="int"), "分数": pd.Series(dtype="float")})

    if not run_dir_str:
        return "", empty_df, [], "", "", _status_html("等待开始…", "#888")

    run_dir  = Path(run_dir_str)
    rel_dir  = Path(run_dir_str).relative_to(_HERE) if run_dir_str.startswith(str(_HERE)) else Path(run_dir_str)
    log_text = _read_log(run_dir)
    plot_df  = _read_plot_df(run_dir)
    history  = [["轮次", "分数", "状态", "改动说明"]] + _read_history(run_dir)
    orig, best = _read_prompts(run_dir)

    # 判断进程是否仍在运行
    alive = proc_state is not None and proc_state.poll() is None
    rows  = _read_history(run_dir)

    if alive:
        if rows:
            last = rows[-1]
            msg = f"🔄 进行中 — 第 {last[0]} 轮 | 最新分数：{last[1]}（{last[2]}）"
        else:
            msg = "🔄 优化进行中…"
        status = _status_html(msg, "#2563eb")
    else:
        if rows:
            last = rows[-1]
            msg = f"✅ 完成 — 共 {last[0]} 轮 | 最终分数：{last[1]} | 目录：{rel_dir}"
        else:
            msg = f"📂 监控中：{rel_dir}"
        status = _status_html(msg, "#16a34a")

    return log_text, plot_df, history, orig, best, status


# ──────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────

CSS = """
#log-box textarea {
    font-family: monospace;
    font-size: 0.78em;
    height: 520px !important;
    overflow-y: scroll !important;
}
footer { display: none !important; }
"""

with gr.Blocks(title="Prompt 自动优化", css=CSS, theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        "# 🔬 Prompt 自动优化\n"
        "<small style='color:#888'>配置参数 → 点击开始 → 监控 results/ 目录实时展示</small>"
    )

    proc_state    = gr.State(None)
    run_dir_state = gr.State("")

    with gr.Row(equal_height=False):

        # ── 左列：参数配置 ──
        with gr.Column(scale=1, min_width=240):
            gr.Markdown("### ⚙️ 参数配置")

            prompt_path_input = gr.Textbox(
                label="Prompt 文件路径（.md）",
                value="prompts/xiaohongshu-copywriter/prompt.md",
            )
            test_cases_input = gr.Textbox(
                label="测试用例 JSON 路径",
                value="test_cases/xiaohongshu.json",
            )
            max_iter_slider = gr.Slider(
                minimum=1, maximum=50, value=10, step=1,
                label="最大迭代轮数",
            )

            with gr.Accordion("🔑 模型配置", open=False):
                model_input    = gr.Textbox(label="模型", value=DEFAULT_MODEL)
                api_key_input  = gr.Textbox(label="API Key（留空用 .env）", type="password", value="")
                base_url_input = gr.Textbox(label="Base URL", value=OPENAI_BASE_URL)

            with gr.Row():
                start_btn = gr.Button("▶ 开始优化", variant="primary")
                stop_btn  = gr.Button("⏹ 停止",    variant="stop")

            gr.Markdown("---")
            gr.Markdown("#### 📊 迭代历史")
            history_df = gr.DataFrame(
                headers=["轮次", "分数", "状态", "改动说明"],
                datatype=["number", "str", "str", "str"],
                col_count=(4, "fixed"),
                interactive=False,
            )

        # ── 中列：趋势图 ──
        with gr.Column(scale=3):
            status_html = gr.HTML(value=_status_html("等待开始…", "#888"))

            score_plot = gr.LinePlot(
                value=pd.DataFrame({"轮次": pd.Series(dtype="int"), "分数": pd.Series(dtype="float")}),
                x="轮次", y="分数",
                title="评分趋势",
                y_lim=[0, 100],
                height=300,
                tooltip=["轮次", "分数"],
                point_size=80,
                x_bin=1,
            )

        # ── 右列：日志 ──
        with gr.Column(scale=1, min_width=240):
            log_box = gr.Textbox(
                label="运行日志（run.log）",
                lines=32, max_lines=500,
                elem_id="log-box",
                interactive=False,
                autoscroll=True,
            )

    # ── 底部：Prompt 对比（全宽）──
    gr.Markdown("#### 🔍 Prompt 对比（v0 原始 vs 当前最优）")
    with gr.Row():
        orig_prompt_box = gr.Code(
            label="原始 Prompt (v0.md)",
            language="markdown", lines=28, interactive=False,
        )
        best_prompt_box = gr.Code(
            label="当前最优 Prompt",
            language="markdown", lines=28, interactive=False,
        )

    # ── 定时轮询（每 3 秒）──
    timer = gr.Timer(value=3.0)
    timer.tick(
        fn=poll_files,
        inputs=[run_dir_state, proc_state],
        outputs=[log_box, score_plot, history_df, orig_prompt_box, best_prompt_box, status_html],
    )

    # ── 事件绑定 ──
    start_btn.click(
        fn=on_start,
        inputs=[
            prompt_path_input, test_cases_input, max_iter_slider,
            model_input, api_key_input, base_url_input,
        ],
        outputs=[proc_state, run_dir_state, status_html],
    )

    stop_btn.click(
        fn=on_stop,
        inputs=[proc_state],
        outputs=[status_html],
        queue=False,
    )

demo.queue()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=False,
        inbrowser=False,
    )
