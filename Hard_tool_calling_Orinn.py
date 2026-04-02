
"""
╔═══════════════════════════════════════════════════════════════════╗
║   ORINN — HARD MODE TOOL-CALLING BENCHMARK                        ║
║   7 Suites: Implicit • Narrative • Safety • Dependencies          ║
║             Distractors • Repetition • Precision                  ║
║                                                                   ║
║   pip install rich plotext openai                                 ║
║   python Hard_tool_calling_Orinn.py                               ║
╚═══════════════════════════════════════════════════════════════════╝
"""

import threading
import time
import datetime
import statistics

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from openai import OpenAI

import hard_benchmark_engine as engine

# ═══════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════
BASE_URL = "https://api-call.orinn.ai/v1"
API_KEY  = "Orinn_API_KEY"
MODEL    = "Orinn-1.6"

engine.model_name  = MODEL
engine.model_color = "cyan"


# ═══════════════════════════════════════════════════════════
#  API CALLER — OpenAI-compatible
# ═══════════════════════════════════════════════════════════
def call_model(test_case: engine.HardTestCase) -> tuple:
    """Call Orinn and return (actual_calls: list[{name,args}], time, error)."""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    tools_to_send = engine.MEDICAL_TOOLS[:test_case.distractor_tools]

    messages = [
        {"role": "system", "content": engine.SYSTEM_PROMPT},
        {"role": "user", "content": test_case.prompt},
    ]

    try:
        t_start = time.perf_counter()
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools_to_send,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=8000,
        )
        t_end = time.perf_counter()

        calls = []
        if response.choices and response.choices[0].message.tool_calls:
            for tc in response.choices[0].message.tool_calls:
                args = {}
                if tc.function.arguments:
                    try:
                        import json
                        args = json.loads(tc.function.arguments)
                    except Exception:
                        args = {}
                calls.append({"name": tc.function.name, "args": args})

        return calls, t_end - t_start, None

    except Exception as e:
        return [], 0, str(e)[:400]


def run_warmup():
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    for _ in range(2):
        try:
            client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "Hello"}],
                tools=[engine.MEDICAL_TOOLS[0]],
                tool_choice="auto",
                max_tokens=100,
            )
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════
#  TEST EXECUTOR
# ═══════════════════════════════════════════════════════════
def execute_test(test_case: engine.HardTestCase, score: engine.TestScore):
    with engine.scores_lock:
        score.test_id = test_case.id
        score.test_name = test_case.name
        score.suite = test_case.suite
        score.phase = "calling"

    calls, elapsed, error = call_model(test_case)

    with engine.scores_lock:
        if error:
            score.error = error
            score.phase = "error"
            score.done = True
            return

        score.phase = "scoring"

    # Score
    result = engine.score_test_result(test_case, calls)

    with engine.scores_lock:
        score.call_scores = result.call_scores
        score.tool_selection_score = result.tool_selection_score
        score.arg_precision_score = result.arg_precision_score
        score.over_calling_penalty = result.over_calling_penalty
        score.safety_order_score = result.safety_order_score
        score.overall_score = result.overall_score
        score.actual_calls = calls
        score.total_time = elapsed
        score.done = True
        score.phase = "done"


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════
def main():
    console = Console()
    console.clear()

    with Live(engine.build_dashboard(), console=console, refresh_per_second=4, screen=False) as live:

        # Warm-up
        engine.current_phase = "warmup"
        live.update(engine.build_dashboard())
        wt = threading.Thread(target=run_warmup)
        wt.start()
        while wt.is_alive():
            live.update(engine.build_dashboard())
            time.sleep(0.25)
        wt.join()

        # Run all test cases
        for idx, tc in enumerate(engine.ALL_TEST_CASES):
            engine.current_phase = "running"
            engine.current_test_name = f"[{tc.id}] {tc.name}"
            engine.progress_text = f"{idx+1}/{len(engine.ALL_TEST_CASES)}"

            score = engine.TestScore()
            engine.current_score = score
            engine.all_scores.append(score)

            t = threading.Thread(target=execute_test, args=(tc, score))
            t.start()
            while t.is_alive():
                live.update(engine.build_dashboard())
                time.sleep(0.25)
            t.join()
            live.update(engine.build_dashboard())

            if idx < len(engine.ALL_TEST_CASES) - 1:
                engine.current_phase = "cooldown"
                live.update(engine.build_dashboard())
                time.sleep(engine.COOLDOWN_SEC)

        engine.current_phase = "done"
        live.update(engine.build_dashboard())

    # Export
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"hard_orinn_{ts}.json"
    engine.export_json(json_path)
    console.print(f"\n  📊 JSON: [cyan]{json_path}[/]")

    html_path = engine.generate_html_report(json_path)
    console.print(f"  📄 HTML: [cyan]{html_path}[/]")

    # Final
    good = [s for s in engine.all_scores if s.done and not s.error]
    if good:
        avg = statistics.mean([s.overall_score for s in good])
        pass_ct = sum(1 for s in good if s.overall_score >= 60)

        if avg >= 90: grade = "A+"
        elif avg >= 80: grade = "A"
        elif avg >= 70: grade = "B"
        elif avg >= 60: grade = "C"
        elif avg >= 50: grade = "D"
        else: grade = "F"

        color = "green" if avg >= 70 else ("yellow" if avg >= 50 else "red")
        console.print()
        console.print(Panel(
            f"[bold bright_{color}]Grade: {grade} — {avg:.0f}% Overall[/]\n\n"
            f"  Model:       {MODEL}\n"
            f"  Pass Rate:   {pass_ct}/{len(good)} tests\n"
            f"  Suites:      7 (Implicit, Narrative, Safety, Dependencies, Distractors, Repetition, Precision)\n\n"
            f"  [dim]Share the HTML report on LinkedIn[/]",
            border_style=color, title=f"[bold {color}] Hard Mode Results "
        ))

    console.print(f"\n  💡 Open [cyan]{html_path}[/] in browser\n")


if __name__ == "__main__":
    main()