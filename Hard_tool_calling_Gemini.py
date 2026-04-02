#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║   GEMINI — HARD MODE TOOL-CALLING BENCHMARK                      ║
║   7 Suites: Implicit • Narrative • Safety • Dependencies         ║
║             Distractors • Repetition • Precision                 ║
║                                                                  ║
║   pip install google-genai rich plotext                          ║
║   python Hard_tool_calling_Gemini.py                             ║
╚══════════════════════════════════════════════════════════════════╝
"""

import threading
import time
import datetime
import statistics

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from google import genai
from google.genai import types

import hard_benchmark_engine as engine

# ═══════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════
GEMINI_API_KEY = "Gemini_API_KEY"
MODEL          = "gemini-3.1-pro-preview"

engine.model_name  = MODEL
engine.model_color = "bright_blue"


# ═══════════════════════════════════════════════════════════
#  CONVERT OpenAI tools → Gemini FunctionDeclarations
# ═══════════════════════════════════════════════════════════
def _to_gemini_decl(openai_tool: dict) -> types.FunctionDeclaration:
    """Convert one OpenAI-format tool to Gemini FunctionDeclaration."""
    fn = openai_tool["function"]
    params = fn.get("parameters", {})

    # Convert properties
    gemini_props = {}
    for pname, pdef in params.get("properties", {}).items():
        sp = {"type": pdef["type"].upper()}
        if "description" in pdef:
            sp["description"] = pdef["description"]
        if "enum" in pdef:
            sp["enum"] = pdef["enum"]
        if pdef["type"] == "array" and "items" in pdef:
            sp["items"] = {"type": pdef["items"]["type"].upper()}
        gemini_props[pname] = sp

    gemini_params = {"type": "OBJECT", "properties": gemini_props}
    if "required" in params:
        gemini_params["required"] = params["required"]

    return types.FunctionDeclaration(
        name=fn["name"],
        description=fn["description"],
        parameters=gemini_params,
    )


GEMINI_TOOLS = [_to_gemini_decl(t) for t in engine.MEDICAL_TOOLS]


# ═══════════════════════════════════════════════════════════
#  API CALLER — Google Gemini
# ═══════════════════════════════════════════════════════════
def call_model(test_case: engine.HardTestCase) -> tuple:
    """Call Gemini and return (actual_calls: list[{name,args}], time, error)."""
    client = genai.Client(api_key=GEMINI_API_KEY)

    tools_to_send = GEMINI_TOOLS[:test_case.distractor_tools]

    try:
        t_start = time.perf_counter()
        response = client.models.generate_content(
            model=MODEL,
            contents=test_case.prompt,
            config=types.GenerateContentConfig(
                system_instruction=engine.SYSTEM_PROMPT,
                tools=[types.Tool(function_declarations=tools_to_send)],
                temperature=0.1,
                max_output_tokens=8000,
            ),
        )
        t_end = time.perf_counter()

        calls = []
        fcs = response.function_calls
        if fcs:
            for fc in fcs:
                calls.append({
                    "name": fc.name,
                    "args": dict(fc.args) if fc.args else {},
                })

        return calls, t_end - t_start, None

    except Exception as e:
        return [], 0, str(e)[:400]


def run_warmup():
    client = genai.Client(api_key=GEMINI_API_KEY)
    for _ in range(2):
        try:
            client.models.generate_content(
                model=MODEL,
                contents="Hello",
                config=types.GenerateContentConfig(
                    tools=[types.Tool(function_declarations=[GEMINI_TOOLS[0]])],
                    temperature=0.1,
                    max_output_tokens=100,
                ),
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
    json_path = f"hard_gemini_{ts}.json"
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