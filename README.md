# 🏥 Clinical Tool-Call Benchmark — Gemini 3.1 Pro vs Orinn-1.6

**Hard-mode evaluation of parallel tool-calling ability for medical AI workflows.**

| Model | Overall Score | Grade | Pass Rate |
|-------|:---:|:---:|:---:|
| **Orinn-1.6** | **97%** | **A+** | 20/20 |
| Gemini 3.1 Pro | 93% | A+ | 19/20 |

---

## Why This Benchmark?

In healthcare, tool calling isn't a nice-to-have — it's the backbone of agentic clinical workflows. The model needs to pick the **right function**, with the **right arguments**, in the **right order** — every single time.

Standard benchmarks test "can the model call a tool?" We test **"can you trust it in a real clinical workflow?"**

---

## 7 Hard-Mode Test Suites (20 Tests)

| Suite | What It Tests | Why It's Hard |
|-------|---------------|---------------|
| **1. Implicit Tool Calling** | No tool names in prompt — model must reason | "Assess metabolic risk" → must infer BMI + glucose + HbA1c |
| **2. Messy Narrative Extraction** | Extract args from real doctor-speak | Rambling notes, abbreviations (PMHx, DM2, CKD3) |
| **3. Safety-First Ordering** | Check allergies/interactions BEFORE prescribing | Penicillin allergy + amoxicillin = must check first |
| **4. Tool Dependency Chains** | Tool B needs Tool A's output | eGFR → dose adjustment, history → interaction → prescription |
| **5. Distractor Resistance** | 20 tools available, only 1-3 needed | Penalizes over-calling unnecessary tools |
| **6. Repeated Tool Calls** | Same tool, different arguments | 4 drug interaction checks with different drug pairs |
| **7. Parameter Precision** | Don't swap/hallucinate argument values | Height vs weight swap, "6 weeks" → 42 days conversion |

### Scoring Formula

```
Overall = 35% tool selection + 35% argument precision + 15% over-calling penalty + 15% safety ordering
```

---

## Repository Structure

```
├── hard_benchmark_engine.py          # Shared engine (tools, scoring, dashboard, reports)
├── Hard_tool_calling_Orinn.py        # Orinn-1.6 benchmark runner
├── Hard_tool_calling_Gemini.py       # Gemini 3.1 Pro benchmark runner
├── Clinical-tool-call-Gemini-3.1-pro-vs-Orinn-1.6-result.html    #results (Orinn vs Gemini tabs)
└── README.md
```

---

## Quick Start

### Hard-Mode Benchmark (recommended)

```bash
# Install dependencies
pip install rich plotext openai google-genai

# Run Orinn benchmark
python Hard_tool_calling_Orinn.py

# Run Gemini benchmark
python Hard_tool_calling_Gemini.py
```

All 3 files must be in the same folder:
- `hard_benchmark_engine.py` — shared engine (don't run directly)
- `Hard_tool_calling_Orinn.py` — run this for Orinn
- `Hard_tool_calling_Gemini.py` — run this for Gemini

Each run produces:
- **Live terminal dashboard** with real-time scoring
- **JSON export** with full raw data
- **HTML report** with interactive charts

### Easy-Mode: Parallel Scaling (1→40 tools)

```bash
# Orinn — scales from 1 to 40 parallel tool calls
python orinn_tool_benchmark.py

# Gemini — same test, same tools, same prompts
python gemini_tool_benchmark.py
```

### Load Testing

```bash
# Live dashboard with throughput, latency, TTFT, ITL metrics
python load_test_suite.py
```

---

## Configuration

Edit the API keys and model names at the top of each runner file:

**Orinn:**
```python
BASE_URL = "https://api-call.orinn.ai/v1"
API_KEY  = "your-orinn-api-key"
MODEL    = "Orinn-1.6"
```

**Gemini:**
```python
GEMINI_API_KEY = "your-gemini-api-key"
MODEL          = "gemini-3.1-pro-preview"
```

---

## 20 Medical Tools Used

The benchmark uses 20 realistic clinical workflow tools:

**Drug & Medication:** `get_drug_info`, `check_drug_interactions`, `get_dosage_recommendation`, `check_allergy_cross_reactivity`, `generate_prescription`

**Lab & Diagnostics:** `get_lab_reference_range`, `order_lab_test`, `get_imaging_recommendation`

**Clinical Calculators:** `calculate_bmi`, `calculate_egfr`, `calculate_chadsvasc_score`

**Patient Data:** `get_differential_diagnosis`, `get_icd10_code`, `get_clinical_guidelines`, `get_patient_history`, `get_vital_signs_trend`

**Actions:** `schedule_followup`, `send_patient_message`, `create_referral`, `flag_critical_result`

---

## View Results

Open `Clinical-tool-call-Gemini-3.1-pro-vs-Orinn-1.6-result.html` in any browser to see the interactive comparison with tabs for each model.

---

**Built by [Labops Meditechnology](https://github.com/Labops-Meditechnology)**
