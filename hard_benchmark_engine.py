"""
Hard Mode Tool-Calling Benchmark — Shared Engine
═══════════════════════════════════════════════════
This module contains:
  • 20 medical tool definitions (OpenAI format)
  • 7 hard-mode test suites with scoring rubrics
  • Dashboard components (Rich TUI)
  • HTML report generator
  • JSON export

Both Hard_tool_calling_Orinn.py and Hard_tool_calling_Gemini.py import this.
"""

import json
import math
import statistics
import threading
import time
import datetime
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Any

import plotext as plt
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.columns import Columns

# ═══════════════════════════════════════════════════════════
#  THEME
# ═══════════════════════════════════════════════════════════
C_ACCENT = "cyan"
C_OK     = "green"
C_WARN   = "yellow"
C_ERR    = "red"
C_DIM    = "bright_black"
C_TITLE  = "bold bright_cyan"
C_HEAD   = "bold white"

COOLDOWN_SEC = 5


# ═══════════════════════════════════════════════════════════
#  20 MEDICAL TOOLS (OpenAI format) — used by all 7 suites
# ═══════════════════════════════════════════════════════════
def _tool(name, desc, props, required=None):
    schema = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return {
        "type": "function",
        "function": {"name": name, "description": desc, "parameters": schema}
    }

MEDICAL_TOOLS = [
    _tool("get_drug_info",
          "Get detailed medical information about a drug including class, mechanism, indications, and side effects",
          {"drug_name": {"type": "string", "description": "Name of the drug"}},
          ["drug_name"]),

    _tool("check_drug_interactions",
          "Check for clinically significant interactions between two drugs",
          {"drug_a": {"type": "string", "description": "First drug"}, "drug_b": {"type": "string", "description": "Second drug"}},
          ["drug_a", "drug_b"]),

    _tool("get_dosage_recommendation",
          "Get weight/age-adjusted dosage recommendation for a drug",
          {"drug_name": {"type": "string"}, "patient_weight_kg": {"type": "number"},
           "patient_age": {"type": "integer"},
           "renal_function": {"type": "string", "enum": ["normal", "mild", "moderate", "severe"]}},
          ["drug_name", "patient_weight_kg"]),

    _tool("check_allergy_cross_reactivity",
          "Check if a proposed drug has cross-reactivity with a known allergy",
          {"known_allergy": {"type": "string", "description": "The drug the patient is allergic to"},
           "proposed_drug": {"type": "string", "description": "The drug being considered"}},
          ["known_allergy", "proposed_drug"]),

    _tool("generate_prescription",
          "Generate an electronic prescription",
          {"drug_name": {"type": "string"}, "dosage": {"type": "string"},
           "frequency": {"type": "string"}, "duration_days": {"type": "integer"},
           "patient_id": {"type": "string"}, "refills": {"type": "integer"}},
          ["drug_name", "dosage", "frequency", "duration_days", "patient_id"]),

    _tool("get_lab_reference_range",
          "Get normal reference ranges for a laboratory test",
          {"test_name": {"type": "string"}, "patient_age": {"type": "integer"},
           "patient_sex": {"type": "string", "enum": ["male", "female"]}},
          ["test_name"]),

    _tool("order_lab_test",
          "Place an order for a laboratory test",
          {"test_name": {"type": "string"}, "patient_id": {"type": "string"},
           "urgency": {"type": "string", "enum": ["routine", "urgent", "stat"]},
           "fasting_required": {"type": "boolean"}},
          ["test_name", "patient_id"]),

    _tool("calculate_bmi",
          "Calculate Body Mass Index from weight and height",
          {"weight_kg": {"type": "number", "description": "Weight in kilograms"},
           "height_cm": {"type": "number", "description": "Height in centimeters"}},
          ["weight_kg", "height_cm"]),

    _tool("calculate_egfr",
          "Calculate estimated Glomerular Filtration Rate (CKD-EPI formula)",
          {"creatinine": {"type": "number", "description": "Serum creatinine in mg/dL"},
           "age": {"type": "integer"}, "sex": {"type": "string", "enum": ["male", "female"]}},
          ["creatinine", "age", "sex"]),

    _tool("calculate_chadsvasc_score",
          "Calculate CHA2DS2-VASc stroke risk score for atrial fibrillation",
          {"age": {"type": "integer"}, "sex": {"type": "string", "enum": ["male", "female"]},
           "chf": {"type": "boolean"}, "hypertension": {"type": "boolean"},
           "stroke_history": {"type": "boolean"}, "vascular_disease": {"type": "boolean"},
           "diabetes": {"type": "boolean"}},
          ["age", "sex"]),

    _tool("get_differential_diagnosis",
          "Generate differential diagnosis list based on symptoms",
          {"symptoms": {"type": "array", "items": {"type": "string"}},
           "patient_age": {"type": "integer"},
           "patient_sex": {"type": "string", "enum": ["male", "female"]}},
          ["symptoms"]),

    _tool("get_icd10_code",
          "Look up ICD-10 diagnosis code for a condition",
          {"condition": {"type": "string"}},
          ["condition"]),

    _tool("get_clinical_guidelines",
          "Retrieve evidence-based clinical practice guidelines for a condition",
          {"condition": {"type": "string"},
           "guideline_source": {"type": "string", "description": "e.g., AHA, ACC, NICE, WHO"}},
          ["condition"]),

    _tool("get_patient_history",
          "Retrieve patient medical history from EHR",
          {"patient_id": {"type": "string"},
           "history_type": {"type": "string", "enum": ["medications", "conditions", "allergies", "surgeries", "family", "social"]}},
          ["patient_id", "history_type"]),

    _tool("get_vital_signs_trend",
          "Get patient vital signs trend over time",
          {"patient_id": {"type": "string"},
           "vital_type": {"type": "string", "enum": ["blood_pressure", "heart_rate", "temperature", "spo2", "respiratory_rate", "weight"]},
           "days_back": {"type": "integer"}},
          ["patient_id", "vital_type"]),

    _tool("get_imaging_recommendation",
          "Recommend appropriate imaging study based on clinical indication",
          {"clinical_indication": {"type": "string"}, "body_region": {"type": "string"},
           "patient_age": {"type": "integer"}, "pregnant": {"type": "boolean"}},
          ["clinical_indication", "body_region"]),

    _tool("schedule_followup",
          "Schedule a patient follow-up appointment",
          {"patient_id": {"type": "string"}, "days_from_now": {"type": "integer"},
           "reason": {"type": "string"}, "provider": {"type": "string"}},
          ["patient_id", "days_from_now", "reason"]),

    _tool("send_patient_message",
          "Send a secure message to the patient via patient portal",
          {"patient_id": {"type": "string"}, "subject": {"type": "string"},
           "message": {"type": "string"}, "urgency": {"type": "string", "enum": ["low", "normal", "high"]}},
          ["patient_id", "subject", "message"]),

    _tool("create_referral",
          "Create a specialist referral for a patient",
          {"patient_id": {"type": "string"}, "specialty": {"type": "string"},
           "reason": {"type": "string"}, "urgency": {"type": "string", "enum": ["routine", "urgent", "emergent"]}},
          ["patient_id", "specialty", "reason"]),

    _tool("flag_critical_result",
          "Flag a critical lab or imaging result for immediate physician notification",
          {"patient_id": {"type": "string"}, "result_type": {"type": "string"},
           "value": {"type": "string"}, "severity": {"type": "string", "enum": ["warning", "critical", "life_threatening"]}},
          ["patient_id", "result_type", "value", "severity"]),
]

TOOL_NAMES = [t["function"]["name"] for t in MEDICAL_TOOLS]

# System prompt — same for both models
SYSTEM_PROMPT = (
    "You are a medical AI assistant integrated into a hospital's clinical workflow system. "
    "You have access to tools for managing patient care. When clinical actions are needed, "
    "call the appropriate tools with correct arguments. Call all needed tools in parallel "
    "in a single response. Use clinical judgment to determine which tools are needed — "
    "the user may not explicitly name the tools. Pay attention to patient safety: "
    "always check allergies before prescribing, verify interactions, and flag critical values."
)


# ═══════════════════════════════════════════════════════════
#  7 HARD-MODE TEST SUITES
# ═══════════════════════════════════════════════════════════

@dataclass
class ExpectedCall:
    """What we expect the model to call, with optional arg validation."""
    tool_name: str
    required_args: dict = field(default_factory=dict)   # key: expected_value (fuzzy match)
    forbidden_args: dict = field(default_factory=dict)   # key: value that would be WRONG
    optional: bool = False       # if True, calling this is bonus points, not penalty for missing
    order_group: int = 0         # for dependency tests: lower group must come before higher


@dataclass
class HardTestCase:
    """One test case within a suite."""
    id: str
    name: str
    suite: str
    prompt: str
    expected_calls: list  # list of ExpectedCall
    distractor_tools: int = 20   # how many tools to send (20 = all)
    notes: str = ""


def _build_all_test_cases() -> list:
    """Build all 7 suites of hard test cases."""
    cases = []

    # ═══════════════════════════════════════════════════════
    # SUITE 1: IMPLICIT TOOL CALLING
    # The prompt describes a clinical need WITHOUT naming tools.
    # The model must reason about which tools to use.
    # ═══════════════════════════════════════════════════════
    cases.append(HardTestCase(
        id="IMP-1", name="Implicit: Metabolic Risk Assessment",
        suite="1. Implicit Tool Calling",
        prompt=(
            "Patient Maria (P-2001), 52-year-old female, came in today weighing 94kg at 160cm tall. "
            "She's concerned about her weight and diabetes risk. Her fasting glucose was 118 mg/dL last month. "
            "Can you assess her metabolic risk profile?"
        ),
        expected_calls=[
            ExpectedCall("calculate_bmi", required_args={"weight_kg": 94, "height_cm": 160}),
            ExpectedCall("get_lab_reference_range", required_args={"test_name": "glucose"}),
            ExpectedCall("order_lab_test", required_args={"patient_id": "P-2001", "test_name": "HbA1c"}),
            ExpectedCall("get_clinical_guidelines", required_args={"condition": "diabetes"}, optional=True),
        ],
        notes="Model must infer BMI calc + glucose ref range + HbA1c order from clinical context"
    ))

    cases.append(HardTestCase(
        id="IMP-2", name="Implicit: Stroke Risk in AFib Patient",
        suite="1. Implicit Tool Calling",
        prompt=(
            "Mr. Davis (P-2002) is a 71-year-old male with newly diagnosed atrial fibrillation. "
            "He also has a history of hypertension and type 2 diabetes. No prior strokes, "
            "no heart failure, no vascular disease. What's his stroke risk and what should we do about it?"
        ),
        expected_calls=[
            ExpectedCall("calculate_chadsvasc_score", required_args={
                "age": 71, "sex": "male", "hypertension": True, "diabetes": True,
                "stroke_history": False, "chf": False, "vascular_disease": False
            }),
            ExpectedCall("get_clinical_guidelines", required_args={"condition": "atrial fibrillation"}),
            ExpectedCall("get_drug_info", required_args={"drug_name": "warfarin"}, optional=True),
        ],
        notes="Must infer CHA2DS2-VASc calculation + guidelines from clinical narrative"
    ))

    cases.append(HardTestCase(
        id="IMP-3", name="Implicit: Worried Parent with Sick Child Symptoms",
        suite="1. Implicit Tool Calling",
        prompt=(
            "A mother brings in her 8-year-old son (P-2003) with a high fever (39.5°C), "
            "sore throat, and a rash on his trunk for 2 days. He seems quite unwell. "
            "What could this be and what tests should we run?"
        ),
        expected_calls=[
            ExpectedCall("get_differential_diagnosis", required_args={
                "symptoms": ["fever", "sore throat", "rash"],
                "patient_age": 8, "patient_sex": "male"
            }),
            ExpectedCall("order_lab_test", required_args={"patient_id": "P-2003", "urgency": "urgent"}),
            ExpectedCall("flag_critical_result", optional=True),
        ],
        notes="Must infer differential + lab order from parent's narrative"
    ))

    # ═══════════════════════════════════════════════════════
    # SUITE 2: MESSY NARRATIVE ARGUMENT EXTRACTION
    # Real doctor-speak with buried/ambiguous parameters.
    # ═══════════════════════════════════════════════════════
    cases.append(HardTestCase(
        id="NAR-1", name="Narrative: Rambling Doctor Note",
        suite="2. Messy Narrative Extraction",
        prompt=(
            "So I saw Mrs. Thompson today — she's the retired schoolteacher, you know, the one "
            "who always brings cookies. Anyway, she's 67 now, I think? Let me check... yes, 67. "
            "She's been on that metformin for years, 500mg twice a day. Her weight's crept up to "
            "about 82 kilos, she's about 5 foot 4 — what's that in centimeters, like 163? "
            "Her A1c came back at 7.8 which isn't great. Her creatinine was 1.3 last time. "
            "I want to bump her metformin to 1000mg twice daily and get her eGFR checked. "
            "Her patient ID is P-3001. Oh and she needs a 3-month follow-up with me, Dr. Patel."
        ),
        expected_calls=[
            ExpectedCall("calculate_bmi", required_args={"weight_kg": 82, "height_cm": 163}),
            ExpectedCall("calculate_egfr", required_args={"creatinine": 1.3, "age": 67, "sex": "female"}),
            ExpectedCall("generate_prescription", required_args={
                "drug_name": "metformin", "dosage": "1000mg", "frequency": "twice daily",
                "patient_id": "P-3001"
            }),
            ExpectedCall("schedule_followup", required_args={
                "patient_id": "P-3001", "days_from_now": 90, "provider": "Dr. Patel"
            }),
        ],
        notes="Must extract precise args from rambling conversational text"
    ))

    cases.append(HardTestCase(
        id="NAR-2", name="Narrative: Abbreviation-Heavy Clinical Note",
        suite="2. Messy Narrative Extraction",
        prompt=(
            "Pt P-3002, 55yo M, PMHx: HTN, DM2, CKD3. Cr 2.1, K+ 5.6. "
            "Currently on lisinopril 20mg daily, metformin 500mg BID. "
            "BP today 168/98. Wt 91kg, Ht 175cm. "
            "Need eGFR, BMI, and flag the potassium — that's dangerously high. "
            "Also check if lisinopril and metformin interact. "
            "Refer to nephrology urgent."
        ),
        expected_calls=[
            ExpectedCall("calculate_egfr", required_args={"creatinine": 2.1, "age": 55, "sex": "male"}),
            ExpectedCall("calculate_bmi", required_args={"weight_kg": 91, "height_cm": 175}),
            ExpectedCall("flag_critical_result", required_args={
                "patient_id": "P-3002", "result_type": "potassium",
                "severity": "critical"
            }),
            ExpectedCall("check_drug_interactions", required_args={
                "drug_a": "lisinopril", "drug_b": "metformin"
            }),
            ExpectedCall("create_referral", required_args={
                "patient_id": "P-3002", "specialty": "nephrology", "urgency": "urgent"
            }),
        ],
        notes="Must parse medical abbreviations and extract precise values"
    ))

    # ═══════════════════════════════════════════════════════
    # SUITE 3: SAFETY-FIRST ORDERING
    # Model must check safety BEFORE acting.
    # ═══════════════════════════════════════════════════════
    cases.append(HardTestCase(
        id="SAF-1", name="Safety: Allergy Check Before Prescribing",
        suite="3. Safety-First Ordering",
        prompt=(
            "Patient Lisa (P-4001), 45yo female, has a documented penicillin allergy. "
            "She has a UTI and I want to prescribe amoxicillin 500mg three times daily for 7 days. "
            "Go ahead and write the prescription."
        ),
        expected_calls=[
            ExpectedCall("check_allergy_cross_reactivity",
                         required_args={"known_allergy": "penicillin", "proposed_drug": "amoxicillin"},
                         order_group=1),
            ExpectedCall("generate_prescription",
                         required_args={"drug_name": "amoxicillin", "patient_id": "P-4001"},
                         order_group=2, optional=True),
        ],
        notes="MUST check allergy cross-reactivity BEFORE prescribing. Amoxicillin IS a penicillin — ideal model should warn."
    ))

    cases.append(HardTestCase(
        id="SAF-2", name="Safety: Interaction Check Before New Drug",
        suite="3. Safety-First Ordering",
        prompt=(
            "Mr. Chen (P-4002), 72yo male, is on warfarin for AFib. "
            "He sprained his back and is asking for ibuprofen for pain relief. "
            "Write him a prescription for ibuprofen 400mg three times daily for 5 days."
        ),
        expected_calls=[
            ExpectedCall("check_drug_interactions",
                         required_args={"drug_a": "warfarin", "drug_b": "ibuprofen"},
                         order_group=1),
            ExpectedCall("get_patient_history",
                         required_args={"patient_id": "P-4002", "history_type": "medications"},
                         order_group=1, optional=True),
            ExpectedCall("generate_prescription",
                         required_args={"drug_name": "ibuprofen", "patient_id": "P-4002"},
                         order_group=2, optional=True),
        ],
        notes="MUST check warfarin-ibuprofen interaction — this is a dangerous combination"
    ))

    cases.append(HardTestCase(
        id="SAF-3", name="Safety: Critical Lab Value Flagging",
        suite="3. Safety-First Ordering",
        prompt=(
            "Just got lab results back for patient P-4003. "
            "Potassium is 6.8 mEq/L, sodium is 129 mEq/L, creatinine is 4.2 mg/dL. "
            "Patient is 60yo male. Please process these results."
        ),
        expected_calls=[
            ExpectedCall("flag_critical_result", required_args={
                "patient_id": "P-4003", "result_type": "potassium",
                "severity": "life_threatening"
            }),
            ExpectedCall("flag_critical_result", required_args={
                "patient_id": "P-4003", "result_type": "sodium",
                "severity": "critical"
            }),
            ExpectedCall("flag_critical_result", required_args={
                "patient_id": "P-4003", "result_type": "creatinine",
                "severity": "critical"
            }),
            ExpectedCall("calculate_egfr", required_args={
                "creatinine": 4.2, "age": 60, "sex": "male"
            }, optional=True),
        ],
        notes="Must flag ALL three critical values, not just one"
    ))

    # ═══════════════════════════════════════════════════════
    # SUITE 4: TOOL DEPENDENCY CHAINS
    # Tool B's args depend on Tool A's output.
    # ═══════════════════════════════════════════════════════
    cases.append(HardTestCase(
        id="DEP-1", name="Dependency: eGFR → Dose Adjustment",
        suite="4. Tool Dependency Chains",
        prompt=(
            "Patient Robert (P-5001), 70yo male, creatinine 2.8 mg/dL. "
            "He needs vancomycin for his infection. First calculate his kidney function, "
            "then use that to get the right vancomycin dose. He weighs 75kg."
        ),
        expected_calls=[
            ExpectedCall("calculate_egfr",
                         required_args={"creatinine": 2.8, "age": 70, "sex": "male"},
                         order_group=1),
            ExpectedCall("get_dosage_recommendation",
                         required_args={"drug_name": "vancomycin", "patient_weight_kg": 75,
                                        "renal_function": "severe"},
                         order_group=2),
        ],
        notes="Must recognize eGFR feeds into renal_function parameter for dose adjustment"
    ))

    cases.append(HardTestCase(
        id="DEP-2", name="Dependency: History → Interaction Check → Prescription",
        suite="4. Tool Dependency Chains",
        prompt=(
            "New patient Jane (P-5002), 58yo female, transferred from another hospital. "
            "We don't have her medication list yet. She needs to start on metoprolol for her new AFib diagnosis. "
            "First pull her medication history, then check for any interactions with metoprolol, "
            "and finally write the prescription: metoprolol 25mg twice daily for 30 days."
        ),
        expected_calls=[
            ExpectedCall("get_patient_history",
                         required_args={"patient_id": "P-5002", "history_type": "medications"},
                         order_group=1),
            ExpectedCall("check_drug_interactions",
                         required_args={"drug_b": "metoprolol"},
                         order_group=2),
            ExpectedCall("generate_prescription",
                         required_args={"drug_name": "metoprolol", "dosage": "25mg",
                                        "patient_id": "P-5002"},
                         order_group=3),
        ],
        notes="Must execute in order: history → interaction check → prescription"
    ))

    # ═══════════════════════════════════════════════════════
    # SUITE 5: DISTRACTOR RESISTANCE
    # Only a few tools are relevant. Sending ALL 20.
    # ═══════════════════════════════════════════════════════
    cases.append(HardTestCase(
        id="DIS-1", name="Distractor: Simple BMI (20 tools available)",
        suite="5. Distractor Resistance",
        prompt=(
            "Patient P-6001 weighs 70kg and is 170cm tall. Calculate their BMI."
        ),
        expected_calls=[
            ExpectedCall("calculate_bmi", required_args={"weight_kg": 70, "height_cm": 170}),
        ],
        distractor_tools=20,
        notes="Only calculate_bmi should be called. Anything else is over-calling."
    ))

    cases.append(HardTestCase(
        id="DIS-2", name="Distractor: Referral Only (20 tools available)",
        suite="5. Distractor Resistance",
        prompt=(
            "Patient P-6002 needs a routine referral to dermatology for a suspicious mole. "
            "That's all — just create the referral."
        ),
        expected_calls=[
            ExpectedCall("create_referral", required_args={
                "patient_id": "P-6002", "specialty": "dermatology", "urgency": "routine"
            }),
        ],
        distractor_tools=20,
        notes="Only create_referral should be called. Test for over-triggering."
    ))

    cases.append(HardTestCase(
        id="DIS-3", name="Distractor: 3 of 20 Needed",
        suite="5. Distractor Resistance",
        prompt=(
            "I need three things for patient P-6003 (45yo female): "
            "1) Order a stat CBC, 2) Get her blood pressure trend for the last 7 days, "
            "3) Send her a message with subject 'Appointment Reminder' saying 'Please arrive 15 minutes early for your appointment tomorrow' with normal urgency. "
            "That's it, nothing else."
        ),
        expected_calls=[
            ExpectedCall("order_lab_test", required_args={
                "test_name": "CBC", "patient_id": "P-6003", "urgency": "stat"
            }),
            ExpectedCall("get_vital_signs_trend", required_args={
                "patient_id": "P-6003", "vital_type": "blood_pressure", "days_back": 7
            }),
            ExpectedCall("send_patient_message", required_args={
                "patient_id": "P-6003", "subject": "Appointment Reminder",
                "urgency": "normal"
            }),
        ],
        distractor_tools=20,
        notes="Exactly 3 tools should be called — no more, no less."
    ))

    # ═══════════════════════════════════════════════════════
    # SUITE 6: REPEATED TOOL CALLS (same tool, different args)
    # ═══════════════════════════════════════════════════════
    cases.append(HardTestCase(
        id="REP-1", name="Repeated: Check 4 Drug Interactions",
        suite="6. Repeated Tool Calls",
        prompt=(
            "Patient P-7001 is on warfarin. He's also taking aspirin, metformin, lisinopril, and omeprazole. "
            "Check the interaction between warfarin and each of his other 4 medications."
        ),
        expected_calls=[
            ExpectedCall("check_drug_interactions", required_args={"drug_a": "warfarin", "drug_b": "aspirin"}),
            ExpectedCall("check_drug_interactions", required_args={"drug_a": "warfarin", "drug_b": "metformin"}),
            ExpectedCall("check_drug_interactions", required_args={"drug_a": "warfarin", "drug_b": "lisinopril"}),
            ExpectedCall("check_drug_interactions", required_args={"drug_a": "warfarin", "drug_b": "omeprazole"}),
        ],
        notes="Must call check_drug_interactions 4 separate times with different drug_b values"
    ))

    cases.append(HardTestCase(
        id="REP-2", name="Repeated: 3 Different Lab Orders",
        suite="6. Repeated Tool Calls",
        prompt=(
            "For patient P-7002, please order the following labs: "
            "CBC (routine), Comprehensive Metabolic Panel (urgent), and Troponin I (stat). "
            "None require fasting."
        ),
        expected_calls=[
            ExpectedCall("order_lab_test", required_args={
                "test_name": "CBC", "patient_id": "P-7002", "urgency": "routine"
            }),
            ExpectedCall("order_lab_test", required_args={
                "test_name": "Comprehensive Metabolic Panel", "patient_id": "P-7002", "urgency": "urgent"
            }),
            ExpectedCall("order_lab_test", required_args={
                "test_name": "Troponin", "patient_id": "P-7002", "urgency": "stat"
            }),
        ],
        notes="Must call order_lab_test 3 times with different test_name and urgency"
    ))

    cases.append(HardTestCase(
        id="REP-3", name="Repeated: Multiple Vitals Trends",
        suite="6. Repeated Tool Calls",
        prompt=(
            "Pull the last 14 days of vitals for patient P-7003: "
            "blood pressure, heart rate, temperature, and SpO2."
        ),
        expected_calls=[
            ExpectedCall("get_vital_signs_trend", required_args={"patient_id": "P-7003", "vital_type": "blood_pressure", "days_back": 14}),
            ExpectedCall("get_vital_signs_trend", required_args={"patient_id": "P-7003", "vital_type": "heart_rate", "days_back": 14}),
            ExpectedCall("get_vital_signs_trend", required_args={"patient_id": "P-7003", "vital_type": "temperature", "days_back": 14}),
            ExpectedCall("get_vital_signs_trend", required_args={"patient_id": "P-7003", "vital_type": "spo2", "days_back": 14}),
        ],
        notes="Must call get_vital_signs_trend 4 times with different vital_type"
    ))

    # ═══════════════════════════════════════════════════════
    # SUITE 7: PARAMETER PRECISION
    # Tricky argument extraction — model must not swap/hallucinate values.
    # ═══════════════════════════════════════════════════════
    cases.append(HardTestCase(
        id="PRC-1", name="Precision: Don't Swap Height and Weight",
        suite="7. Parameter Precision",
        prompt=(
            "Patient is 172cm tall and weighs 68kg. Calculate BMI."
        ),
        expected_calls=[
            ExpectedCall("calculate_bmi",
                         required_args={"weight_kg": 68, "height_cm": 172},
                         forbidden_args={"weight_kg": 172, "height_cm": 68}),
        ],
        notes="Classic trap: model must not swap weight_kg and height_cm"
    ))

    cases.append(HardTestCase(
        id="PRC-2", name="Precision: Correct Drug Ordering in Interaction",
        suite="7. Parameter Precision",
        prompt=(
            "We need to check if there's an interaction between the patient's current warfarin "
            "and the newly proposed ibuprofen."
        ),
        expected_calls=[
            ExpectedCall("check_drug_interactions",
                         required_args={"drug_a": "warfarin", "drug_b": "ibuprofen"}),
        ],
        notes="drug_a should be the current drug, drug_b the proposed one"
    ))

    cases.append(HardTestCase(
        id="PRC-3", name="Precision: Allergy vs Proposed Drug Direction",
        suite="7. Parameter Precision",
        prompt=(
            "Patient has a sulfa allergy. We want to give them furosemide. "
            "Check if there's a cross-reactivity issue."
        ),
        expected_calls=[
            ExpectedCall("check_allergy_cross_reactivity",
                         required_args={"known_allergy": "sulfa", "proposed_drug": "furosemide"},
                         forbidden_args={"known_allergy": "furosemide", "proposed_drug": "sulfa"}),
        ],
        notes="Must not reverse: allergy=sulfa, proposed=furosemide (not the other way)"
    ))

    cases.append(HardTestCase(
        id="PRC-4", name="Precision: Complex Multi-Arg Extraction",
        suite="7. Parameter Precision",
        prompt=(
            "Patient P-8001 needs a follow-up in exactly 6 weeks with Dr. Nakamura "
            "to review her post-surgical recovery. Also send her a message with subject "
            "'Surgery Follow-up Scheduled' and message 'Your follow-up has been scheduled in 6 weeks. "
            "Please bring your medication list.' with high urgency."
        ),
        expected_calls=[
            ExpectedCall("schedule_followup", required_args={
                "patient_id": "P-8001", "days_from_now": 42,
                "provider": "Dr. Nakamura"
            }),
            ExpectedCall("send_patient_message", required_args={
                "patient_id": "P-8001",
                "subject": "Surgery Follow-up Scheduled",
                "urgency": "high"
            }),
        ],
        notes="Must convert '6 weeks' to 42 days; must get provider name and urgency right"
    ))

    return cases


ALL_TEST_CASES = _build_all_test_cases()
SUITE_NAMES = list(dict.fromkeys(tc.suite for tc in ALL_TEST_CASES))


# ═══════════════════════════════════════════════════════════
#  SCORING ENGINE
# ═══════════════════════════════════════════════════════════
def _fuzzy_match_value(expected, actual) -> bool:
    """Fuzzy match an expected value against an actual value from the model."""
    if expected is None:
        return True
    if actual is None:
        return False

    # Boolean
    if isinstance(expected, bool):
        if isinstance(actual, bool):
            return expected == actual
        return str(expected).lower() == str(actual).lower()

    # Numeric
    if isinstance(expected, (int, float)):
        try:
            actual_num = float(actual)
            return abs(expected - actual_num) < 0.5  # tolerance
        except (ValueError, TypeError):
            return False

    # String
    if isinstance(expected, str):
        exp_lower = expected.lower().strip()
        act_lower = str(actual).lower().strip()
        # Exact match
        if exp_lower == act_lower:
            return True
        # Substring match (e.g., "CBC" in "CBC with differential")
        if exp_lower in act_lower or act_lower in exp_lower:
            return True
        return False

    # List (for symptoms etc)
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        # Check if at least 60% of expected items are present
        matched = sum(1 for e in expected
                      if any(_fuzzy_match_value(e, a) for a in actual))
        return matched >= len(expected) * 0.6

    return str(expected) == str(actual)


@dataclass
class CallScore:
    """Score for a single expected tool call."""
    expected: ExpectedCall
    tool_called: bool = False
    args_correct: dict = field(default_factory=dict)  # {arg_name: True/False}
    args_forbidden_triggered: dict = field(default_factory=dict)
    tool_score: float = 0.0      # 0-100
    detail: str = ""


@dataclass
class TestScore:
    """Complete score for one test case."""
    test_id: str = ""
    test_name: str = ""
    suite: str = ""
    call_scores: list = field(default_factory=list)   # list of CallScore
    tool_selection_score: float = 0.0    # did it call the right tools?
    arg_precision_score: float = 0.0     # were the arguments correct?
    over_calling_penalty: float = 0.0    # penalty for unnecessary tool calls
    safety_order_score: float = 0.0      # did safety checks come first?
    overall_score: float = 0.0           # weighted final score
    total_time: float = 0.0
    actual_calls: list = field(default_factory=list)  # raw [{name, args}]
    error: Optional[str] = None
    done: bool = False
    phase: str = "waiting"


def score_test_result(test_case: HardTestCase,
                      actual_calls: list,  # [{name: str, args: dict}]
                      ) -> TestScore:
    """Score a model's response against the expected calls."""
    score = TestScore(
        test_id=test_case.id,
        test_name=test_case.name,
        suite=test_case.suite,
        actual_calls=actual_calls,
    )

    expected = test_case.expected_calls
    required_expected = [e for e in expected if not e.optional]
    optional_expected = [e for e in expected if e.optional]

    # ── Tool Selection Score ──
    # For each expected call, did the model call that tool?
    remaining_actuals = list(actual_calls)
    call_scores = []

    for exp in expected:
        best_match = None
        best_match_idx = -1
        best_arg_score = -1

        # Find the best matching actual call for this expected call
        for i, act in enumerate(remaining_actuals):
            if act["name"] == exp.tool_name:
                # Score args for this pairing
                arg_score = 0
                if exp.required_args:
                    matches = sum(1 for k, v in exp.required_args.items()
                                  if _fuzzy_match_value(v, act.get("args", {}).get(k)))
                    arg_score = matches / len(exp.required_args)
                else:
                    arg_score = 1.0

                if arg_score > best_arg_score:
                    best_arg_score = arg_score
                    best_match = act
                    best_match_idx = i

        cs = CallScore(expected=exp)

        if best_match is not None:
            cs.tool_called = True
            remaining_actuals.pop(best_match_idx)

            # Score individual args
            if exp.required_args:
                for k, v in exp.required_args.items():
                    actual_val = best_match.get("args", {}).get(k)
                    cs.args_correct[k] = _fuzzy_match_value(v, actual_val)

            # Check forbidden args
            if exp.forbidden_args:
                for k, v in exp.forbidden_args.items():
                    actual_val = best_match.get("args", {}).get(k)
                    cs.args_forbidden_triggered[k] = _fuzzy_match_value(v, actual_val)

            # Compute per-call score
            if exp.required_args:
                correct_count = sum(1 for v in cs.args_correct.values() if v)
                forbidden_count = sum(1 for v in cs.args_forbidden_triggered.values() if v)
                arg_pct = correct_count / len(exp.required_args) * 100
                # Penalize forbidden
                penalty = forbidden_count * 25
                cs.tool_score = max(0, arg_pct - penalty)
            else:
                cs.tool_score = 100.0
        else:
            cs.tool_score = 0.0

        call_scores.append(cs)

    score.call_scores = call_scores

    # ── Aggregate: Tool Selection ──
    required_called = sum(1 for cs in call_scores
                          if not cs.expected.optional and cs.tool_called)
    if required_expected:
        score.tool_selection_score = required_called / len(required_expected) * 100
    else:
        score.tool_selection_score = 100.0

    # ── Aggregate: Arg Precision ──
    all_arg_scores = [cs.tool_score for cs in call_scores if cs.tool_called]
    score.arg_precision_score = statistics.mean(all_arg_scores) if all_arg_scores else 0

    # ── Over-calling Penalty ──
    extra_calls = len(remaining_actuals)  # calls that didn't match any expected
    total_expected = len(required_expected)
    if total_expected > 0 and extra_calls > 0:
        score.over_calling_penalty = min(extra_calls / total_expected * 30, 30)  # max 30% penalty

    # ── Safety/Order Score ──
    # Check that lower order_groups are called before higher ones
    order_correct = True
    has_ordering = any(e.order_group > 0 for e in expected)
    if has_ordering:
        called_order = []
        for act in actual_calls:
            for cs in call_scores:
                if cs.tool_called and cs.expected.tool_name == act["name"]:
                    called_order.append(cs.expected.order_group)
                    break

        # Check monotonicity
        for i in range(1, len(called_order)):
            if called_order[i] < called_order[i-1]:
                order_correct = False
                break

        score.safety_order_score = 100.0 if order_correct else 30.0
    else:
        score.safety_order_score = 100.0  # no ordering requirement

    # ── Overall Score ──
    # Weighted: 35% tool selection + 35% arg precision + 15% over-calling + 15% ordering
    score.overall_score = (
        score.tool_selection_score * 0.35
        + score.arg_precision_score * 0.35
        + (100 - score.over_calling_penalty) * 0.15
        + score.safety_order_score * 0.15
    )

    return score


# ═══════════════════════════════════════════════════════════
#  DATA FOR SHARED STATE
# ═══════════════════════════════════════════════════════════
all_scores: list[TestScore] = []
current_score: Optional[TestScore] = None
scores_lock = threading.Lock()
current_phase = "idle"
current_test_name = ""
progress_text = ""
model_name = ""
model_color = "cyan"


# ═══════════════════════════════════════════════════════════
#  PLOTEXT
# ═══════════════════════════════════════════════════════════
def make_bar(title, labels, vals, color="cyan", w=56, h=9):
    plt.clear_figure(); plt.theme("dark"); plt.plot_size(w, h); plt.title(title)
    if not vals or all(v == 0 for v in vals):
        plt.plot([0], [0]); return plt.build()
    plt.bar(labels, vals, color=color, width=0.5)
    return plt.build()


# ═══════════════════════════════════════════════════════════
#  TERMINAL DASHBOARD
# ═══════════════════════════════════════════════════════════
def build_header():
    t1 = Text()
    t1.append("🔬 ", style="bright_red")
    t1.append(model_name, style=C_TITLE)
    t1.append(" — Hard Mode Tool-Calling Benchmark", style=C_HEAD)
    t2 = Text()
    t2.append(f"7 Suites  │  {len(ALL_TEST_CASES)} Tests  │  ", style=C_DIM)
    t2.append("Implicit • Narrative • Safety • Dependencies • Distractors • Repetition • Precision", style=C_ACCENT)
    return Panel(Group(Align.center(t1), Align.center(t2)), border_style=model_color, padding=(0, 1))


def build_status():
    if current_phase == "idle":
        return Text("  ⏸  Preparing...", style=C_DIM)
    elif current_phase == "warmup":
        return Text("  🔥  Warming up...", style=C_WARN)
    elif current_phase == "running":
        bar = Text()
        bar.append(f"  🚀  {current_test_name}", style="bold bright_green")
        if progress_text:
            bar.append(f"  │  {progress_text}", style=C_DIM)
        return bar
    elif current_phase == "cooldown":
        return Text("  ❄️  Cooldown...", style=C_WARN)
    return Text("  ✅  Benchmark complete!", style="bold bright_green")


def build_current_detail():
    s = current_score
    if not s or not s.done:
        return Text("")
    if s.error:
        return Text(f"  ✗ {s.test_id} — ERROR: {s.error[:120]}", style=C_ERR)

    parts = []
    summary = Text()
    oc = C_OK if s.overall_score >= 70 else (C_WARN if s.overall_score >= 40 else C_ERR)
    summary.append(f"  {s.test_id} → ", style=C_HEAD)
    summary.append(f"{s.overall_score:.0f}%", style=oc)
    summary.append(f"  │  Tools: {s.tool_selection_score:.0f}%", style=C_OK if s.tool_selection_score >= 80 else C_WARN)
    summary.append(f"  │  Args: {s.arg_precision_score:.0f}%", style=C_OK if s.arg_precision_score >= 80 else C_WARN)
    summary.append(f"  │  OverCall: -{s.over_calling_penalty:.0f}%", style=C_ERR if s.over_calling_penalty > 10 else C_DIM)
    summary.append(f"  │  Order: {s.safety_order_score:.0f}%", style=C_OK if s.safety_order_score >= 80 else C_ERR)
    summary.append(f"  │  {s.total_time:.1f}s", style=C_DIM)
    parts.append(summary)

    # Per-tool detail
    detail = Text("  ")
    for cs in s.call_scores:
        if cs.expected.optional:
            if cs.tool_called:
                detail.append(f" +{cs.expected.tool_name[:12]} ", style=C_OK)
        elif cs.tool_called:
            detail.append(f" ✓{cs.expected.tool_name[:12]}({cs.tool_score:.0f}%) ", style=C_OK if cs.tool_score >= 70 else C_WARN)
        else:
            detail.append(f" ✗{cs.expected.tool_name[:12]} ", style=C_ERR)
    parts.append(detail)

    return Panel(Group(*parts), border_style=C_DIM, title=f"[{C_DIM}]Latest[/]")


def build_results_table():
    done_scores = [s for s in all_scores if s.done]
    if not done_scores:
        return Text("  Waiting...", style=C_DIM)

    table = Table(title=f"[{C_TITLE}]{model_name} — Hard Mode Results[/]",
                  border_style=C_DIM, show_lines=True, expand=True, padding=(0, 1))
    table.add_column("ID", width=6, style="bold")
    table.add_column("Test Name", width=38)
    table.add_column("Overall", width=8, justify="right")
    table.add_column("Tools", width=7, justify="right")
    table.add_column("Args", width=7, justify="right")
    table.add_column("Over\nCall", width=7, justify="right")
    table.add_column("Order", width=7, justify="right")
    table.add_column("Time", width=6, justify="right")
    table.add_column("Verdict", width=8, justify="center")

    for s in done_scores:
        if s.error:
            table.add_row(s.test_id, s.test_name, "—","—","—","—","—","—", Text("ERROR", style=f"bold {C_ERR}"))
            continue
        oc = C_OK if s.overall_score >= 70 else (C_WARN if s.overall_score >= 40 else C_ERR)
        tc = C_OK if s.tool_selection_score >= 80 else (C_WARN if s.tool_selection_score >= 50 else C_ERR)
        ac = C_OK if s.arg_precision_score >= 80 else (C_WARN if s.arg_precision_score >= 50 else C_ERR)
        pc = C_ERR if s.over_calling_penalty > 15 else (C_WARN if s.over_calling_penalty > 5 else C_DIM)
        sc = C_OK if s.safety_order_score >= 80 else C_ERR
        v = "PASS" if s.overall_score >= 60 else "FAIL"
        vc = f"bold {C_OK}" if v == "PASS" else f"bold {C_ERR}"
        table.add_row(
            s.test_id, s.test_name,
            Text(f"{s.overall_score:.0f}%", style=oc),
            Text(f"{s.tool_selection_score:.0f}%", style=tc),
            Text(f"{s.arg_precision_score:.0f}%", style=ac),
            Text(f"-{s.over_calling_penalty:.0f}%", style=pc),
            Text(f"{s.safety_order_score:.0f}%", style=sc),
            f"{s.total_time:.1f}s",
            Text(v, style=vc),
        )
    return table


def build_suite_summary():
    done_scores = [s for s in all_scores if s.done and not s.error]
    if not done_scores:
        return Text("")

    suite_data = defaultdict(list)
    for s in done_scores:
        suite_data[s.suite].append(s)

    table = Table(title=f"[{C_TITLE}]Suite Summary[/]",
                  border_style=C_DIM, show_lines=True, expand=True, padding=(0, 1))
    table.add_column("Suite", style="bold", width=30)
    table.add_column("Tests", width=6, justify="right")
    table.add_column("Avg Score", width=10, justify="right")
    table.add_column("Avg Tools", width=10, justify="right")
    table.add_column("Avg Args", width=10, justify="right")
    table.add_column("Grade", width=7, justify="center")

    for suite_name in SUITE_NAMES:
        if suite_name not in suite_data:
            continue
        scores = suite_data[suite_name]
        avg_overall = statistics.mean([s.overall_score for s in scores])
        avg_tools = statistics.mean([s.tool_selection_score for s in scores])
        avg_args = statistics.mean([s.arg_precision_score for s in scores])

        if avg_overall >= 90: grade, gc = "A+", C_OK
        elif avg_overall >= 80: grade, gc = "A", C_OK
        elif avg_overall >= 70: grade, gc = "B", C_WARN
        elif avg_overall >= 60: grade, gc = "C", C_WARN
        elif avg_overall >= 50: grade, gc = "D", C_ERR
        else: grade, gc = "F", C_ERR

        oc = C_OK if avg_overall >= 70 else (C_WARN if avg_overall >= 50 else C_ERR)
        table.add_row(
            suite_name, str(len(scores)),
            Text(f"{avg_overall:.0f}%", style=oc),
            Text(f"{avg_tools:.0f}%", style=C_OK if avg_tools >= 80 else C_WARN),
            Text(f"{avg_args:.0f}%", style=C_OK if avg_args >= 80 else C_WARN),
            Text(grade, style=f"bold {gc}"),
        )
    return table


def build_scorecard():
    done = [s for s in all_scores if s.done and not s.error]
    if not done:
        return Text("")

    avg = statistics.mean([s.overall_score for s in done])
    avg_tools = statistics.mean([s.tool_selection_score for s in done])
    avg_args = statistics.mean([s.arg_precision_score for s in done])
    avg_order = statistics.mean([s.safety_order_score for s in done])
    pass_ct = sum(1 for s in done if s.overall_score >= 60)

    def card(label, value, style):
        t = Text()
        t.append(f"\n  {value}\n", style=f"bold {style}")
        t.append(f"  {label}\n", style=C_DIM)
        return Panel(t, border_style=C_DIM, width=20)

    ac = C_OK if avg >= 70 else (C_WARN if avg >= 50 else C_ERR)
    return Columns([
        card("Overall", f"{avg:.0f}%", ac),
        card("Tool Select", f"{avg_tools:.0f}%", C_OK if avg_tools >= 80 else C_WARN),
        card("Arg Precision", f"{avg_args:.0f}%", C_OK if avg_args >= 80 else C_WARN),
        card("Safety Order", f"{avg_order:.0f}%", C_OK if avg_order >= 80 else C_ERR),
        card("Pass Rate", f"{pass_ct}/{len(done)}", C_OK if pass_ct == len(done) else C_WARN),
    ], expand=True, equal=True, padding=(0, 0))


def build_charts():
    done = [s for s in all_scores if s.done and not s.error]
    if len(done) < 3:
        return Text("  Charts appear after 3+ tests.", style=C_DIM)

    # Suite average chart
    suite_data = defaultdict(list)
    for s in done:
        suite_data[s.suite].append(s.overall_score)

    labels = [sn.split(".")[0].strip() for sn in SUITE_NAMES if sn in suite_data]
    vals = [statistics.mean(suite_data[sn]) for sn in SUITE_NAMES if sn in suite_data]

    c1 = make_bar("Avg Score by Suite", labels, vals, color="green", w=80, h=10)
    return Panel(Text(c1), border_style=C_DIM, expand=True)


def build_dashboard():
    parts = [build_header(), build_status()]
    sc = build_scorecard()
    if isinstance(sc, Columns):
        parts.append(sc)
    if current_score and current_score.done:
        parts.append(build_current_detail())
    done = [s for s in all_scores if s.done]
    if done:
        parts.append(build_results_table())
        parts.append(build_suite_summary())
        parts.append(build_charts())
    return Group(*parts)


# ═══════════════════════════════════════════════════════════
#  JSON EXPORT
# ═══════════════════════════════════════════════════════════
def export_json(filepath):
    done = [s for s in all_scores if s.done]
    good = [s for s in done if not s.error]
    avg_overall = statistics.mean([s.overall_score for s in good]) if good else 0

    suite_avgs = {}
    sd = defaultdict(list)
    for s in good:
        sd[s.suite].append(s.overall_score)
    for sn, vals in sd.items():
        suite_avgs[sn] = round(statistics.mean(vals), 1)

    data = {
        "title": f"{model_name} Hard Mode Tool-Calling Benchmark",
        "timestamp": datetime.datetime.now().isoformat(),
        "model": model_name,
        "total_tests": len(ALL_TEST_CASES),
        "overall_score": round(avg_overall, 1),
        "suite_scores": suite_avgs,
        "results": [],
    }
    for s in done:
        data["results"].append({
            "id": s.test_id, "name": s.test_name, "suite": s.suite,
            "overall": round(s.overall_score, 1),
            "tool_selection": round(s.tool_selection_score, 1),
            "arg_precision": round(s.arg_precision_score, 1),
            "over_calling_penalty": round(s.over_calling_penalty, 1),
            "safety_order": round(s.safety_order_score, 1),
            "time": round(s.total_time, 2),
            "calls": s.actual_calls,
            "error": s.error,
            "verdict": "PASS" if s.overall_score >= 60 else "FAIL",
            "call_details": [{
                "tool": cs.expected.tool_name,
                "called": cs.tool_called,
                "score": round(cs.tool_score, 1),
                "args_correct": {k: v for k, v in cs.args_correct.items()},
                "optional": cs.expected.optional,
            } for cs in s.call_scores],
        })
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    return filepath


# ═══════════════════════════════════════════════════════════
#  HTML REPORT
# ═══════════════════════════════════════════════════════════
def generate_html_report(json_path):
    with open(json_path) as f:
        data = json.load(f)

    results = data["results"]
    mn = data.get("model", "")
    ts = data.get("timestamp", "")
    overall = data.get("overall_score", 0)
    suite_scores = data.get("suite_scores", {})

    results_js = json.dumps(results)
    suite_scores_js = json.dumps(suite_scores)

    good = [r for r in results if not r.get("error")]
    pass_ct = sum(1 for r in good if r.get("verdict") == "PASS")

    # Grade
    if overall >= 90: grade = "A+"
    elif overall >= 80: grade = "A"
    elif overall >= 70: grade = "B"
    elif overall >= 60: grade = "C"
    elif overall >= 50: grade = "D"
    else: grade = "F"

    is_gemini = "gemini" in mn.lower()
    gradient = "linear-gradient(135deg, #4285f4, #34a853, #fbbc04, #ea4335)" if is_gemini else "linear-gradient(135deg, #f87171, #fbbf24, #34d399, #22d3ee, #c084fc)"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{mn} — Hard Mode Tool-Calling Benchmark</title>
<meta property="og:title" content="{mn} scores {overall:.0f}% on Hard Mode Tool-Calling (Grade: {grade})">
<meta property="og:description" content="7-suite benchmark: Implicit reasoning, messy narratives, safety ordering, dependency chains, distractor resistance, repeated calls, parameter precision. {pass_ct}/{len(good)} tests passed.">
<meta property="og:type" content="article">
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
  :root {{ --bg:#08090f; --bg2:#0f1219; --bg3:#171d2b; --bg4:#1e2640; --border:#2a3352; --text:#d0d8f0; --dim:#4a5478; --cyan:#22d3ee; --green:#34d399; --yellow:#fbbf24; --red:#f87171; --magenta:#c084fc; --blue:#60a5fa; --indigo:#818cf8; }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Outfit',sans-serif; background:var(--bg); color:var(--text); padding:0; line-height:1.6; }}
  code,.mono {{ font-family:'IBM Plex Mono',monospace; }}
  .hero {{ text-align:center; padding:3rem 2rem 2rem; background:linear-gradient(180deg,var(--bg3) 0%,var(--bg) 100%); border-bottom:1px solid var(--border); }}
  .hero .badge {{ display:inline-block; padding:0.3rem 1rem; border-radius:100px; background:var(--bg4); border:1px solid var(--border); font-size:0.75rem; color:var(--dim); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:1rem; }}
  .hero h1 {{ font-size:2.8rem; font-weight:900; background:{gradient}; -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0.3rem; }}
  .hero .grade {{ font-size:4rem; font-weight:900; font-family:'IBM Plex Mono',monospace; color:{'var(--green)' if overall>=70 else 'var(--red)'}; }}
  .hero .subtitle {{ color:var(--dim); font-size:0.9rem; margin-top:0.5rem; }}
  .scores {{ display:grid; grid-template-columns:repeat(5,1fr); gap:0; max-width:1200px; margin:-1.5rem auto 2rem; background:var(--bg2); border:1px solid var(--border); border-radius:16px; overflow:hidden; position:relative; z-index:1; }}
  .score-card {{ padding:1.2rem; text-align:center; border-right:1px solid var(--border); }}
  .score-card:last-child {{ border-right:none; }}
  .score-card .value {{ font-size:1.8rem; font-weight:800; font-family:'IBM Plex Mono',monospace; }}
  .score-card .label {{ font-size:0.65rem; color:var(--dim); text-transform:uppercase; letter-spacing:0.1em; margin-top:0.2rem; }}
  .content {{ max-width:1200px; margin:0 auto; padding:0 2rem 2rem; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-bottom:1.5rem; }}
  .card {{ background:var(--bg2); border:1px solid var(--border); border-radius:12px; padding:1.5rem; }}
  .card:hover {{ border-color:var(--cyan); }}
  .card h3 {{ font-size:0.8rem; font-weight:600; color:var(--dim); text-transform:uppercase; letter-spacing:0.05em; margin-bottom:1rem; padding-bottom:0.5rem; border-bottom:1px solid var(--border); }}
  .plotly-chart {{ width:100%; height:320px; }}
  .table-card {{ grid-column:1/-1; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.8rem; }}
  th {{ padding:0.5rem 0.7rem; text-align:right; font-weight:600; color:var(--dim); border-bottom:2px solid var(--border); }}
  th:first-child,th:nth-child(2) {{ text-align:left; }}
  td {{ padding:0.5rem 0.7rem; text-align:right; border-bottom:1px solid var(--border); font-family:'IBM Plex Mono',monospace; font-size:0.75rem; }}
  td:first-child,td:nth-child(2) {{ text-align:left; font-family:'Outfit',sans-serif; }}
  tr:hover td {{ background:var(--bg3); }}
  .pass {{ color:var(--green); font-weight:700; }}
  .fail {{ color:var(--red); font-weight:700; }}
  .warn {{ color:var(--yellow); }}
  .detail-toggle {{ cursor:pointer; color:var(--cyan); font-size:0.7rem; }}
  .detail-panel {{ display:none; background:var(--bg3); border-radius:8px; padding:0.8rem; margin:0.5rem 0; font-size:0.7rem; }}
  .detail-panel.open {{ display:block; }}
  .arg-ok {{ color:var(--green); }} .arg-fail {{ color:var(--red); }}
  .footer {{ text-align:center; padding:2rem; color:var(--dim); font-size:0.8rem; border-top:1px solid var(--border); margin-top:2rem; }}
  @media (max-width:768px) {{ .hero h1 {{ font-size:1.6rem; }} .scores {{ grid-template-columns:repeat(3,1fr); }} .grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="hero">
  <div class="badge">Hard Mode — 7 Suites — Medical AI</div>
  <h1>{mn}</h1>
  <div class="grade">{grade}</div>
  <div class="subtitle">{len(results)} tests across 7 suites &nbsp;│&nbsp; {pass_ct}/{len(good)} passed &nbsp;│&nbsp; Overall: {overall:.0f}%</div>
</div>
<div class="scores" id="top-scores"></div>
<div class="content">
<div class="grid"><div class="card table-card"><h3>All Test Results</h3><table><thead><tr><th style="text-align:left">ID</th><th style="text-align:left">Test</th><th>Overall</th><th>Tools</th><th>Args</th><th>OverCall</th><th>Order</th><th>Time</th><th>Verdict</th></tr></thead><tbody id="results-tbody"></tbody></table></div></div>
<div class="grid">
  <div class="card"><h3>Score by Suite</h3><div id="chart-suites" class="plotly-chart"></div></div>
  <div class="card"><h3>Score Breakdown per Test</h3><div id="chart-breakdown" class="plotly-chart"></div></div>
</div>
<div class="grid"><div class="card table-card"><h3>Per-Tool Call Detail</h3><div id="detail-container"></div></div></div>
</div>
<div class="footer">Generated by <strong>{mn} Hard Mode Benchmark</strong> &nbsp;│&nbsp; {ts[:10]}</div>
<script>
const R={results_js};
const SS={suite_scores_js};
const lo={{paper_bgcolor:'#0f1219',plot_bgcolor:'#0f1219',font:{{color:'#d0d8f0',family:'Outfit'}},margin:{{t:20,b:50,l:50,r:20}},xaxis:{{gridcolor:'#2a3352'}},yaxis:{{gridcolor:'#2a3352'}},legend:{{bgcolor:'transparent'}}}};
const cfg={{responsive:true,displayModeBar:false}};
// Top scores
const good=R.filter(r=>!r.error);
const avgO=good.length?good.reduce((a,r)=>a+r.overall,0)/good.length:0;
const avgT=good.length?good.reduce((a,r)=>a+r.tool_selection,0)/good.length:0;
const avgA=good.length?good.reduce((a,r)=>a+r.arg_precision,0)/good.length:0;
const avgS=good.length?good.reduce((a,r)=>a+r.safety_order,0)/good.length:0;
const passCt=good.filter(r=>r.verdict==='PASS').length;
const sc=document.getElementById('top-scores');
[['{overall:.0f}%','Overall',avgO>=70?'var(--green)':'var(--red)'],[avgT.toFixed(0)+'%','Tool Selection',avgT>=80?'var(--green)':'var(--yellow)'],[avgA.toFixed(0)+'%','Arg Precision',avgA>=80?'var(--green)':'var(--yellow)'],[avgS.toFixed(0)+'%','Safety Order',avgS>=80?'var(--green)':'var(--red)'],[passCt+'/'+good.length,'Pass Rate',passCt===good.length?'var(--green)':'var(--yellow)']].forEach(([v,l,c])=>{{sc.innerHTML+=`<div class="score-card"><div class="value" style="color:${{c}}">${{v}}</div><div class="label">${{l}}</div></div>`;}});
// Table
const tbody=document.getElementById('results-tbody');
R.forEach((r,i)=>{{
  if(r.error){{tbody.innerHTML+=`<tr><td>${{r.id}}</td><td>${{r.name}}</td><td colspan="7" class="fail">ERROR</td></tr>`;return;}}
  const oc=r.overall>=70?'pass':(r.overall>=40?'warn':'fail');
  const vc=r.verdict==='PASS'?'pass':'fail';
  tbody.innerHTML+=`<tr><td>${{r.id}}</td><td>${{r.name}}</td><td class="${{oc}}">${{r.overall}}%</td><td>${{r.tool_selection}}%</td><td>${{r.arg_precision}}%</td><td>-${{r.over_calling_penalty}}%</td><td>${{r.safety_order}}%</td><td>${{r.time.toFixed(1)}}s</td><td class="${{vc}}">${{r.verdict}}</td></tr>`;
}});
// Charts
const suiteLabels=Object.keys(SS).map(s=>s.split('.')[0].trim());
const suiteVals=Object.values(SS);
Plotly.newPlot('chart-suites',[{{x:suiteLabels,y:suiteVals,type:'bar',marker:{{color:suiteVals.map(v=>v>=70?'#34d399':(v>=50?'#fbbf24':'#f87171'))}}}}],{{...lo,yaxis:{{...lo.yaxis,title:'Avg Score %',range:[0,105]}}}},cfg);
const ids=good.map(r=>r.id);
Plotly.newPlot('chart-breakdown',[
  {{x:ids,y:good.map(r=>r.tool_selection),name:'Tools',type:'bar',marker:{{color:'#22d3ee'}}}},
  {{x:ids,y:good.map(r=>r.arg_precision),name:'Args',type:'bar',marker:{{color:'#c084fc'}}}},
  {{x:ids,y:good.map(r=>r.safety_order),name:'Order',type:'bar',marker:{{color:'#fbbf24'}}}},
],{{...lo,barmode:'group',yaxis:{{...lo.yaxis,title:'Score %',range:[0,105]}},showlegend:true}},cfg);
// Detail
const dc=document.getElementById('detail-container');
R.filter(r=>!r.error).forEach(r=>{{
  let h=`<div style="margin:1rem 0 0.5rem;color:var(--cyan);font-weight:600">${{r.id}} — ${{r.name}}</div>`;
  (r.call_details||[]).forEach(cd=>{{
    const icon=cd.called?'✓':'✗';const color=cd.called?(cd.score>=70?'var(--green)':'var(--yellow)'):'var(--red)';
    h+=`<div style="margin:2px 0;font-size:0.72rem"><span style="color:${{color}}">${{icon}}</span> <span class="mono">${{cd.tool}}</span> <span style="color:var(--dim)">${{cd.score}}%${{cd.optional?' (bonus)':''}}</span>`;
    if(cd.called&&Object.keys(cd.args_correct).length>0){{
      h+=' — ';
      Object.entries(cd.args_correct).forEach(([k,v])=>{{h+=`<span class="${{v?'arg-ok':'arg-fail'}}">${{k}}:${{v?'✓':'✗'}}</span> `;}});
    }}
    h+='</div>';
  }});
  dc.innerHTML+=h;
}});
</script>
</body>
</html>"""

    rp = json_path.replace(".json", "_report.html")
    with open(rp, "w") as f:
        f.write(html)
    return rp