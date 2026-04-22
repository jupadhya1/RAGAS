"""
configs/config_loader.py — Pipeline Configuration
==================================================
Central configuration for all pipeline components.
"""

import os

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════════
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs")

# Source documents
CRA_XLSX = os.path.join(DATA_DIR, "CRA_SHELL.xlsx")
PDF_FILES = [
    os.path.join(DATA_DIR, "shell-energy-transition-strategy-2024.pdf"),
    os.path.join(DATA_DIR, "shell-annual-report-2023.pdf"),
    os.path.join(DATA_DIR, "2023-cdp-climate-change-shell-plc.pdf"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 8
DEFAULT_RETRIEVER = "hybrid"  # "bm25", "vector", "hybrid"

# ═══════════════════════════════════════════════════════════════════════════════
# DSPy DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_OPTIMIZER = "miprov2"  # "miprov2", "copro", "bootstrap", "simba"
DSPY_TRAIN_SPLIT = 0.7
DSPY_MAX_BOOTSTRAPPED = 3
DSPY_MAX_LABELED = 3

# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════
PASS_THRESHOLD = 0.45  # Composite score threshold for pass/fail
METRIC_THRESHOLDS = {
    "faithfulness": 0.40,
    "answer_relevancy": 0.40,
    "context_precision": 0.35,
    "context_recall": 0.35,
    "answer_correctness": 0.35,
    "f1_score": 0.30,
}

# ═══════════════════════════════════════════════════════════════════════════════
# CRA SECTION MAPPING
# ═══════════════════════════════════════════════════════════════════════════════
CRA_SECTIONS = {
    "0. Cover": {"icon": "🏢", "description": "Entity metadata and classification"},
    "1. Disclosures & Data Sources": {"icon": "📋", "description": "TCFD alignment, CDP score, data sources"},
    "2.Gross Physical Risk": {"icon": "🌊", "description": "Physical risk ranking, PD scenarios"},
    "3.Physical Risk Adaptation": {"icon": "🛡️", "description": "Adaptation measures, scenario assessment"},
    "4.Gross Transition Risk": {"icon": "⚡", "description": "Transition risk, temperature alignment"},
    "5.Credible Transition Plan": {"icon": "🗺️", "description": "Net-zero targets, investment, governance"},
    "6.Outcome": {"icon": "🎯", "description": "CRA score, BRAG rating"},
    "7.Risk Trigger-Addtl Analysis": {"icon": "🔔", "description": "Risk triggers, monitoring"},
}

# ═══════════════════════════════════════════════════════════════════════════════
# QUESTION CATEGORIES (from RAG_optimizer_pipeline)
# ═══════════════════════════════════════════════════════════════════════════════
QUESTION_CATEGORIES = {
    "Q1.1": "extended", "Q1.2": "simple_yes_no", "Q1.3": "extended",
    "Q1.4": "extended", "Q1.5": "extended",
    "Q2.1": "extended", "Q2.12": "extended", "Q2.13": "extended", "Q2.14": "extended",
    "Q3.1": "extended", "Q3.1.1": "extended", "Q3.1.2": "extended",
    "Q3.2": "extended", "Q3.3": "extended",
    "Q4.1": "extended", "Q4.1.1": "extended", "Q4.1.2": "extended",
    "Q4.2": "extended", "Q4.4": "extended", "Q4.5": "extended",
    "Q4.6": "extended", "Q4.7": "extended",
    "Q5.1": "extended", "Q5.2": "simple_yes_no", "Q5.3": "simple_yes_no",
    "Q5.4": "extended", "Q5.5": "dual_field",
    "Q5.5.1": "extended", "Q5.6": "multi_select",
    "Q5.6.1": "extended", "Q5.6.2": "extended",
    "Q5.7": "simple_yes_no", "Q5.8": "extended",
}

def get_category(q_id: str) -> str:
    return QUESTION_CATEGORIES.get(q_id.upper().strip(), "extended")
