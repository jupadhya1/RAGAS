"""
generation/dspy_signatures.py — DSPy 3.2.0 Signatures for Shell CRA
====================================================================
Defines all typed I/O signatures for the CRA RAG pipeline.

DSPy 3.2.0 available modules:
  dspy.Predict, dspy.ChainOfThought, dspy.ProgramOfThought,
  dspy.ReAct, dspy.MultiChainComparison, dspy.Refine,
  dspy.BestOfN, dspy.CodeAct, dspy.RLM
"""

import dspy


# ═══════════════════════════════════════════════════════════════════════════════
# CORE CRA SIGNATURES
# ═══════════════════════════════════════════════════════════════════════════════

class CRABasicQA(dspy.Signature):
    """Answer a Climate Risk Assessment question using retrieved context.
    You are an expert ESG/Climate Risk analyst for Shell Petroleum Company Limited.
    Answer ONLY from the provided context. Quote specific numbers and cite sources."""
    question: str = dspy.InputField(desc="CRA questionnaire question")
    context: str = dspy.InputField(desc="Retrieved passages from Shell reports (CRA, ETS, AR, CDP)")
    answer: str = dspy.OutputField(desc="Precise answer grounded in context with source citations")


class CRAExtendedQA(dspy.Signature):
    """Provide a detailed ESG answer with evidence-based explanation.
    Ground every claim in the retrieved context. Cite page numbers and sources."""
    question: str = dspy.InputField(desc="CRA questionnaire question")
    context: str = dspy.InputField(desc="Retrieved passages from Shell reports")
    answer_options: str = dspy.InputField(desc="Available answer options (if any)")
    answer: str = dspy.OutputField(desc="Selected answer option or direct answer")
    explanation: str = dspy.OutputField(desc="Evidence-based explanation with source citations")


class CRAYesNoQA(dspy.Signature):
    """Answer a Yes/No CRA question with brief evidence."""
    question: str = dspy.InputField(desc="CRA Yes/No question")
    context: str = dspy.InputField(desc="Retrieved context passages")
    answer: str = dspy.OutputField(desc="Yes or No with qualifier")
    evidence: str = dspy.OutputField(desc="Brief justification citing source document")


class CRAMultiSelectQA(dspy.Signature):
    """Select all applicable options for a CRA multi-select question."""
    question: str = dspy.InputField(desc="CRA multi-select question")
    context: str = dspy.InputField(desc="Retrieved context passages")
    answer_options: str = dspy.InputField(desc="All available options")
    selected: str = dspy.OutputField(desc="Pipe-separated selected options, e.g. 'A|C|F'")
    justification: str = dspy.OutputField(desc="Brief justification for each selected option")


class CRADualFieldQA(dspy.Signature):
    """Answer a CRA question requiring two related fields (e.g., Q5.5 scope + horizon)."""
    question: str = dspy.InputField(desc="CRA dual-field question")
    context: str = dspy.InputField(desc="Retrieved context passages")
    answer_options: str = dspy.InputField(desc="Options for both fields")
    scope_coverage: str = dspy.OutputField(desc="Scope coverage answer")
    target_horizon: str = dspy.OutputField(desc="Target horizon answer")
    explanation: str = dspy.OutputField(desc="Justification for both fields")


class CRANumericQA(dspy.Signature):
    """Extract precise numeric data from climate risk documents.
    Return exact figures: percentages, monetary values, emission tonnes, scores."""
    question: str = dspy.InputField(desc="Question requiring numeric precision")
    context: str = dspy.InputField(desc="Retrieved context with financial/emissions data")
    answer: str = dspy.OutputField(desc="Precise numeric answer with units")
    source: str = dspy.OutputField(desc="Exact source reference (document, page, section)")


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY OPTIMIZATION SIGNATURES
# ═══════════════════════════════════════════════════════════════════════════════

class QueryRewrite(dspy.Signature):
    """Rewrite a CRA question into an optimized retrieval query.
    Focus on key entities, metrics, and domain-specific terms."""
    original_query: str = dspy.InputField(desc="Original CRA question")
    rewritten_query: str = dspy.OutputField(desc="Optimized search query for ESG documents")


class HyDE(dspy.Signature):
    """Generate a hypothetical answer to use as a retrieval query (HyDE technique).
    This creates a plausible answer passage that will match relevant documents."""
    question: str = dspy.InputField(desc="CRA question")
    hypothetical_passage: str = dspy.OutputField(desc="Hypothetical ~100 word passage answering the question")


class QueryDecompose(dspy.Signature):
    """Break a complex CRA question into simpler sub-questions for multi-hop retrieval."""
    question: str = dspy.InputField(desc="Complex CRA question")
    sub_questions: str = dspy.OutputField(desc="Numbered list of 2-4 simpler sub-questions")


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION / JUDGE SIGNATURES
# ═══════════════════════════════════════════════════════════════════════════════

class GroundednessJudge(dspy.Signature):
    """Judge whether an answer is fully grounded in the provided context.
    Check every claim against the context. Flag any unsupported statements."""
    question: str = dspy.InputField(desc="Original question")
    context: str = dspy.InputField(desc="Retrieved context")
    answer: str = dspy.InputField(desc="Candidate answer to evaluate")
    score: str = dspy.OutputField(desc="Score 0.0-1.0 as a decimal string")
    unsupported_claims: str = dspy.OutputField(desc="List of claims not supported by context, or 'None'")


class AnswerCompare(dspy.Signature):
    """Compare a generated answer against the ground truth for correctness.
    Consider semantic equivalence, not just exact string matching."""
    question: str = dspy.InputField(desc="Original question")
    ground_truth: str = dspy.InputField(desc="Expected correct answer")
    generated: str = dspy.InputField(desc="Generated answer to evaluate")
    score: str = dspy.OutputField(desc="Correctness score 0.0-1.0")
    reasoning: str = dspy.OutputField(desc="What matches and what doesn't")


# ═══════════════════════════════════════════════════════════════════════════════
# QUESTION CATEGORY ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

QUESTION_CATEGORIES = {
    "Q1.1": "extended", "Q1.2": "yes_no", "Q1.3": "extended",
    "Q1.4": "extended", "Q1.5": "extended",
    "Q2.1": "extended", "Q2.1.1": "extended", "Q2.12": "extended",
    "Q2.13": "numeric", "Q2.14": "numeric",
    "Q3.1": "extended", "Q3.1.1": "extended", "Q3.1.2": "yes_no",
    "Q3.2": "extended", "Q3.3": "extended",
    "Q4.1": "extended", "Q4.1.1": "extended", "Q4.1.2": "extended",
    "Q4.2": "extended", "Q4.4": "numeric", "Q4.5": "numeric",
    "Q4.6": "numeric", "Q4.7": "numeric",
    "Q5.1": "extended", "Q5.2": "yes_no", "Q5.3": "yes_no",
    "Q5.4": "extended", "Q5.5": "dual_field",
    "Q5.5.1": "extended", "Q5.6": "multi_select",
    "Q5.6.1": "extended", "Q5.6.2": "extended",
    "Q5.7": "yes_no", "Q5.8": "extended",
}

SIGNATURE_MAP = {
    "basic": CRABasicQA,
    "extended": CRAExtendedQA,
    "yes_no": CRAYesNoQA,
    "multi_select": CRAMultiSelectQA,
    "dual_field": CRADualFieldQA,
    "numeric": CRANumericQA,
}


def get_signature_for_question(q_id: str) -> type:
    """Get the appropriate DSPy signature for a CRA question."""
    category = QUESTION_CATEGORIES.get(q_id.upper().strip(), "extended")
    return SIGNATURE_MAP.get(category, CRAExtendedQA)


def get_category(q_id: str) -> str:
    """Get question category string."""
    return QUESTION_CATEGORIES.get(q_id.upper().strip(), "extended")
