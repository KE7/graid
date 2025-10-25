"""
Profiling utilities for GRAID question generation statistics.

This module provides reusable functions for logging and formatting
profiling statistics from question generation processes.
"""

import logging

logger = logging.getLogger(__name__)


def log_profiling_statistics(question_stats, title="Question Processing Statistics"):
    """
    Log profiling statistics to console using consistent formatting.

    Args:
        question_stats: Dictionary containing detailed profiling statistics
        title: Title for the statistics table
    """
    if not question_stats or "detailed_stats" not in question_stats:
        return

    logger.info(f"📊 {title}:")
    logger.info("=" * 95)

    # Table header
    logger.info(
        f"{'Question Type':<40} {'is_app(ms)':<12} {'apply(ms)':<12} {'Hit Rate':<10} {'Failed':<8} {'Total QA':<10}"
    )
    logger.info("-" * 95)

    # Table rows
    for qtype, stats in question_stats["detailed_stats"].items():
        # Calculate averages
        is_app_time, is_app_count = stats.get("is_applicable_time", (0, 1))
        apply_time, apply_count = stats.get("apply_time", (0, 1))
        is_app_avg = (is_app_time / max(is_app_count, 1)) * 1000
        apply_avg = (apply_time / max(apply_count, 1)) * 1000

        # Calculate success metrics
        is_applicable_count = stats.get("is_applicable_true_count", 0)
        empty_results = stats.get("apply_empty_results", 0)
        total_qa_generated = stats.get("total_qa_generated", 0)
        successful_cases = is_applicable_count - empty_results
        hit_rate = (
            (successful_cases / max(is_applicable_count, 1)) * 100
            if is_applicable_count > 0
            else 0
        )

        question_text = stats.get("question_text", qtype)
        # Truncate for console alignment
        question_text_short = question_text[:39]
        logger.info(
            f"{question_text_short:<40} {is_app_avg:<12.2f} {apply_avg:<12.2f} {hit_rate:<10.1f}% {empty_results:<8} {total_qa_generated:<10}"
        )

    logger.info("=" * 95)
    logger.info("Notes: Hit Rate = % of applicable cases that generated ≥1 QA pair")
    logger.info(
        "       Failed = cases where is_applicable=True but apply returned no QA pairs"
    )


def format_profiling_table(question_stats, format_type="markdown"):
    """
    Format profiling statistics table in either markdown or console format.

    Args:
        question_stats: Dictionary containing detailed profiling statistics
        format_type: "markdown" for README tables, "console" for logging

    Returns:
        Formatted table as string
    """
    if not question_stats or "detailed_stats" not in question_stats:
        return ""

    if format_type == "markdown":
        # Markdown table format for README
        table_lines = [
            "| Question Type | is_applicable Avg (ms) | apply Avg (ms) | Predicate -> QA Hit Rate | Empty cases |",
            "|---------------|------------------------|----------------|--------------------------|-------------|",
        ]
    else:
        # Console table format for logging
        table_lines = [
            f"{'Question Type':<40} {'is_app(ms)':<12} {'apply(ms)':<12} {'Hit Rate':<10} {'Failed':<8} {'Total QA':<10}",
            "-" * 95,
        ]

    for qtype, stats in question_stats["detailed_stats"].items():
        # Calculate averages
        is_app_time, is_app_count = stats.get("is_applicable_time", (0, 1))
        apply_time, apply_count = stats.get("apply_time", (0, 1))
        is_app_avg = (is_app_time / max(is_app_count, 1)) * 1000
        apply_avg = (apply_time / max(apply_count, 1)) * 1000

        # Calculate success metrics
        is_applicable_count = stats.get("is_applicable_true_count", 0)
        empty_results = stats.get("apply_empty_results", 0)
        total_qa_generated = stats.get("total_qa_generated", 0)
        successful_cases = is_applicable_count - empty_results
        hit_rate = (
            (successful_cases / max(is_applicable_count, 1)) * 100
            if is_applicable_count > 0
            else 0
        )

        question_text = stats.get("question_text", qtype)

        if format_type == "markdown":
            table_lines.append(
                f"| {question_text} | {is_app_avg:.2f} | {apply_avg:.2f} | {hit_rate:.1f}% | {empty_results} |"
            )
        else:
            # Truncate for console alignment
            question_text_short = question_text[:39]
            table_lines.append(
                f"{question_text_short:<40} {is_app_avg:<12.2f} {apply_avg:<12.2f} {hit_rate:<10.1f}% {empty_results:<8} {total_qa_generated:<10}"
            )

    return "\n".join(table_lines)


def format_profiling_notes(format_type="markdown"):
    """
    Format profiling explanation notes.

    Args:
        format_type: "markdown" for README, "console" for logging

    Returns:
        Formatted notes as string
    """
    if format_type == "markdown":
        return """\n**Notes:**
- `is_applicable` checks if a question type can be applied to an image
- `apply` generates the actual question-answer pairs
- Predicate -> QA Hit Rate = Percentage of applicable cases that generated at least one QA pair
- Empty cases = Number of times is_applicable=True but apply returned no QA pairs"""
    else:
        return """Notes: Hit Rate = % of applicable cases that generated ≥1 QA pair
       Failed = cases where is_applicable=True but apply returned no QA pairs"""
