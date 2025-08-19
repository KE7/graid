"""
GRAID VLM Evaluation Module

This module provides functionality to evaluate Vision Language Models (VLMs) 
using SQLite databases containing questions and answers.

Supported VLMs:
- GPT models (GPT-4, GPT-4o, etc.) with optional Chain-of-Thought (CoT)
- Gemini models (1.5-pro, 2.5-pro-preview) with optional CoT
- Llama models with optional CoT
- Each model supports Constrained Decoding (CD) variants

Supported Metrics:
- ExactMatch: Exact string matching
- Contains: Substring matching
- LLMJudge: LLM-based evaluation

Supported Prompts:
- ZeroShotPrompt: Direct question answering
- SetOfMarkPrompt: With visual markers/annotations
- CoT: Chain-of-Thought reasoning

Usage:
    # As a function
    accuracy = evaluate_vlm(
        db_path="path/to/database.sqlite",
        vlm_type="Llama_CD",
        metric="LLMJudge",
        prompt="ZeroShotPrompt",
        sample_size=100
    )
    
    # From command line
    python -m graid.evaluator.eval_vlms --db-path path/to/db.sqlite --vlm Llama_CD
"""

import argparse
import ast
import json
import os
import random
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sqlitedict import SqliteDict
from tqdm import tqdm

# Import evaluation components
from graid.evaluator.metrics import Contains, ExactMatch, LLMJudge
from graid.evaluator.prompts import (
    CoT,
    SetOfMarkPrompt,
    ZeroShotPrompt,
    ZeroShotPrompt_batch,
)
from graid.evaluator.vlms import (
    GPT,
    GPT_CD,
    Gemini,
    Gemini_CD,
    Gemini_CoT_CD,
    GPT_CoT_CD,
    Llama,
    Llama_CD,
    Llama_CoT_CD,
)
from graid.utilities.common import project_root_dir

# Set random seed for reproducibility
random.seed(42)

# Default paths
DB_PATH = project_root_dir() / "data/databases_ablations"
BDD_PATH = project_root_dir() / "data/bdd_val_filtered"
NU_PATH = project_root_dir() / "data/nuimages_val_filtered"
WAYMO_PATH = project_root_dir() / "data/waymo_validation_interesting"

# Configuration for VLM types
VLM_CONFIGS = {
    "GPT": {
        "class": GPT,
        "requires_model_selection": True,
        "models": ["gpt-4.1-2025-04-14", "o4-mini-2025-04-16", "gpt-4o"],
        "supports_batch": False,
        "description": "OpenAI GPT models",
    },
    "GPT_CD": {
        "class": GPT_CD,
        "requires_model_selection": True,
        "models": ["gpt-4.1-2025-04-14", "o4-mini-2025-04-16", "gpt-4o"],
        "supports_batch": False,
        "description": "OpenAI GPT with Constrained Decoding",
    },
    "GPT_CoT_CD": {
        "class": GPT_CoT_CD,
        "requires_model_selection": True,
        "models": ["gpt-4.1-2025-04-14", "o4-mini-2025-04-16", "gpt-4o"],
        "supports_batch": False,
        "description": "OpenAI GPT with Chain-of-Thought and Constrained Decoding",
    },
    "Gemini": {
        "class": Gemini,
        "requires_model_selection": True,
        "models": ["gemini-1.5-pro", "gemini-2.5-pro-preview-03-25"],
        "supports_batch": True,
        "description": "Google Gemini models",
    },
    "Gemini_CD": {
        "class": Gemini_CD,
        "requires_model_selection": True,
        "models": ["gemini-1.5-pro", "gemini-2.5-pro-preview-03-25"],
        "supports_batch": False,
        "description": "Google Gemini with Constrained Decoding",
    },
    "Gemini_CoT_CD": {
        "class": Gemini_CoT_CD,
        "requires_model_selection": True,
        "models": ["gemini-1.5-pro", "gemini-2.5-pro-preview-03-25"],
        "supports_batch": False,
        "description": "Google Gemini with Chain-of-Thought and Constrained Decoding",
    },
    "Llama": {
        "class": Llama,
        "requires_model_selection": False,
        "models": [],
        "supports_batch": False,
        "description": "Meta Llama models",
    },
    "Llama_CD": {
        "class": Llama_CD,
        "requires_model_selection": False,
        "models": [],
        "supports_batch": False,
        "description": "Meta Llama with Constrained Decoding",
    },
    "Llama_CoT_CD": {
        "class": Llama_CoT_CD,
        "requires_model_selection": False,
        "models": [],
        "supports_batch": False,
        "description": "Meta Llama with Chain-of-Thought and Constrained Decoding",
    },
}

METRIC_CONFIGS = {
    "ExactMatch": {
        "class": ExactMatch,
        "description": "Exact string matching evaluation",
    },
    "Contains": {"class": Contains, "description": "Substring matching evaluation"},
    "LLMJudge": {
        "class": LLMJudge,
        "description": "LLM-based evaluation using another model as judge",
    },
}

PROMPT_CONFIGS = {
    "ZeroShotPrompt": {
        "class": ZeroShotPrompt,
        "batch_class": ZeroShotPrompt_batch,
        "supports_cd": True,
        "description": "Direct question answering without examples",
    },
    "SetOfMarkPrompt": {
        "class": SetOfMarkPrompt,
        "batch_class": None,
        "supports_cd": False,
        "requires_gpu": True,
        "description": "Question answering with visual markers and annotations",
    },
    "CoT": {
        "class": CoT,
        "batch_class": None,
        "supports_cd": False,
        "incompatible_metrics": ["ExactMatch"],
        "description": "Chain-of-Thought reasoning prompts",
    },
}


def list_available_vlms() -> dict[str, list[str]]:
    """
    List all available VLM types and their models.

    Returns:
        Dictionary with VLM types as keys and model lists as values
    """
    result = {}
    for vlm_type, config in VLM_CONFIGS.items():
        if config["requires_model_selection"]:
            result[vlm_type] = config["models"]
        else:
            result[vlm_type] = ["default"]
    return result


def list_available_metrics() -> list[str]:
    """
    List all available evaluation metrics.

    Returns:
        List of metric names
    """
    return list(METRIC_CONFIGS.keys())


def list_available_prompts() -> list[str]:
    """
    List all available prompt types.

    Returns:
        List of prompt names
    """
    return list(PROMPT_CONFIGS.keys())


def create_vlm(
    vlm_type: str, model_name: Optional[str] = None, region: str = "us-central1"
) -> Any:
    """
    Create a VLM instance based on type and model name.

    Args:
        vlm_type: Type of VLM (e.g., "GPT", "Llama_CD")
        model_name: Specific model name (required for some VLM types)
        region: Cloud region for certain models (default: us-central1)

    Returns:
        Configured VLM instance

    Raises:
        ValueError: If VLM type is unknown or model_name is required but not provided
    """
    if vlm_type not in VLM_CONFIGS:
        raise ValueError(
            f"Unknown VLM type: {vlm_type}. Available: {list(VLM_CONFIGS.keys())}"
        )

    config = VLM_CONFIGS[vlm_type]
    vlm_class = config["class"]

    if config["requires_model_selection"]:
        if not model_name:
            raise ValueError(
                f"Model name required for {vlm_type}. Available models: {config['models']}"
            )

        if "Gemini" in vlm_type:
            return vlm_class(model_name=model_name, location=region)
        else:
            return vlm_class(model_name=model_name)
    else:
        return vlm_class()


def create_metric(metric_type: str) -> Any:
    """
    Create a metric instance based on type.

    Args:
        metric_type: Type of metric (e.g., "LLMJudge", "ExactMatch")

    Returns:
        Configured metric instance

    Raises:
        ValueError: If metric type is unknown
    """
    if metric_type not in METRIC_CONFIGS:
        raise ValueError(
            f"Unknown metric type: {metric_type}. Available: {list(METRIC_CONFIGS.keys())}"
        )

    return METRIC_CONFIGS[metric_type]["class"]()


def create_prompt(
    prompt_type: str, vlm_type: str, use_batch: bool = False, gpu_id: int = 7
) -> Any:
    """
    Create a prompt instance based on type and VLM.

    Args:
        prompt_type: Type of prompt (e.g., "ZeroShotPrompt", "CoT")
        vlm_type: Type of VLM for compatibility checking
        use_batch: Whether to use batch processing
        gpu_id: GPU ID for prompts that require GPU

    Returns:
        Configured prompt instance

    Raises:
        ValueError: If prompt type is unknown or incompatible
    """
    if prompt_type not in PROMPT_CONFIGS:
        raise ValueError(
            f"Unknown prompt type: {prompt_type}. Available: {list(PROMPT_CONFIGS.keys())}"
        )

    config = PROMPT_CONFIGS[prompt_type]

    if prompt_type == "SetOfMarkPrompt":
        if config.get("requires_gpu", False):
            return config["class"](gpu=gpu_id)
        else:
            return config["class"]()

    elif prompt_type == "CoT":
        if use_batch:
            raise ValueError("CoT does not support batch processing")
        return config["class"]()

    elif prompt_type == "ZeroShotPrompt":
        if use_batch and config["batch_class"]:
            return config["batch_class"]()
        else:
            using_cd = "CD" in vlm_type if config["supports_cd"] else False
            return config["class"](using_cd=using_cd)

    else:
        return config["class"]()


def validate_configuration(
    vlm_type: str, metric_type: str, prompt_type: str
) -> tuple[bool, Optional[str]]:
    """
    Validate that the combination of VLM, metric, and prompt is compatible.

    Args:
        vlm_type: Type of VLM
        metric_type: Type of metric
        prompt_type: Type of prompt

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if prompt is incompatible with metric
    prompt_config = PROMPT_CONFIGS.get(prompt_type, {})
    incompatible_metrics = prompt_config.get("incompatible_metrics", [])

    if metric_type in incompatible_metrics:
        suggested_metric = "Contains" if metric_type == "ExactMatch" else "LLMJudge"
        return (
            False,
            f"{prompt_type} is incompatible with {metric_type}. Try {suggested_metric} instead.",
        )

    return True, None


def _determine_dataset_path(db_path: str) -> Path:
    """Determine the dataset base path from database path."""
    if "bdd" in db_path.lower():
        return BDD_PATH
    elif "nuimage" in db_path.lower():
        return NU_PATH
    else:
        return WAYMO_PATH


def _create_output_directory(
    db_path: str, vlm_type: str, prompt_type: str, metric_type: str
) -> Path:
    """Create and return the output directory for results."""
    path_parts = db_path.split("/")
    db_name = (
        "_".join([path_parts[-2], path_parts[-1]])
        if len(path_parts) > 1
        else path_parts[-1]
    )

    output_dir = Path(db_name.split(".py")[0])
    results_dir = output_dir / f"{vlm_type}_{prompt_type}_{metric_type}"
    output_dir.mkdir(parents=True, exist_ok=True)

    return results_dir


def _load_database_tables(db_path: str) -> dict[str, pd.DataFrame]:
    """Load all tables from SQLite database."""
    conn = sqlite3.connect(db_path)

    # Get all table names
    tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql(tables_query, conn)["name"].tolist()

    # Load all tables
    dataframes = {}
    for table in tables:
        df = pd.read_sql(f"SELECT * FROM '{table}'", conn)
        dataframes[table] = df

    conn.close()
    return dataframes


def _filter_and_sample_data(
    dataframes: dict[str, pd.DataFrame], sample_size: int
) -> dict[str, tuple[pd.DataFrame, int]]:
    """Filter and sample data from database tables."""
    sampled_dataframes = {}
    print("Filtering rows...")

    for table_name, df in dataframes.items():
        filtered_rows = []
        for img_idx in tqdm(range(len(df)), desc=f"Processing {table_name}"):
            row = df.iloc[img_idx]
            d = row.to_dict()

            _, v = d["key"], json.loads(d["value"])
            qa_list = v.get("qa_list", None)

            if not qa_list or qa_list == "Question not applicable":
                continue

            if isinstance(qa_list[0], list):
                qa_list = [random.choice(qa_list)]

            filtered_rows.append(row)

        filtered_df = pd.DataFrame(filtered_rows).reset_index(drop=True)
        available_samples = len(filtered_df)

        if available_samples >= sample_size:
            sampled_df = filtered_df.sample(n=sample_size, random_state=42).reset_index(
                drop=True
            )
        else:
            print(
                f"Table '{table_name}' has only {available_samples} valid rows. Using all."
            )
            sampled_df = filtered_df.copy()

        sampled_dataframes[table_name] = (sampled_df, available_samples)

    return sampled_dataframes


def evaluate_vlm(
    db_path: str,
    vlm_type: str = "Llama",
    model_name: Optional[str] = None,
    metric: str = "LLMJudge",
    prompt: str = "ZeroShotPrompt",
    sample_size: int = 100,
    region: str = "us-central1",
    gpu_id: int = 7,
    use_batch: bool = False,
    output_dir: Optional[str] = None,
) -> float:
    """
    Evaluate a Vision Language Model using a SQLite database.

    Args:
        db_path: Path to the SQLite database containing questions and answers
        vlm_type: Type of VLM to use (default: "Llama")
        model_name: Specific model name (required for some VLM types)
        metric: Evaluation metric to use (default: "LLMJudge")
        prompt: Prompt type to use (default: "ZeroShotPrompt")
        sample_size: Number of samples per table (default: 100)
        region: Cloud region for certain models (default: "us-central1")
        gpu_id: GPU ID for GPU-requiring prompts (default: 7)
        use_batch: Whether to use batch processing (default: False)
        output_dir: Custom output directory (optional)

    Returns:
        Average accuracy across all evaluation samples

    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If database file doesn't exist
    """
    # Validate inputs
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")

    is_valid, error_msg = validate_configuration(vlm_type, metric, prompt)
    if not is_valid:
        raise ValueError(error_msg)

    # Create components
    my_vlm = create_vlm(vlm_type, model_name, region)
    my_metric = create_metric(metric)
    my_prompt = create_prompt(prompt, vlm_type, use_batch, gpu_id)

    # Handle metric compatibility issues
    if prompt == "CoT" and metric == "ExactMatch":
        print("Warning: CoT cannot use ExactMatch, switching to Contains metric.")
        my_metric = create_metric("Contains")

    # Set up paths and directories
    db_base_path = _determine_dataset_path(db_path)

    if output_dir:
        results_dir = Path(output_dir)
    else:
        results_dir = _create_output_directory(db_path, vlm_type, prompt, metric)

    results_dir.mkdir(parents=True, exist_ok=True)

    # Set up VLM cache
    vlm_cache_loc = results_dir.parent / f"{vlm_type}_cache.db"
    vlm_cache = SqliteDict(
        str(vlm_cache_loc),
        tablename="vlm_cache",
        autocommit=True,
        encode=json.dumps,
        decode=json.loads,
    )

    try:
        # Load and process data
        dataframes = _load_database_tables(db_path)
        sampled_dataframes = _filter_and_sample_data(dataframes, sample_size)

        # Determine consistent sample size
        min_available_samples = sample_size
        for _, (_, available_samples) in sampled_dataframes.items():
            min_available_samples = min(min_available_samples, available_samples)

        min_available_samples = max(1, min_available_samples)
        print(
            f"\nUsing consistent sample size of {min_available_samples} across all tables\n"
        )

        all_correctness = []

        # Process each table
        table_idx = 0
        for table_name, (sampled_df, _) in sorted(sampled_dataframes.items()):
            table_idx += 1

            # Check for existing results
            output_path = results_dir / f"{table_idx}.txt"

            if os.path.exists(output_path):
                print(f"Loading existing results from {output_path}")
                with open(output_path, "r") as f:
                    text = f.read()
                    match = re.search(r"Correctness:\s*\n(.*?)\n", text)
                    if match:
                        existing_scores = ast.literal_eval(match.group(1))
                        if isinstance(existing_scores, list):
                            if len(existing_scores) >= min_available_samples:
                                all_correctness.extend(
                                    existing_scores[:min_available_samples]
                                )
                                continue

            # Get the dataframe for the current table
            df_to_process, _ = sampled_dataframes[table_name]

            # Limit to min_available_samples
            df_to_process = df_to_process.head(min_available_samples)

            # Process new samples if needed
            print(f"Processing table {table_idx}: {table_name}")
            
            # Lists to store data for this table
            questions, answers, preds, correctness = [], [], [], []
            
            for _, row in tqdm(df_to_process.iterrows(), total=len(df_to_process)):
                d = row.to_dict()
                image_path, v = d["key"], json.loads(d["value"])
                
                # Construct full image path
                image_path = str(db_base_path / image_path)

                qa_list = v.get("qa_list", [])
                if not qa_list or qa_list == "Question not applicable":
                    continue

                qa_pair = random.choice(qa_list) if isinstance(qa_list[0], list) else qa_list
                q, a = qa_pair[0], qa_pair[1]
                
                # Generate prompt and unique cache key
                annotated_image, messages = my_prompt.generate_prompt(image_path, q)
                cache_key = f"{vlm_type}_{prompt}_{image_path}_{str(messages)}"
                
                if cache_key in vlm_cache:
                    pred = vlm_cache[cache_key]
                else:
                    pred, _ = my_vlm.generate_answer(annotated_image, messages)
                    vlm_cache[cache_key] = pred
                
                correct = my_metric.evaluate(pred, a)
                
                questions.append(q)
                answers.append(a)
                preds.append(pred)
                correctness.append(correct)

            all_correctness.extend(correctness)

            # Save results
            with open(str(output_path), "w") as log_file:
                log_file.write(f"Table: {table_name}\n")
                log_file.write(f"VLM: {vlm_type}\n")
                log_file.write(f"Metric: {metric}\n")
                log_file.write(f"Prompt: {prompt}\n")
                log_file.write(f"Sample Size: {len(correctness)}\n")
                log_file.write(f"Correctness: \n{correctness}\n")
                log_file.write(f"Questions: \n{questions}\n")
                log_file.write(f"Answers: \n{answers}\n")
                log_file.write(f"Predictions: \n{preds}\n")

        # Calculate and return final accuracy
        if len(all_correctness) == 0:
            print("Warning: No correctness scores found!")
            return 0.0

        final_accuracy = np.mean(all_correctness)
        print(f"Final accuracy: {final_accuracy:.4f}")
        return final_accuracy

    finally:
        vlm_cache.close()


def main():
    """Command line interface for VLM evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Vision Language Models using SQLite databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation with Llama
  %(prog)s --db-path path/to/database.sqlite
  
  # Evaluate with GPT-4 and Chain-of-Thought
  %(prog)s --db-path path/to/db.sqlite --vlm GPT_CoT_CD --model gpt-4o --prompt CoT
  
  # Custom sample size and metric
  %(prog)s --db-path path/to/db.sqlite --vlm Gemini_CD --metric Contains --sample-size 50
  
  # List available options
  %(prog)s --list-vlms
  %(prog)s --list-metrics
  %(prog)s --list-prompts
        """,
    )

    # Information commands
    parser.add_argument(
        "--list-vlms", action="store_true", help="List available VLM types"
    )
    parser.add_argument(
        "--list-metrics", action="store_true", help="List available metrics"
    )
    parser.add_argument(
        "--list-prompts", action="store_true", help="List available prompts"
    )

    # Main arguments
    parser.add_argument("--db-path", type=str, help="Path to SQLite database")
    parser.add_argument(
        "--vlm",
        type=str,
        default="Llama",
        choices=list(VLM_CONFIGS.keys()),
        help="VLM type to use",
    )
    parser.add_argument(
        "--model", type=str, help="Specific model name (required for some VLMs)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="LLMJudge",
        choices=list(METRIC_CONFIGS.keys()),
        help="Evaluation metric",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="ZeroShotPrompt",
        choices=list(PROMPT_CONFIGS.keys()),
        help="Prompt type",
    )
    parser.add_argument(
        "--sample-size", "-n", type=int, default=100, help="Sample size per table"
    )
    parser.add_argument(
        "--region", type=str, default="us-central1", help="Cloud region"
    )
    parser.add_argument("--gpu-id", type=int, default=7, help="GPU ID")
    parser.add_argument("--batch", action="store_true", help="Use batch processing")
    parser.add_argument("--output-dir", type=str, help="Custom output directory")

    args = parser.parse_args()

    # Handle information commands
    if args.list_vlms:
        print("Available VLM types:")
        for vlm_type, config in VLM_CONFIGS.items():
            print(f"  {vlm_type}: {config['description']}")
            if config["requires_model_selection"]:
                print(f"    Available models: {', '.join(config['models'])}")
        return

    if args.list_metrics:
        print("Available metrics:")
        for metric_type, config in METRIC_CONFIGS.items():
            print(f"  {metric_type}: {config['description']}")
        return

    if args.list_prompts:
        print("Available prompts:")
        for prompt_type, config in PROMPT_CONFIGS.items():
            print(f"  {prompt_type}: {config['description']}")
        return

    # Validate required arguments
    if not args.db_path:
        parser.error("--db-path is required")

    # Check if model name is required
    vlm_config = VLM_CONFIGS[args.vlm]
    if vlm_config["requires_model_selection"] and not args.model:
        print(f"Model selection required for {args.vlm}.")
        print(f"Available models: {', '.join(vlm_config['models'])}")
        parser.error(f"--model is required for {args.vlm}")

    try:
        # Run evaluation
        accuracy = evaluate_vlm(
            db_path=args.db_path,
            vlm_type=args.vlm,
            model_name=args.model,
            metric=args.metric,
            prompt=args.prompt,
            sample_size=args.sample_size,
            region=args.region,
            gpu_id=args.gpu_id,
            use_batch=args.batch,
            output_dir=args.output_dir,
        )

        print(f"\nFinal Results:")
        print(f"Database: {args.db_path}")
        print(f"VLM: {args.vlm}" + (f" ({args.model})" if args.model else ""))
        print(f"Metric: {args.metric}")
        print(f"Prompt: {args.prompt}")
        print(f"Sample Size: {args.sample_size}")
        print(f"Accuracy: {accuracy:.4f}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
