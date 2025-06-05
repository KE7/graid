# GRAID VLM Evaluation Interface

This module provides a clean, user-friendly interface for evaluating Vision Language Models (VLMs) using SQLite databases containing questions and answers about images.

## Features

- **Multiple VLM Support**: GPT, Gemini, Llama models with various capabilities
- **Comprehensive Evaluation**: Multiple metrics and prompting strategies
- **Dual Interface**: Both command-line and programmatic access
- **Interactive Mode**: Guided interface for easy use
- **Batch Processing**: Support for efficient evaluation of large datasets

## Supported VLMs

### OpenAI GPT Models
- **GPT**: Basic GPT models (GPT-4, GPT-4o, etc.)
- **GPT_CD**: GPT with Constrained Decoding capabilities
- **GPT_CoT_CD**: GPT with Chain-of-Thought reasoning and Constrained Decoding

### Google Gemini Models  
- **Gemini**: Basic Gemini models (1.5-pro, 2.5-pro-preview)
- **Gemini_CD**: Gemini with Constrained Decoding
- **Gemini_CoT_CD**: Gemini with Chain-of-Thought and Constrained Decoding

### Meta Llama Models
- **Llama**: Basic Llama models (local deployment)
- **Llama_CD**: Llama with Constrained Decoding
- **Llama_CoT_CD**: Llama with Chain-of-Thought and Constrained Decoding

## Evaluation Metrics

- **ExactMatch**: Exact string matching between prediction and ground truth
- **Contains**: Substring matching (ground truth contained in prediction)
- **LLMJudge**: LLM-based evaluation using another model as judge

## Prompt Types

- **ZeroShotPrompt**: Direct question answering without examples
- **SetOfMarkPrompt**: Question answering with visual markers and annotations (requires GPU)
- **CoT**: Chain-of-Thought reasoning prompts (incompatible with ExactMatch metric)

## Usage

### Command Line Interface

#### Basic Usage
```bash
# Interactive mode - will prompt for database path
python -m graid.graid_cli eval-vlms

# Direct evaluation with Llama (default)
python -m graid.graid_cli eval-vlms --db-path path/to/database.sqlite

# Use GPT-4 with specific model
python -m graid.graid_cli eval-vlms --db-path path/to/db.sqlite --vlm GPT --model gpt-4o

# Chain-of-Thought with Gemini
python -m graid.graid_cli eval-vlms --db-path path/to/db.sqlite --vlm Gemini_CoT_CD --model gemini-1.5-pro --prompt CoT

# Custom sample size and metric
python -m graid.graid_cli eval-vlms --db-path path/to/db.sqlite --sample-size 50 --metric Contains
```

#### Information Commands
```bash
# List all available VLM types and models
python -m graid.graid_cli eval-vlms --list-vlms

# List available evaluation metrics
python -m graid.graid_cli eval-vlms --list-metrics

# List available prompt types
python -m graid.graid_cli eval-vlms --list-prompts
```

#### Advanced Options
```bash
# Batch processing (where supported)
python -m graid.graid_cli eval-vlms --db-path path/to/db.sqlite --vlm Gemini --batch

# Custom GPU and cloud region
python -m graid.graid_cli eval-vlms --db-path path/to/db.sqlite --vlm Gemini --region us-west1 --gpu-id 0

# Custom output directory
python -m graid.graid_cli eval-vlms --db-path path/to/db.sqlite --output-dir ./custom_results
```

### Programmatic Interface

#### Basic Function Call
```python
from graid.evaluator.eval_vlms import evaluate_vlm

# Simple evaluation with defaults
accuracy = evaluate_vlm(
    db_path="path/to/database.sqlite",
    vlm_type="Llama_CD",
    metric="LLMJudge",
    sample_size=100
)
print(f"Accuracy: {accuracy:.4f}")
```

#### Advanced Configuration
```python
from graid.evaluator.eval_vlms import evaluate_vlm

# Comprehensive evaluation setup
accuracy = evaluate_vlm(
    db_path="path/to/database.sqlite",
    vlm_type="GPT_CoT_CD",
    model_name="gpt-4o",
    metric="Contains",
    prompt="CoT",
    sample_size=200,
    region="us-central1",
    gpu_id=7,
    use_batch=False,
    output_dir="./evaluation_results"
)
```

#### Utility Functions
```python
from graid.evaluator.eval_vlms import (
    list_available_vlms,
    list_available_metrics, 
    list_available_prompts
)

# Get available options
vlms = list_available_vlms()
metrics = list_available_metrics()
prompts = list_available_prompts()

print("Available VLMs:", vlms)
print("Available metrics:", metrics)
print("Available prompts:", prompts)
```

## Configuration Requirements

### Model-Specific Requirements

#### OpenAI GPT Models
- Requires API key configuration
- Model name must be specified (gpt-4o, gpt-4.1-2025-04-14, etc.)

#### Google Gemini Models  
- Requires Google Cloud authentication
- Model name must be specified (gemini-1.5-pro, gemini-2.5-pro-preview-03-25)
- Cloud region can be customized

#### Meta Llama Models
- Requires local model deployment
- No specific model name needed (uses default deployment)

### GPU Requirements
- **SetOfMarkPrompt**: Requires GPU for visual marker processing
- Default GPU ID: 7 (can be customized with --gpu-id)

## Database Format

The SQLite database should contain:
- **Tables**: One or more tables with image data
- **Schema**: Each row should have 'key' and 'value' columns
- **Value Format**: JSON containing 'qa_list' with question-answer pairs

Example value structure:
```json
{
    "qa_list": [
        ["Where is the car located?", "on the right side"],
        ["What color is the traffic light?", "red"]
    ]
}
```

## Output Format

Results are saved in structured directories:
```
{db_name}/
├── {vlm_type}_{prompt_type}_{metric_type}/
│   ├── 1.txt  # Results for table 1
│   ├── 2.txt  # Results for table 2
│   └── ...
└── {vlm_type}_cache.db  # VLM response cache
```

Each result file contains:
- Image paths and questions
- Model predictions
- Ground truth answers  
- Correctness scores
- Evaluation metadata

## Examples

### Evaluate with Different VLMs
```bash
# Llama with Context-aware Detection
python -m graid.graid_cli eval-vlms --db-path data.sqlite --vlm Llama_CD

# GPT-4o with Chain-of-Thought
python -m graid.graid_cli eval-vlms --db-path data.sqlite --vlm GPT_CoT_CD --model gpt-4o --prompt CoT

# Gemini with visual markers
python -m graid.graid_cli eval-vlms --db-path data.sqlite --vlm Gemini --model gemini-1.5-pro --prompt SetOfMarkPrompt
```

### Batch Evaluation Script
```python
from graid.evaluator.eval_vlms import evaluate_vlm

# Evaluate multiple configurations
configs = [
    {"vlm_type": "Llama", "metric": "ExactMatch"},
    {"vlm_type": "Llama_CD", "metric": "Contains"},
    {"vlm_type": "GPT", "model_name": "gpt-4o", "metric": "LLMJudge"}
]

results = {}
for config in configs:
    accuracy = evaluate_vlm(db_path="data.sqlite", **config)
    key = f"{config['vlm_type']}_{config['metric']}"
    results[key] = accuracy

print("Evaluation Results:")
for key, accuracy in results.items():
    print(f"{key}: {accuracy:.4f}")
```

## Error Handling

The interface includes comprehensive error handling:
- **Invalid VLM types**: Clear error messages with available options
- **Missing model names**: Automatic detection and helpful prompts
- **Database issues**: File existence and format validation
- **Configuration conflicts**: Automatic compatibility checking (e.g., CoT + ExactMatch)

## Performance Tips

1. **Use caching**: VLM responses are automatically cached to avoid re-computation
2. **Batch processing**: Enable batch mode for supported models (Gemini)
3. **Sample size**: Start with smaller sample sizes for testing
4. **GPU selection**: Use appropriate GPU IDs for SetOfMarkPrompt

## Integration with Main GRAID CLI

The eval-vlms functionality is fully integrated into the main GRAID CLI:

```bash
# See all GRAID capabilities
python -m graid.graid_cli info

# Access eval-vlms directly
python -m graid.graid_cli eval-vlms --help
```

This provides a unified interface for both database generation and VLM evaluation workflows. 