# specalign

A command-line tool for generating, iterating, and maintaining LLM prompts through an evolutionary optimization process.

## Overview

specalign helps you systematically develop and optimize prompts for Large Language Models (LLMs). It provides a structured workflow for:

- Defining prompt specifications
- Compiling specifications into concrete prompts
- Generating synthetic test cases based on specifications
- Evaluating prompt performance on datasets
- Optimizing prompts using evolutionary algorithms (GEPA)

## Installation

Requires Python 3.13 or higher.

### Using uv

```bash
uv pip install specalign
```

Activate virtual environment:
```bash
source .venv/bin/activate
```

## Quick Start

### 1. Initialize a Workspace

Create a new specalign workspace in your project directory (e.g., under `examples/{task}`):

```bash
specalign init
```

This creates a `.specalign/` directory with the following structure:

```
.specalign/
 specs/      # Markdown files describing your prompt requirements
 prompts/    # Generated prompt versions (numbered)
 models/     # Model configuration YAML files
 data/       # Data configuration JSON files
 results/    # Evaluation and optimization results
 test_cases/ # Generated synthetic test cases (promptfoo format)
```

### 2. Compile a Prompt

Generate a prompt from your specifications:

```bash
specalign compile --model .specalign/models/default.yaml
```

The compiled prompt is saved in `.specalign/prompts/1/`.

### 3. Evaluate Performance

Test your prompt against a dataset:

```bash
specalign evaluate \
  --model .specalign/models/default.yaml \
  --eval-model .specalign/models/default.yaml \
  --data .specalign/data/training_data.json \
  --prompt 1 \
  --max-samples 30
```

### 4. Generate Synthetic Test Cases

Generate synthetic test cases based on your specifications:

```bash
specalign generate \
  --model .specalign/models/default.yaml \
  --count 10
```

Options:
- `--count`: Total number of test cases to generate (default: 10)
- `--per-spec`: Number of test cases per specification (overrides count distribution)
- `--output`: Custom output path (default: `.specalign/test_cases/test_cases_TIMESTAMP.yaml`)
- `--prompt`: Optional prompt number to use as context

The generated test cases are in promptfoo format and include metadata linking back to specifications. You can run them with promptfoo:

```bash
promptfoo eval -c .specalign/test_cases/test_cases_TIMESTAMP.yaml
```

### 5. Optimize with GEPA

Use evolutionary algorithms to improve your prompt:

```bash
specalign optimize \
  --model .specalign/models/default.yaml \
  --eval-model .specalign/models/default.yaml \
  --data .specalign/data/training_data.json \
  --train-samples 30 \
  --val-samples 30
```

This creates improved prompt versions based on performance metrics.
