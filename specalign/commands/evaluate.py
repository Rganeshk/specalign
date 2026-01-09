"""Implementation of 'specalign evaluate' command."""

import csv
import datetime
import json
from pathlib import Path
from typing import Any, Optional

import click

from specalign.llm_client import LLMClient
from specalign.utils import batch_inference, evaluate_responses_against_specs, run_task
from specalign.workspace import Workspace


def load_data_samples(data_config_path: Path, max_samples: Optional[int] = None) -> list[dict[str, Any]]:
    """Load data samples from configured dataset.

    Args:
        data_config_path: Path to data configuration file.
        max_samples: Maximum number of samples to load.

    Returns:
        List of data samples.
    """
    with open(data_config_path) as f:
        data_config = json.load(f)

    data_path = Path(data_config["path"])
    data_format = data_config.get("format", "csv")

    samples = []

    if data_format == "csv":
        with open(data_path) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_samples and i >= max_samples:
                    break
                samples.append({"prompt": row.get("prompt", row.get("input", ""))})

    elif data_format == "json":
        with open(data_path) as f:
            data = json.load(f)
            if isinstance(data, list):
                samples = data[:max_samples] if max_samples else data
            else:
                samples = [data]

    elif data_format == "jsonl":
        with open(data_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                samples.append(json.loads(line))

    return samples


def run_evaluate(
    workspace: Workspace,
    model_config_path: Path,
    data_config_path: Path,
    max_samples: Optional[int] = None,
    prompt_number: Optional[int] = None,
    max_workers: int = 32,
    eval_model_config_path: Path = None
) -> None:
    """Run the evaluate command.

    Args:
        workspace: Workspace instance.
        model_config_path: Path to model configuration.
        data_config_path: Path to data configuration.
        max_samples: Maximum number of samples to evaluate.
        prompt_number: Specific prompt number to use (latest if None).
        max_workers: Maximum number of parallel workers (default: 10).
        eval_model_config_path: Path to evaluation model configuration.
    """
    if not workspace.exists():
        click.echo("Error: Workspace not initialized. Run 'specalign init' first.", err=True)
        return

    # Get prompt to evaluate
    if prompt_number is None:
        prompt_number = workspace.get_next_prompt_number() - 1

    if prompt_number < 1:
        click.echo("Error: No prompts found. Run 'specalign compile' first.", err=True)
        return

    prompt_dir = workspace.prompts_dir / str(prompt_number)
    prompt_file = prompt_dir / "prompt.md"

    if not prompt_file.exists():
        click.echo(f"Error: Prompt #{prompt_number} not found.", err=True)
        return

    with open(prompt_file) as f:
        system_prompt = f.read()

    click.echo(f"Using prompt #{prompt_number}")

    # Load data
    click.echo(f"Loading data from {data_config_path}...")
    samples = load_data_samples(data_config_path, max_samples)
    click.echo(f"Loaded {len(samples)} samples")

    # Load specifications
    spec_files = workspace.get_all_spec_files()
    specs = {}
    for spec_file in spec_files:
        with open(spec_file) as f:
            spec_name = spec_file.stem
            specs[spec_name] = f.read()

    click.echo(f"Evaluating against {len(specs)} specifications:")
    for spec_name in specs:
        click.echo(f"  - {spec_name}")

    # Create LLM clients
    model_client = LLMClient(model_config_path=model_config_path)
    click.echo(f"Using evaluation model: {eval_model_config_path}")
    eval_client = LLMClient(model_config_path=eval_model_config_path)
    eval_model_display = str(eval_model_config_path)

    # Run evaluation
    results = {
        "run_metadata": {
            "command": f"specalign evaluate --model={model_config_path} --data={data_config_path} --max_samples={max_samples or 'all'}",
            "model_path": str(model_config_path),
            "evaluation_model": eval_model_display,
            "data_path": str(data_config_path),
            "max_samples": len(samples),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "prompt_number": prompt_number,
        },
        "specs_evaluated": list(specs.keys()),
        "samples": []
    }

    # Step 1: Generate responses for all samples in parallel
    click.echo(f"Generating responses in parallel (max {max_workers} workers)...")
    prompts = [
        sample.get("prompt", sample.get("input", ""))
        for sample in samples
    ]
    generation_args = [
        {
            "client": model_client,
            "system_prompt": system_prompt,
            "instruction": prompt,
        }
        for prompt in prompts
    ]

    generation_results = batch_inference(
        run_task,
        generation_args,
        use_process=False,
        max_workers=max_workers
    )

    # Step 2: Evaluate all samples in parallel using shared workflow
    click.echo(f"Evaluating responses in parallel (max {max_workers} workers)...")

    # Prepare responses as (user_input, model_output) tuples
    responses = list(zip(prompts, generation_results))

    # Use shared evaluation workflow
    all_evaluations = evaluate_responses_against_specs(
        responses,
        specs,
        eval_client,
        max_workers=max_workers
    )

    # Format results with sample metadata
    sample_results = [
        {
            "sample_id": sample_id,
            "input": {"prompt": prompts[sample_id]},
            "model_output": {"text": generation_results[sample_id]},
            "evaluations": all_evaluations[sample_id]
        }
        for sample_id in range(len(generation_results))
    ]

    # Add all results
    results["samples"] = sample_results

    # Calculate summary statistics
    total_samples = len(results["samples"])
    per_spec_passes = {spec_name: 0 for spec_name in specs}

    for sample in results["samples"]:
        for spec_name, eval_result in sample["evaluations"].items():
            if eval_result["score"] == 1:
                per_spec_passes[spec_name] += 1

    results["summary"] = {
        "total_samples": total_samples,
        "overall_pass_rate": sum(per_spec_passes.values()) / (total_samples * len(specs)) if total_samples > 0 else 0,
        "per_spec_pass_rate": {
            spec_name: count / total_samples if total_samples > 0 else 0
            for spec_name, count in per_spec_passes.items()
        }
    }

    # Save results
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_file = workspace.results_dir / f"eval_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    click.echo(f"\nEvaluation complete!")
    click.echo(f"Results saved to: {results_file}")
    click.echo(f"\nSummary:")
    click.echo(f"  Total samples: {total_samples}")
    click.echo(f"  Overall pass rate: {results['summary']['overall_pass_rate']:.1%}")
    click.echo(f"  Per-spec pass rates:")
    for spec_name, pass_rate in results["summary"]["per_spec_pass_rate"].items():
        click.echo(f"    - {spec_name}: {pass_rate:.1%}")
