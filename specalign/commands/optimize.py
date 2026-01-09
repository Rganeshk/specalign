"""Implementation of 'specalign optimize' command."""

import datetime
from pathlib import Path
from typing import Optional

import click
from gepa import optimize, EvaluationBatch, GEPAAdapter

from specalign.llm_client import LLMClient
from specalign.utils import batch_inference, evaluate_responses_against_specs, run_task
from specalign.workspace import Workspace
from specalign.commands.evaluate import load_data_samples


class SpecalignGEPAAdapter(GEPAAdapter):
    """GEPA adapter for specalign prompt optimization."""

    def __init__(self, specs: dict[str, str], n_concurrent: int = 6):
        """Initialize adapter.

        Args:
            specs: Dict mapping spec names to spec content.
            n_concurrent: Number of concurrent evaluations (unused, kept for compatibility).
        """
        self.specs = specs
        self.n_concurrent = n_concurrent

    def evaluate(
        self,
        batch: list[dict],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Evaluate a batch of examples with a candidate prompt.

        Args:
            batch: List of examples with 'prompt', 'lm', 'eval_lm' keys.
            candidate: Dict with 'system_prompt' key.
            capture_traces: Whether to capture detailed traces.

        Returns:
            EvaluationBatch with outputs, scores, and trajectories.
        """
        outputs = []
        scores = []
        trajectories = []

        student_model = batch[0]["lm"]
        eval_model = batch[0]["eval_lm"]

        # Phase 1: Generate responses
        student_results = batch_inference(
            run_task,
            [
                {
                    "client": student_model,
                    "system_prompt": candidate["system_prompt"],
                    "instruction": example["prompt"],
                }
                for example in batch
            ],
            use_process=False
        )

        # Phase 2: Evaluate responses
        responses = [
            (example["prompt"], student_response)
            for example, student_response in zip(batch, student_results)
        ]
        eval_results = evaluate_responses_against_specs(
            responses,
            self.specs,
            eval_model
        )

        # Collect results
        for example, student_response, per_spec_results in zip(batch, student_results, eval_results):
            total_score = sum(result["score"] for result in per_spec_results.values())
            avg_score = total_score / len(self.specs) if self.specs else 0
            analysis = {
                "average_score": avg_score,
                "per_spec_results": per_spec_results,
                "passed_specs": [name for name, res in per_spec_results.items() if res["score"] == 1],
                "failed_specs": [name for name, res in per_spec_results.items() if res["score"] == 0],
            }
            output = {
                "student_response": student_response,
                "analysis": analysis,
            }
            trajectory = {
                "messages": [
                    {"role": "system", "content": candidate["system_prompt"]},
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": student_response},
                ],
                "score": avg_score,
                "analysis": analysis,
            }
            outputs.append(output)
            scores.append(avg_score)
            trajectories.append(trajectory)
        print(scores)

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ):
        """Create reflective dataset for prompt optimization.

        Args:
            candidate: Current candidate prompt.
            eval_batch: Evaluation results.
            components_to_update: Components to optimize (e.g., ['system_prompt']).

        Returns:
            Dict with reflective examples for each component.
        """
        reflective_dataset = {"system_prompt": []}

        for score, trajectory in zip(eval_batch.scores, eval_batch.trajectories):
            # Create feedback summary
            analysis = trajectory["analysis"]
            feedback_lines = [
                f"Average Score: {analysis['average_score']:.2f}",
                f"Passed Specs: {', '.join(analysis['passed_specs']) or 'None'}",
                f"Failed Specs: {', '.join(analysis['failed_specs']) or 'None'}",
                "",
                "Detailed Feedback:",
            ]

            for spec_name, result in analysis["per_spec_results"].items():
                feedback_lines.append(f"- {spec_name}: {result['rationale']}")
                if result["violations"]:
                    feedback_lines.append(f"  Violations: {', '.join(result['violations'])}")

            feedback = "\n".join(feedback_lines)

            reflective_dataset["system_prompt"].append(
                {
                    "Message History": trajectory["messages"],
                    "System Prompt": candidate["system_prompt"],
                    "Score": score,
                    "Feedback": feedback,
                }
            )

        return reflective_dataset


def run_optimize(
    workspace: Workspace,
    model_config_path: Path,
    data_config_path: Path,
    train_samples: int,
    val_samples: Optional[int] = None,
    prompt_number: Optional[int] = None,
    max_metric_calls: int = 150,
    reflection_lm: str = "openai/gpt-5.2",
    eval_model_config_path: Path = None,
    use_wandb: bool = False,
) -> None:
    """Run the optimize command using GEPA.

    Args:
        workspace: Workspace instance.
        model_config_path: Path to model configuration.
        data_config_path: Path to data configuration.
        train_samples: Number of training samples.
        val_samples: Number of validation samples.
        prompt_number: Specific prompt number to use as seed (latest if None).
        max_metric_calls: Maximum number of optimization iterations.
        reflection_lm: LLM to use for reflection/optimization.
        eval_model_config_path: Path to evaluation model configuration.
        use_wandb: Whether to use Weights & Biases logging.
    """
    if not workspace.exists():
        click.echo("Error: Workspace not initialized. Run 'specalign init' first.", err=True)
        return

    # Get seed prompt
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
        seed_prompt_text = f.read()

    click.echo(f"Using prompt #{prompt_number} as seed")

    # Load data
    click.echo(f"Loading data from {data_config_path}...")
    all_samples = load_data_samples(data_config_path)

    if len(all_samples) < train_samples:
        click.echo(f"Error: Dataset has only {len(all_samples)} samples, but {train_samples} training samples requested.", err=True)
        return

    if val_samples is None:
        val_samples = min(train_samples, len(all_samples) - train_samples)

    if len(all_samples) < train_samples + val_samples:
        click.echo(f"Error: Dataset has only {len(all_samples)} samples, but {train_samples + val_samples} total samples requested.", err=True)
        return

    click.echo(f"Using {train_samples} training samples and {val_samples} validation samples")

    # Load specifications
    spec_files = workspace.get_all_spec_files()
    specs = {}
    for spec_file in spec_files:
        with open(spec_file) as f:
            spec_name = spec_file.stem
            specs[spec_name] = f.read()

    click.echo(f"Optimizing against {len(specs)} specifications:")
    for spec_name in specs:
        click.echo(f"  - {spec_name}")

    # Create LLM clients
    click.echo("Initializing language models...")

    # Student model (the one being evaluated)
    model_config = {}
    with open(model_config_path) as f:
        import yaml
        model_config = yaml.safe_load(f)

    student_model_name = model_config["model"]["name"]
    student_lm = LLMClient(model_config_path=model_config_path)

    # Eval model (for evaluating responses)
    click.echo(f"Using evaluation model: {eval_model_config_path}")
    eval_lm = LLMClient(model_config_path=eval_model_config_path)

    # Prepare datasets
    trainset = [
        {
            "prompt": sample.get("prompt", sample.get("input", "")),
            "lm": student_lm,
            "eval_lm": eval_lm,
        }
        for sample in all_samples[:train_samples]
    ]

    valset = [
        {
            "prompt": sample.get("prompt", sample.get("input", "")),
            "lm": student_lm,
            "eval_lm": eval_lm,
        }
        for sample in all_samples[train_samples:train_samples + val_samples]
    ]

    # Create seed candidate
    seed_candidate = {"system_prompt": seed_prompt_text}

    # Create adapter
    adapter = SpecalignGEPAAdapter(specs=specs)

    # Run GEPA optimization
    click.echo(f"\nStarting GEPA optimization (max {max_metric_calls} iterations)...")
    click.echo(f"Reflection LLM: {reflection_lm}")

    gepa_result = optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        max_metric_calls=max_metric_calls,
        reflection_lm=reflection_lm,
        use_wandb=use_wandb,
    )

    # Save optimized prompt
    new_prompt_number = workspace.get_next_prompt_number()
    new_prompt_dir = workspace.prompts_dir / str(new_prompt_number)
    new_prompt_dir.mkdir(parents=True, exist_ok=True)

    optimized_prompt_file = new_prompt_dir / "prompt.md"
    with open(optimized_prompt_file, "w") as f:
        f.write(gepa_result.best_candidate["system_prompt"])

    # Save config
    config = {
        "optimization": {
            "method": "GEPA",
            "seed_prompt_number": prompt_number,
            "model": student_model_name,
            "reflection_lm": reflection_lm,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "max_metric_calls": max_metric_calls,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "best_score": float(gepa_result.best_score) if hasattr(gepa_result, 'best_score') else None,
        }
    }

    config_file = new_prompt_dir / "config.yaml"
    with open(config_file, "w") as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)

    # Save optimization log
    log_lines = [
        f"GEPA Prompt Optimization",
        f"=" * 50,
        f"Timestamp: {config['optimization']['timestamp']}",
        f"Seed Prompt: #{prompt_number}",
        f"Model: {student_model_name}",
        f"Reflection LLM: {reflection_lm}",
        f"Training Samples: {train_samples}",
        f"Validation Samples: {val_samples}",
        f"Max Iterations: {max_metric_calls}",
        f"Specifications: {', '.join(specs.keys())}",
        f"",
        f"Optimized Prompt:",
        f"-" * 50,
        gepa_result.best_candidate["system_prompt"],
    ]

    log_file = new_prompt_dir / "optimization.log"
    with open(log_file, "w") as f:
        f.write("\n".join(log_lines))

    click.echo(f"\nOptimization complete!")
    click.echo(f"Optimized prompt saved to: {optimized_prompt_file}")
    click.echo(f"Prompt number: #{new_prompt_number}")
    if hasattr(gepa_result, 'best_score'):
        click.echo(f"Best validation score: {gepa_result.best_score:.3f}")
