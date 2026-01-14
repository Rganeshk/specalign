"""Command line interface for specalign."""

from pathlib import Path

import click

from specalign.commands.compile import run_compile
from specalign.commands.evaluate import run_evaluate
from specalign.commands.generate import run_generate
from specalign.commands.init import run_init
from specalign.commands.optimize import run_optimize
from specalign.workspace import Workspace

@click.group()
@click.version_option(version="0.1.0")
def cli():
    """specalign - Generate, iterate, and maintain LLM prompts."""
    pass


@cli.command()
@click.option(
    "--path",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Workspace path (defaults to current directory)"
)
@click.option(
    "--extraction-model",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to extraction model configuration YAML file (optional, uses default if not provided)"
)
def init(path: Path, extraction_model: Path):
    """Initialize a new specalign workspace."""
    workspace = Workspace(path)
    run_init(workspace, extraction_model)


@cli.command()
@click.option(
    "--path",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Workspace path (defaults to current directory)"
)
@click.option(
    "--model",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to model configuration YAML file"
)
def compile(path: Path, model: Path):
    """Generate a new prompt based on current specifications."""
    workspace = Workspace(path)
    run_compile(workspace, model)


@cli.command()
@click.option(
    "--path",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Workspace path (defaults to current directory)"
)
@click.option(
    "--model",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to model configuration YAML file"
)
@click.option(
    "--data",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to data configuration JSON file"
)
@click.option(
    "--max-samples",
    type=int,
    default=None,
    help="Maximum number of samples to evaluate"
)
@click.option(
    "--prompt",
    type=int,
    default=None,
    help="Specific prompt number to evaluate (defaults to latest)"
)
@click.option(
    "--workers",
    type=int,
    default=32,
    help="Maximum number of parallel workers for sample processing (default: 32)"
)
@click.option(
    "--eval-model",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to evaluation model configuration YAML file"
)
def evaluate(path: Path, model: Path, data: Path, max_samples: int, prompt: int, workers: int, eval_model: Path):
    """Run evaluation on a prompt using specified model and data."""
    workspace = Workspace(path)
    run_evaluate(workspace, model, data, max_samples, prompt, workers, eval_model)


@cli.command()
@click.option(
    "--path",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Workspace path (defaults to current directory)"
)
@click.option(
    "--model",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to model configuration YAML file"
)
@click.option(
    "--output",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Output path for test cases file (default: .specalign/test_cases/test_cases_TIMESTAMP.yaml)"
)
@click.option(
    "--count",
    type=int,
    default=10,
    help="Total number of test cases to generate (default: 10)"
)
@click.option(
    "--per-spec",
    type=int,
    default=None,
    help="Number of test cases per specification (overrides count distribution)"
)
def generate(path: Path, model: Path, output: Path, count: int, per_spec: int):
    """Generate synthetic test cases based on specifications."""
    workspace = Workspace(path)
    run_generate(workspace, model, output, count, per_spec)


@cli.command()
@click.option(
    "--path",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Workspace path (defaults to current directory)"
)
@click.option(
    "--model",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to model configuration YAML file"
)
@click.option(
    "--data",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to data configuration JSON file"
)
@click.option(
    "--train-samples",
    required=True,
    type=int,
    help="Number of training samples to use for optimization"
)
@click.option(
    "--val-samples",
    type=int,
    default=None,
    help="Number of validation samples (defaults to same as train-samples)"
)
@click.option(
    "--prompt",
    type=int,
    default=None,
    help="Specific prompt number to use as seed (defaults to latest)"
)
@click.option(
    "--max-metric-calls",
    type=int,
    default=150,
    help="Maximum number of metric calls during optimization (default: 150)"
)
@click.option(
    "--reflection-lm",
    default="openai/gpt-5.2",
    help="LLM to use for reflection/optimization (default: openai/gpt-5.2)"
)
@click.option(
    "--eval-model",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to evaluation model configuration YAML file"
)
@click.option(
    "--use-wandb",
    is_flag=True,
    help="Enable Weights & Biases logging"
)
def optimize(
    path: Path,
    model: Path,
    data: Path,
    train_samples: int,
    val_samples: int,
    prompt: int,
    max_metric_calls: int,
    reflection_lm: str,
    eval_model: Path,
    use_wandb: bool
):
    """Optimize a prompt using GEPA (Generalized Evolutionary Prompt Adaptation)."""
    workspace = Workspace(path)
    run_optimize(
        workspace,
        model,
        data,
        train_samples,
        val_samples,
        prompt,
        max_metric_calls,
        reflection_lm,
        eval_model,
        use_wandb
    )


if __name__ == "__main__":
    cli()
