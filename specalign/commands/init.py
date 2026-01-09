"""Implementation of 'specalign init' command."""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import click
import yaml

from specalign.llm_client import LLMClient
from specalign.workspace import Workspace


SPEC_EXTRACTION_PROMPT = """You are a specification extraction expert. Your task is to analyze a given LLM prompt and extract clear, testable specifications from it.

The specifications should follow this structure:

# [spec-name] Specification

## Purpose
[Brief description of what this specification governs]

## Requirements

### Requirement: [Requirement Name]

[Description of the requirement using "SHALL" language]

#### Scenario: [Scenario name]

- **WHEN** [condition]
- **THEN** [expected behavior]
  - [specific requirement 1]
  - [specific requirement 2]

## Evaluation Guidelines

### Pass Criteria
A response **passes** if it:
- [criterion 1]
- [criterion 2]

### Fail Criteria
A response **fails** if it:
- [criterion 1]
- [criterion 2]

## Examples

### Good Example
[Example of compliant behavior]

### Bad Example
[Example of non-compliant behavior]

---

Please analyze the following prompt and extract ALL distinct specifications (e.g., response-style, factual-accuracy, code-quality, etc.). Create a separate specification document for each major category.

Output format: Provide your response in XML format with the following structure:

<specifications>
  <spec filename="response-style.md">
    <![CDATA[
# response-style Specification

[Full markdown content here...]
    ]]>
  </spec>
  <spec filename="factual-accuracy.md">
    <![CDATA[
# factual-accuracy Specification

[Full markdown content here...]
    ]]>
  </spec>
</specifications>

IMPORTANT:
- Use CDATA sections to wrap the markdown content
- Each spec should have a unique filename attribute
- Output ONLY the XML, no additional text before or after

PROMPT TO ANALYZE:
{prompt}
"""


def extract_specs_from_prompt(prompt_text: str, llm_client: LLMClient) -> dict[str, str]:
    """Extract specifications from a prompt using an LLM.

    Args:
        prompt_text: The prompt to analyze.
        llm_client: LLM client for extraction.

    Returns:
        Dictionary mapping spec filenames to their markdown content.
    """
    extraction_prompt = SPEC_EXTRACTION_PROMPT.format(prompt=prompt_text)
    response = llm_client.generate(extraction_prompt)

    # Try to parse XML from the response
    response = response.strip()

    # Remove markdown code blocks if present
    if response.startswith("```xml"):
        response = response[6:]
    elif response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()

    try:
        # Parse XML
        root = ET.fromstring(response)

        specs = {}
        for spec_elem in root.findall('spec'):
            filename = spec_elem.get('filename')
            content = spec_elem.text

            if filename and content:
                # Clean up the content
                content = content.strip()
                specs[filename] = content

        if not specs:
            click.echo("Warning: No specifications found in XML response. Please review manually.")
            return {"extracted-spec.md": response}

        return specs

    except ET.ParseError as e:
        click.echo(f"Warning: Could not parse LLM response as XML: {e}")
        click.echo("Attempting to extract specs manually...")

        # Fallback: try to extract specs manually using regex
        specs = {}

        # Look for spec tags
        spec_pattern = r'<spec\s+filename="([^"]+)">\s*<!\[CDATA\[(.*?)\]\]>\s*</spec>'
        matches = re.findall(spec_pattern, response, re.DOTALL)

        if matches:
            for filename, content in matches:
                specs[filename] = content.strip()
            return specs

        # If all else fails, return the raw response
        return {"extracted-spec.md": response}


def run_init(workspace: Workspace, extraction_model_config_path: Optional[Path] = None) -> None:
    """Run the init command.

    Args:
        workspace: Workspace instance.
        extraction_model_config_path: Path to extraction model configuration (optional).
    """
    click.echo("Initializing specalign workspace...")

    # Check if already initialized
    if workspace.exists():
        if not click.confirm("Workspace already exists. Reinitialize?", default=False):
            click.echo("Initialization cancelled.")
            return

    # Create directory structure
    workspace.initialize()
    click.echo(f"Created workspace structure at {workspace.workspace_root}")

    # Ask for model configuration
    click.echo("\n--- Model Configuration ---")
    model_name = click.prompt(
        "Enter model name for evaluation (e.g., openai/gpt-4, vertex_ai/gemini-2.5-flash",
        default="vertex_ai/gemini-2.5-flash"
    )

    model_config = {
        "model": {
            "name": model_name,
            "temperature": click.prompt("Temperature", default=0.7, type=float),
            "max_tokens": click.prompt("Max tokens", default=16000, type=int),
        }
    }

    # Add API key field based on provider
    if "openai" in model_name.lower():
        model_config["model"]["api_key"] = "${OPENAI_API_KEY}"
    elif "vertex" in model_name.lower():
        model_config["model"]["vertex_credentials"] = "${VERTEX_CREDENTIALS}"

    config_filename = click.prompt("Model config filename", default="default")
    config_path = workspace.get_model_config(config_filename)

    with open(config_path, "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)

    click.echo(f"Saved model config to {config_path}")

    # Ask for data configuration
    click.echo("\n--- Data Configuration ---")
    if click.confirm("Configure a dataset?", default=True):
        dataset_name = click.prompt("Dataset name", default="training_data")
        dataset_path = click.prompt("Dataset path", default="/path/to/data.csv")
        dataset_format = click.prompt("Dataset format", default="csv")

        data_config = {
            "dataset_name": dataset_name,
            "description": click.prompt("Dataset description", default="Main dataset"),
            "path": dataset_path,
            "format": dataset_format,
        }

        data_config_path = workspace.get_data_config(dataset_name)
        with open(data_config_path, "w") as f:
            json.dump(data_config, f, indent=2)

        click.echo(f"Saved data config to {data_config_path}")

    # Ask for existing prompt
    click.echo("\n--- Specification Extraction ---")
    if click.confirm("Do you have an existing prompt to extract specifications from?", default=False):
        click.echo("Please paste your prompt content (press Ctrl+D or Ctrl+Z when finished):")

        # Read multiline input from user
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass

        prompt_text = "\n".join(lines).strip()

        if not prompt_text:
            click.echo("No prompt content provided. Skipping specification extraction.")
            click.echo(f"You can manually create spec files in the {workspace.specs_dir} directory")
        else:
            # Save the original prompt to prompts/0/
            prompt_dir = workspace.create_prompt_dir(number=0)
            prompt_file = prompt_dir / "prompt.md"
            with open(prompt_file, "w") as f:
                f.write(prompt_text)
            click.echo(f"Saved original prompt to {prompt_file}")

            click.echo("Extracting specifications using LLM (this may take a while)...")

            if extraction_model_config_path:
                click.echo(f"Using extraction model: {extraction_model_config_path}")
                extraction_client = LLMClient(model_config_path=extraction_model_config_path)
            else:
                default_extraction_model = "openai/gpt-4.1"
                click.echo(f"Using default extraction model: {default_extraction_model}")
                extraction_client = LLMClient.create_default_client(default_extraction_model)

            try:
                specs = extract_specs_from_prompt(prompt_text, extraction_client)

                # Save extracted specs
                for filename, content in specs.items():
                    spec_path = workspace.specs_dir / filename
                    with open(spec_path, "w") as f:
                        f.write(content)
                    click.echo(f"Created spec file: {spec_path}")

                click.echo("\n--- Review Extracted Specifications ---")
                click.echo(f"Please review the specifications in {workspace.specs_dir}")
                click.echo("You can edit them manually before running 'specalign compile'")

            except Exception as e:
                click.echo(f"Error during spec extraction: {e}", err=True)
                click.echo(f"You can manually create spec files in the {workspace.specs_dir} directory")

    else:
        click.echo(f"You can manually create spec files in the {workspace.specs_dir} directory")

    click.echo("\nInitialization complete!")
    click.echo("\nNext steps:")
    click.echo(f"  1. Review/edit specifications in {workspace.specs_dir}")
    click.echo("  2. Run 'specalign compile' to generate a prompt")
    click.echo("  3. Run 'specalign evaluate' to test your prompt")
