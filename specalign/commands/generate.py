"""Implementation of 'specalign generate' command for synthetic data generation."""

import datetime
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import yaml

from specalign.llm_client import LLMClient
from specalign.workspace import Workspace


SYNTHETIC_DATA_GENERATION_PROMPT = """You are a test case generation expert. Your task is to generate synthetic test cases based on LLM prompt specifications.

For each specification, generate diverse test cases that:
1. Cover different scenarios mentioned in the specification
2. Include edge cases and boundary conditions
3. Test both positive (pass) and negative (fail) scenarios
4. Are realistic and representative of real-world usage

Each test case should be formatted as a promptfoo test case with:
- vars: Input variables (e.g., {"input": "user query here"})
- assert: Assertions to validate the output
- metadata: Link to the specification(s) it tests

Output format: JSON array of test cases, where each test case follows this structure:
{{
  "vars": {{
    "input": "example user input"
  }},
  "assert": [
    {{
      "type": "contains",
      "value": "expected text"
    }},
    {{
      "type": "javascript",
      "value": "output.length > 10"
    }}
  ],
  "metadata": {{
    "spec_requirements": ["spec-name-1", "spec-name-2"],
    "scenario": "WHEN condition THEN expected behavior",
    "requirement": "Requirement name from spec"
  }}
}}

Generate diverse test cases covering all the specifications provided."""


def parse_test_cases_from_llm_response(response: str) -> List[Dict[str, Any]]:
    """Parse test cases from LLM response.

    Args:
        response: Raw LLM response text.

    Returns:
        List of test case dictionaries.
    """
    # Clean up the response
    text = response.strip()
    
    # Remove markdown code blocks if present
    if "```json" in text:
        # Extract JSON from code block
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    elif "```" in text:
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    
    try:
        # Try to parse as JSON
        test_cases = json.loads(text)
        
        # Ensure it's a list
        if isinstance(test_cases, dict):
            # Sometimes LLM wraps in an object
            if "test_cases" in test_cases:
                test_cases = test_cases["test_cases"]
            elif "tests" in test_cases:
                test_cases = test_cases["tests"]
            else:
                test_cases = [test_cases]
        
        if not isinstance(test_cases, list):
            test_cases = [test_cases]
        
        return test_cases
    
    except json.JSONDecodeError as e:
        click.echo(f"Warning: Could not parse LLM response as JSON: {e}", err=True)
        click.echo("Attempting to extract JSON from response...")
        
        # Try to find JSON array in the text
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # If all else fails, return empty list
        click.echo("Could not extract valid test cases from response.", err=True)
        return []


def generate_test_cases(
    specs: Dict[str, str],
    llm_client: LLMClient,
    count: int,
    per_spec: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Generate synthetic test cases using LLM.

    Args:
        specs: Dictionary mapping spec names to spec content.
        llm_client: LLM client for generation.
        count: Total number of test cases to generate.
        per_spec: Number of test cases per spec (if None, distributes evenly).

    Returns:
        List of test case dictionaries.
    """
    all_test_cases = []
    
    if per_spec:
        # Generate per_spec test cases for each specification
        cases_per_spec = per_spec
    else:
        # Distribute evenly across specs
        num_specs = len(specs)
        cases_per_spec = max(1, count // num_specs) if num_specs > 0 else count
    
    for spec_name, spec_content in specs.items():
        click.echo(f"Generating test cases for spec: {spec_name}...")
        
        user_prompt = f"""Generate {cases_per_spec} diverse test cases based on this specification:

=== {spec_name} ===
{spec_content}

Focus on:
- Different scenarios mentioned in the specification
- Edge cases and boundary conditions
- Both positive (should pass) and negative (should fail) test cases
- Realistic, representative examples

Each test case must include metadata linking it to the "{spec_name}" specification."""

        try:
            response = llm_client.generate(
                prompt=user_prompt,
                system_prompt=SYNTHETIC_DATA_GENERATION_PROMPT
            )
            
            test_cases = parse_test_cases_from_llm_response(response)
            
            # Ensure metadata links to this spec
            for test_case in test_cases:
                if "metadata" not in test_case:
                    test_case["metadata"] = {}
                
                # Ensure spec_requirements includes this spec
                if "spec_requirements" not in test_case["metadata"]:
                    test_case["metadata"]["spec_requirements"] = []
                
                if spec_name not in test_case["metadata"]["spec_requirements"]:
                    test_case["metadata"]["spec_requirements"].append(spec_name)
            
            all_test_cases.extend(test_cases)
            click.echo(f"  Generated {len(test_cases)} test cases")
        
        except Exception as e:
            click.echo(f"  Error generating test cases for {spec_name}: {e}", err=True)
            continue
    
    # Limit to requested count
    if len(all_test_cases) > count:
        all_test_cases = all_test_cases[:count]
    
    return all_test_cases


def format_as_promptfoo(test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format test cases as promptfoo configuration.

    Args:
        test_cases: List of test case dictionaries.

    Returns:
        Promptfoo configuration dictionary.
    """
    return {
        "tests": test_cases
    }


def run_generate(
    workspace: Workspace,
    model_config_path: Path,
    output_path: Optional[Path] = None,
    count: int = 10,
    per_spec: Optional[int] = None,
) -> None:
    """Run the generate command.

    Args:
        workspace: Workspace instance.
        model_config_path: Path to model configuration YAML file.
        output_path: Optional path to save test cases file. If None, uses default location.
        count: Total number of test cases to generate.
        per_spec: Number of test cases per specification (overrides count distribution).
    """
    if not workspace.exists():
        click.echo("Error: Workspace not initialized. Run 'specalign init' first.", err=True)
        return
    
    # Ensure test_cases directory exists
    workspace.test_cases_dir.mkdir(parents=True, exist_ok=True)
    
    # Load specifications
    spec_files = workspace.get_all_spec_files()
    
    if not spec_files:
        click.echo(f"Error: No specification files found in {workspace.specs_dir}", err=True)
        return
    
    click.echo(f"Found {len(spec_files)} specification file(s):")
    specs = {}
    for spec_file in spec_files:
        with open(spec_file) as f:
            spec_name = spec_file.stem
            specs[spec_name] = f.read()
            click.echo(f"  - {spec_name}")
    
    # Create LLM client
    click.echo(f"\nUsing model config: {model_config_path}")
    llm_client = LLMClient(model_config_path=model_config_path)
    
    # Generate test cases
    click.echo(f"\nGenerating {count} test cases...")
    test_cases = generate_test_cases(specs, llm_client, count, per_spec)
    
    if not test_cases:
        click.echo("Error: No test cases were generated.", err=True)
        return
    
    click.echo(f"Successfully generated {len(test_cases)} test cases")
    
    # Format as promptfoo
    promptfoo_config = format_as_promptfoo(test_cases)
    
    # Determine output path
    if output_path is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = workspace.test_cases_dir / f"test_cases_{timestamp}.yaml"
    else:
        # Ensure it's in test_cases directory if relative
        if not output_path.is_absolute():
            output_path = workspace.test_cases_dir / output_path
    
    # Save test cases
    with open(output_path, "w") as f:
        yaml.dump(promptfoo_config, f, default_flow_style=False, sort_keys=False)
    
    # Save metadata
    metadata = {
        "generation": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model": str(model_config_path),
            "total_test_cases": len(test_cases),
            "specs_covered": list(specs.keys()),
        },
        "test_case_summary": {
            spec_name: sum(
                1 for tc in test_cases
                if spec_name in tc.get("metadata", {}).get("spec_requirements", [])
            )
            for spec_name in specs.keys()
        }
    }
    
    metadata_path = output_path.with_suffix(".metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    click.echo(f"\nTest cases generated successfully!")
    click.echo(f"  Test cases: {output_path}")
    click.echo(f"  Metadata: {metadata_path}")
    click.echo(f"\n  Test cases per spec:")
    for spec_name, count in metadata["test_case_summary"].items():
        click.echo(f"    - {spec_name}: {count}")
    
    click.echo(f"\nTo run with promptfoo:")
    click.echo(f"  promptfoo eval -c {output_path}")