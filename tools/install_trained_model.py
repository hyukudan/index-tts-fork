#!/usr/bin/env python3
"""
Install a trained model for use in WebUI.

This script safely copies a trained checkpoint and tokenizer to the models/
directory with descriptive names, ensuring the base models in checkpoints/
are never overwritten.

Example:
    python tools/install_trained_model.py \\
        --checkpoint training/catalan/checkpoints/model_step5000.pth \\
        --tokenizer training/catalan/tokenizer/catalan_bpe.model \\
        --output-name catalan \\
        --description "Catalan fine-tuned model"
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install a trained model to the models/ directory for WebUI use."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained GPT checkpoint (.pth file)",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="Path to the BPE tokenizer (.model file)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        required=True,
        help="Name identifier for the model (e.g., 'catalan', 'french', 'multilingual_romance')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Output directory for installed models (default: models/)",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Optional description of the model",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional custom config.yaml to copy (default: use checkpoints/config.yaml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing model files without prompting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually copying files",
    )
    return parser.parse_args()


def validate_paths(checkpoint: Path, tokenizer: Path, config: Optional[Path]) -> None:
    """Validate that input paths exist and have correct extensions."""
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if not checkpoint.suffix == ".pth":
        raise ValueError(f"Checkpoint must be a .pth file, got: {checkpoint.suffix}")

    if not tokenizer.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer}")

    if not tokenizer.suffix == ".model":
        raise ValueError(f"Tokenizer must be a .model file, got: {tokenizer.suffix}")

    if config and not config.exists():
        raise FileNotFoundError(f"Config not found: {config}")


def check_protected_directory(output_dir: Path) -> None:
    """Warn if trying to install to protected directories."""
    protected_dirs = ["checkpoints", "checkpoint"]
    if output_dir.name.lower() in protected_dirs:
        print(f"\n⚠️  WARNING: Installing to '{output_dir}' may overwrite base models!", file=sys.stderr)
        print(f"⚠️  Recommended: Use a separate directory like 'models/' instead.", file=sys.stderr)
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.", file=sys.stderr)
            sys.exit(1)


def prompt_overwrite(file_path: Path) -> bool:
    """Ask user if they want to overwrite an existing file."""
    response = input(f"File {file_path} already exists. Overwrite? [y/N]: ")
    return response.lower() == 'y'


def install_model(
    checkpoint: Path,
    tokenizer: Path,
    output_name: str,
    output_dir: Path,
    config: Optional[Path] = None,
    description: str = "",
    force: bool = False,
    dry_run: bool = False,
) -> None:
    """Install trained model to output directory."""

    # Validate inputs
    validate_paths(checkpoint, tokenizer, config)
    check_protected_directory(output_dir)

    # Create output directory
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output filenames
    gpt_output = output_dir / f"gpt_{output_name}.pth"
    tokenizer_output = output_dir / f"{output_name}_bpe.model"
    config_output = output_dir / f"{output_name}_config.yaml" if config else None

    # Check for existing files
    files_to_copy = [
        (checkpoint, gpt_output, "GPT checkpoint"),
        (tokenizer, tokenizer_output, "BPE tokenizer"),
    ]

    if config:
        files_to_copy.append((config, config_output, "Config"))
    else:
        # Use default config from checkpoints/
        default_config = Path("checkpoints/config.yaml")
        if default_config.exists():
            config_output = output_dir / f"{output_name}_config.yaml"
            files_to_copy.append((default_config, config_output, "Config (default)"))

    # Display plan
    print("\n" + "="*70)
    print("INSTALLATION PLAN")
    print("="*70)
    print(f"Model Name: {output_name}")
    if description:
        print(f"Description: {description}")
    print(f"Output Directory: {output_dir}")
    print()

    all_good = True
    for src, dest, label in files_to_copy:
        status = ""
        if dest.exists():
            if force:
                status = " [WILL OVERWRITE]"
            else:
                status = " [EXISTS]"
                all_good = False

        print(f"  {label}:")
        print(f"    Source: {src}")
        print(f"    Dest:   {dest}{status}")

    print("="*70)

    if dry_run:
        print("\n[DRY RUN] No files were copied.")
        return

    # Handle existing files
    if not all_good and not force:
        print("\nSome files already exist.")
        for src, dest, label in files_to_copy:
            if dest.exists():
                if not prompt_overwrite(dest):
                    print(f"Skipping {label}")
                    continue

                # User confirmed overwrite, proceed with this file
                print(f"Copying {label}...")
                shutil.copy2(src, dest)
                print(f"  ✓ {dest}")
        return

    # Copy files
    print("\nCopying files...")
    for src, dest, label in files_to_copy:
        shutil.copy2(src, dest)
        print(f"  ✓ {label}: {dest}")

    # Create metadata file
    metadata_file = output_dir / f"{output_name}_metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write(f"Model: {output_name}\n")
        if description:
            f.write(f"Description: {description}\n")
        f.write(f"Source Checkpoint: {checkpoint}\n")
        f.write(f"Source Tokenizer: {tokenizer}\n")
        if config:
            f.write(f"Source Config: {config}\n")

    print(f"  ✓ Metadata: {metadata_file}")

    # Success message
    print("\n" + "="*70)
    print("✅ MODEL INSTALLED SUCCESSFULLY")
    print("="*70)
    print(f"\nYour model is now available as:")
    print(f"  - GPT: {gpt_output.name}")
    print(f"  - Tokenizer: {tokenizer_output.name}")
    if config_output:
        print(f"  - Config: {config_output.name}")

    print(f"\nTo use in WebUI:")
    print(f"  python webui.py --model-dir {output_dir}")

    print(f"\nThe model will appear in the dropdown as:")
    print(f'  "{gpt_output.name} (..., {output_name})"')

    print(f"\nTokenizer will be auto-detected as:")
    print(f"  {tokenizer_output.name}")

    print("\n" + "="*70)


def main() -> int:
    args = parse_args()

    try:
        install_model(
            checkpoint=args.checkpoint,
            tokenizer=args.tokenizer,
            output_name=args.output_name,
            output_dir=args.output_dir,
            config=args.config,
            description=args.description,
            force=args.force,
            dry_run=args.dry_run,
        )
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
