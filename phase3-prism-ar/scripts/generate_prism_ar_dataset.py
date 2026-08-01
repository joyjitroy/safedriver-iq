"""Generate the PRISM-AR dataset.

Usage:
    python generate_prism_ar_dataset.py --output_dir data/prism_ar --n_per_template 5
"""
import argparse
import os

from prism_ar.dataset_generation.dataset_builder import PRISMARDatasetBuilder


def main():
    parser = argparse.ArgumentParser(description="Generate PRISM-AR dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/prism_ar",
        help="Output directory for images and annotations",
    )
    parser.add_argument(
        "--n_per_template",
        type=int,
        default=5,
        help="Number of scenarios per risk template",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), args.output_dir
    )
    builder = PRISMARDatasetBuilder(
        output_dir=output_dir,
        n_per_template=args.n_per_template,
        seed=args.seed,
    )
    csv_path = builder.build()
    builder.close()
    print(f"Dataset generated: {csv_path}")
    print(f"Images: {os.path.join(output_dir, 'images')}")


if __name__ == "__main__":
    main()
