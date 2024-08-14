"""
Calibration script to find the optimal threshold for the AdvancedRouter in Chat99.
"""

import argparse
import json
from typing import List, Dict
from advanced_router import AdvancedRouter


def load_sample_queries(file_path: str) -> List[Dict[str, str]]:
    """Load sample queries from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def calibrate_threshold(
    router: AdvancedRouter, queries: List[Dict[str, str]], target_strong_pct: float
) -> float:
    """
    Calibrate the complexity threshold for the AdvancedRouter.
    """
    complexities = [router.determine_complexity(
        query["content"]) for query in queries]
    strong_model_calls = sum(
        1 for complexity in complexities if complexity == "high")
    actual_strong_pct = strong_model_calls / len(queries)

    if actual_strong_pct < target_strong_pct:
        return 0.7  # Lower threshold to increase strong model usage
    elif actual_strong_pct > target_strong_pct:
        return 0.3  # Raise threshold to decrease strong model usage
    else:
        return 0.5  # Keep current threshold


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate AdvancedRouter threshold for Chat99"
    )
    parser.add_argument(
        "--sample-queries",
        type=str,
        required=True,
        help="Path to JSON file containing sample queries",
    )
    parser.add_argument(
        "--strong-model-pct",
        type=float,
        default=0.5,
        help="Target percentage of queries to route to the strong model (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="calibration_results.json",
        help="Output file to save calibration results (default: calibration_results.json)",
    )
    args = parser.parse_args()

    # Load sample queries
    sample_queries = load_sample_queries(args.sample_queries)

    # Set up AdvancedRouter
    router = AdvancedRouter()

    # Calibrate threshold
    threshold = calibrate_threshold(
        router, sample_queries, args.strong_model_pct)

    print(
        f"Optimal threshold for {args.strong_model_pct*100}% strong model calls: {threshold}"
    )

    # Save results
    results = {
        "strong_model_percentage": args.strong_model_pct,
        "optimal_threshold": threshold,
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Calibration results saved to {args.output}")


if __name__ == "__main__":
    main()
