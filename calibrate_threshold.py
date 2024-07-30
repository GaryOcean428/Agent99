"""
Calibration script to find the optimal threshold for RouteLLM in Chat99.
"""

import argparse
import json
from typing import List, Dict

from routellm.controller import Controller
from routellm.calibrate_threshold import calibrate_threshold

from config import HIGH_TIER_MODEL, MID_TIER_MODEL

def load_sample_queries(file_path: str) -> List[Dict[str, str]]:
    """Load sample queries from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Calibrate RouteLLM threshold for Chat99")
    parser.add_argument("--sample-queries", type=str, required=True,
                        help="Path to JSON file containing sample queries")
    parser.add_argument("--router", type=str, default="mf",
                        help="RouteLLM router to use (default: mf)")
    parser.add_argument("--strong-model-pct", type=float, default=0.5,
                        help="Target percentage of queries to route to the strong model (default: 0.5)")
    parser.add_argument("--output", type=str, default="calibration_results.json",
                        help="Output file to save calibration results (default: calibration_results.json)")
    args = parser.parse_args()

    # Load sample queries
    sample_queries = load_sample_queries(args.sample_queries)

    # Set up RouteLLM controller
    controller = Controller(
        routers=[args.router],
        strong_model=HIGH_TIER_MODEL,
        weak_model=MID_TIER_MODEL,
    )

    # Calibrate threshold
    threshold = calibrate_threshold(
        controller=controller,
        router=args.router,
        prompts=[query['content'] for query in sample_queries],
        target_strong_pct=args.strong_model_pct
    )

    print(f"Optimal threshold for {args.strong_model_pct*100}% strong model calls: {threshold}")

    # Save results
    results = {
        "router": args.router,
        "strong_model_percentage": args.strong_model_pct,
        "optimal_threshold": threshold
    }
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Calibration results saved to {args.output}")

if __name__ == "__main__":
    main()