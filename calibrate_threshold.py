"""
Calibration script to find the optimal threshold for RouteLLM in Chat99.
"""

import sys
import os
print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")
print(f"Sys path: {sys.path}")
print(f"Current working directory: {os.getcwd()}")

import argparse
import json
from typing import List, Dict

try:
    import routellm
    print(f"RouteLLM is installed at: {routellm.__file__}")
    print(f"RouteLLM contents: {dir(routellm)}")
    
    # Print the contents of the routellm package directory
    routellm_dir = os.path.dirname(routellm.__file__)
    print(f"Contents of {routellm_dir}:")
    for item in os.listdir(routellm_dir):
        print(f"  {item}")
    
    # Import Controller and calibrate_threshold
    from routellm.routers.routers import ROUTER_CLS
    Controller = ROUTER_CLS['base']
    print("Imported Controller from routellm.routers.routers")

    from routellm.calibrate_threshold import calibrate_threshold
    print("Imported calibrate_threshold from routellm.calibrate_threshold")

except ImportError as e:
    print(f"Error importing RouteLLM modules: {e}")
    print("Please ensure RouteLLM is installed correctly")
    sys.exit(1)

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure OpenAI API key is set
if 'OPENAI_API_KEY' not in os.environ:
    print("Error: OPENAI_API_KEY environment variable is not set.")
    print("Please set it in your .env file or export it in your shell.")
    sys.exit(1)

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
        strong_model="claude-3-5-sonnet-20240620",
        weak_model="anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1",
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
