"""
main.py: Main entry point for the Chat99 application.
"""

import argparse
import os
import logging
from utils import setup_logging
from config import DEFAULT_ROUTER, DEFAULT_THRESHOLD, get_profile_name, get_user_name
from chat99 import chat_with_99
from calibrate_threshold import main as calibrate


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the Chat99 application.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Chat99 - An intelligent AI assistant")
    parser.add_argument(
        "--calibrate", action="store_true", help="Run threshold calibration"
    )
    parser.add_argument(
        "--use-dynamic-routing", action="store_true", help="Use dynamic routing"
    )
    parser.add_argument(
        "--router",
        type=str,
        default=DEFAULT_ROUTER,
        help=f"Router to use (default: {DEFAULT_ROUTER})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Routing threshold (default: {DEFAULT_THRESHOLD})",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main function to run the Chat99 application.
    """
    profile_name = get_profile_name()
    user_name = get_user_name()
    script_filename = os.path.basename(__file__)

    # Setup logging
    logger = setup_logging(script_filename, profile_name, user_name)
    logger.info("Starting the Chat99 application...")

    args = parse_args()

    if args.calibrate:
        calibrate()
    else:
        chat_with_99(args)


if __name__ == "__main__":
    main()
