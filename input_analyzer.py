"""
This module provides functionality to analyze user input and classify it as simple or complex.
"""

import re


class InputAnalyzer:
    """A class to analyze and classify user input."""

    def __init__(self, config):
        """
        Initialize the InputAnalyzer.

        Args:
            config: Configuration object containing settings.
        """
        self.config = config

    def analyze(self, user_input):
        """
        Analyze the user input and classify it as simple or complex.

        Args:
            user_input (str): The user's input to be analyzed.

        Returns:
            str: 'simple' or 'complex' based on the analysis.
        """
        # Check for complex questions or instructions
        complex_patterns = [
            r"\bwhy\b", r"\bhow\b", r"\bexplain\b", r"\bcompare\b",
            r"\banalyze\b", r"\bsteps\b", r"\bprocess\b",
            r"\brelationship between\b", r"\bimpact of\b",
            r"\bconsequences\b", r"\bpros and cons\b",
        ]

        # Check for simple queries or statements
        simple_patterns = [
            r"\bwhat is\b", r"\bwho is\b", r"\bwhen is\b", r"\bwhere is\b",
            r"\byes\b", r"\bno\b", r"\bokay\b", r"\bthanks\b", r"\bhi\b",
            r"\bhello\b",
        ]

        # Check if the input matches any complex patterns
        if any(re.search(pattern, user_input.lower())
               for pattern in complex_patterns):
            return "complex"

        # Check if the input matches any simple patterns
        if any(re.search(pattern, user_input.lower())
               for pattern in simple_patterns):
            return "simple"

        # If no clear pattern is found, default to complex for a thorough response
        return "complex"
