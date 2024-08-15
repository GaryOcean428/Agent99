import re


def analyze_input(user_input):
    # Check for complex questions or instructions
    complex_patterns = [
        r"\bwhy\b",
        r"\bhow\b",
        r"\bexplain\b",
        r"\bcompare\b",
        r"\banalyze\b",
        r"\bsteps\b",
        r"\bprocess\b",
        r"\brelationship between\b",
        r"\bimpact of\b",
        r"\bconsequences\b",
        r"\bpros and cons\b",
    ]

    # Check for simple queries or statements
    simple_patterns = [
        r"\bwhat is\b",
        r"\bwho is\b",
        r"\bwhen is\b",
        r"\bwhere is\b",
        r"\byes\b",
        r"\bno\b",
        r"\bokay\b",
        r"\bthanks\b",
        r"\bhi\b",
        r"\bhello\b",
    ]

    # Check if the input matches any complex patterns
    if any(re.search(pattern, user_input.lower()) for pattern in complex_patterns):
        return "complex"

    # Check if the input matches any simple patterns
    elif any(re.search(pattern, user_input.lower()) for pattern in simple_patterns):
        return "simple"

    # If no clear pattern is found, default to complex for a more thorough response
    else:
        return "complex"
