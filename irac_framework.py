"""
IRAC Framework: Structures responses according to the Issue, Rule, Application, Conclusion method.
"""

import re


def apply_irac_framework(query: str, response: str) -> str:
    """
    Apply the IRAC framework to structure the response.
    """
    # Extract the issue from the query
    issue = f"Issue: {query}\n\n"

    # Split the response into paragraphs
    paragraphs = response.split("\n\n")

    # Attempt to identify the rule, application, and conclusion
    rule = "Rule: "
    application = "Application: "
    conclusion = "Conclusion: "

    for i, paragraph in enumerate(paragraphs):
        if i == 0:
            rule += paragraph + "\n\n"
        elif i == len(paragraphs) - 1:
            conclusion += paragraph + "\n\n"
        else:
            application += paragraph + "\n\n"

    # If we couldn't clearly identify parts, fall back to a simpler structure
    if len(paragraphs) < 3:
        rule = f"Rule: {paragraphs[0] if len(paragraphs) > 0 else ''}\n\n"
        application = f"Application: {paragraphs[1] if len(paragraphs) > 1 else ''}\n\n"
        conclusion = f"Conclusion: {paragraphs[-1]}\n\n"

    return issue + rule + application + conclusion


def apply_comparative_analysis(query: str, response: str) -> str:
    """
    Structure the response as a comparative analysis.
    """
    introduction = f"Analysis of: {query}\n\n"

    # Split the response into paragraphs
    paragraphs = response.split("\n\n")

    # Attempt to identify key points of comparison
    comparison_points = []
    for paragraph in paragraphs:
        if re.match(
            r"^(Comparing|On one hand|On the other hand|In contrast|Similarly)",
            paragraph,
        ):
            comparison_points.append(paragraph)

    # If we couldn't identify clear comparison points, use all paragraphs
    if not comparison_points:
        comparison_points = paragraphs

    body = "\n\n".join(comparison_points)

    conclusion = f"\n\nConclusion: {paragraphs[-1] if len(paragraphs) > 1 else ''}"

    return introduction + body + conclusion


# Add more helper functions for other response strategies as needed
