"""
Benford analysis utilities for physics-paper corpora.

This module extracts numerical values from scientific text, identifies first
significant digits, and compares the observed distribution against Benford's Law.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class BenfordResult:
    sample_size: int
    counts: dict[int, int]
    observed: dict[int, float]
    expected: dict[int, float]
    chi_square: float
    mean_absolute_deviation: float


def benford_probabilities() -> dict[int, float]:
    """Return Benford's expected probability for digits 1 through 9."""
    return {digit: math.log10(1 + 1 / digit) for digit in range(1, 10)}


def extract_numbers(text: str) -> list[str]:
    """Extract numeric strings, including decimals and scientific notation."""
    return NUMBER_PATTERN.findall(text)


def first_significant_digit(value: str | int | float) -> int | None:
    """Return the first significant digit of a numeric value."""
    try:
        number = abs(float(value))
    except (TypeError, ValueError):
        return None

    if number == 0 or math.isnan(number) or math.isinf(number):
        return None

    while number < 1:
        number *= 10

    while number >= 10:
        number /= 10

    digit = int(number)
    return digit if 1 <= digit <= 9 else None


def observed_distribution(numbers: list[str]) -> tuple[dict[int, int], dict[int, float]]:
    """Convert extracted numbers into first-digit counts and proportions."""
    digits = []

    for number in numbers:
        digit = first_significant_digit(number)
        if digit is not None:
            digits.append(digit)

    counts_raw = Counter(digits)
    counts = {digit: counts_raw.get(digit, 0) for digit in range(1, 10)}
    total = sum(counts.values())

    if total == 0:
        proportions = {digit: 0.0 for digit in range(1, 10)}
    else:
        proportions = {digit: counts[digit] / total for digit in range(1, 10)}

    return counts, proportions


def chi_square_statistic(
    observed: dict[int, float],
    expected: dict[int, float],
    sample_size: int,
) -> float:
    """Compute chi-square statistic from observed and expected proportions."""
    statistic = 0.0

    for digit in range(1, 10):
        observed_count = observed[digit] * sample_size
        expected_count = expected[digit] * sample_size

        if expected_count > 0:
            statistic += ((observed_count - expected_count) ** 2) / expected_count

    return statistic


def mean_absolute_deviation(
    observed: dict[int, float],
    expected: dict[int, float],
) -> float:
    """Compute mean absolute deviation across first-digit proportions."""
    return sum(abs(observed[digit] - expected[digit]) for digit in range(1, 10)) / 9


def analyze_text(text: str) -> BenfordResult:
    """Run a Benford first-digit analysis on scientific text."""
    numbers = extract_numbers(text)
    counts, observed = observed_distribution(numbers)
    expected = benford_probabilities()
    sample_size = sum(counts.values())

    return BenfordResult(
        sample_size=sample_size,
        counts=counts,
        observed=observed,
        expected=expected,
        chi_square=chi_square_statistic(observed, expected, sample_size),
        mean_absolute_deviation=mean_absolute_deviation(observed, expected),
    )


def format_report(result: BenfordResult) -> str:
    """Format a human-readable Benford analysis report."""
    lines = [
        "Digit | Count | Observed | Expected | Difference",
        "------|-------|----------|----------|-----------",
    ]

    for digit in range(1, 10):
        observed = result.observed[digit]
        expected = result.expected[digit]
        difference = observed - expected

        lines.append(
            f"{digit:>5} | "
            f"{result.counts[digit]:>5} | "
            f"{observed:>8.4f} | "
            f"{expected:>8.4f} | "
            f"{difference:>9.4f}"
        )

    lines.append("")
    lines.append(f"Sample size: {result.sample_size}")
    lines.append(f"Chi-square statistic: {result.chi_square:.4f}")
    lines.append(f"Mean absolute deviation: {result.mean_absolute_deviation:.4f}")

    return "\n".join(lines)


if __name__ == "__main__":
    demo_text = """
    The detector recorded 123 events, 47 calibration measurements,
    0.0031 seconds of timing uncertainty, 9.81 m/s^2 acceleration,
    6.022e23 particles per mole, and 1.602e-19 coulombs.
    """

    result = analyze_text(demo_text)
    print(format_report(result))
