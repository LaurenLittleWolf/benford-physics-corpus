"""
Visible prospect-facing Benford pipeline.

This self-contained file shows numerical extraction, first-digit analysis,
metrics, reporting, demo data, and the implementation roadmap without requiring
any additional project files.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DIGITS = tuple(range(1, 10))

NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z])[-+]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][-+]?\d+)?(?![A-Za-z])"
)

REFERENCE_PATTERN = re.compile(r"(?:\[[0-9,\-\s]+\]|\([0-9,\-\s]+\))")

PIPELINE_ROADMAP = [
    {
        "step": 1,
        "component": "pdf_text_extraction",
        "status": "planned",
        "priority": 1,
    },
    {
        "step": 2,
        "component": "numeric_token_classification",
        "status": "planned",
        "priority": 1,
    },
    {
        "step": 3,
        "component": "unit_and_scale_normalization",
        "status": "planned",
        "priority": 1,
    },
    {
        "step": 4,
        "component": "benford_eligibility_diagnostics",
        "status": "planned",
        "priority": 1,
    },
    {
        "step": 5,
        "component": "token_exclusion_audit_trail",
        "status": "planned",
        "priority": 2,
    },
    {
        "step": 6,
        "component": "bootstrap_confidence_intervals",
        "status": "planned",
        "priority": 2,
    },
    {
        "step": 7,
        "component": "dataset_comparison_dashboard",
        "status": "planned",
        "priority": 3,
    },
    {
        "step": 8,
        "component": "validated_physics_dataset",
        "status": "planned",
        "priority": 3,
    },
]


@dataclass(frozen=True)
class NumericToken:
    raw: str
    value: float
    start: int
    end: int
    context: str = ""

    @property
    def first_digit(self) -> int | None:
        return first_significant_digit(self.value)


@dataclass(frozen=True)
class BenfordMetrics:
    sample_size: int
    chi_square: float
    mean_absolute_deviation: float
    rmse: float
    kl_divergence: float
    max_absolute_deviation: float


@dataclass(frozen=True)
class BenfordAnalysis:
    label: str
    sample_size: int
    counts: dict[int, int]
    observed: dict[int, float]
    expected: dict[int, float]
    metrics: BenfordMetrics
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass(frozen=True)
class Document:
    label: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetAnalysis:
    label: str
    combined: BenfordAnalysis
    documents: list[BenfordAnalysis]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "combined": self.combined.to_dict(),
            "documents": [document.to_dict() for document in self.documents],
        }


def safe_float(value: str | int | float) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(parsed) or math.isinf(parsed):
        return None

    return parsed


def first_significant_digit(value: str | int | float) -> int | None:
    number = safe_float(value)

    if number is None:
        return None

    number = abs(number)

    if number == 0:
        return None

    while number < 1:
        number *= 10

    while number >= 10:
        number /= 10

    digit = int(number)
    return digit if 1 <= digit <= 9 else None


def context_window(text: str, start: int, end: int, width: int = 50) -> str:
    left = max(0, start - width)
    right = min(len(text), end + width)
    return text[left:right].replace("\n", " ").strip()


def looks_like_year(value: float) -> bool:
    return value.is_integer() and 1900 <= value <= 2099


def looks_like_reference_context(context: str) -> bool:
    return bool(REFERENCE_PATTERN.search(context))


def extract_numeric_tokens(
    text: str,
    *,
    include_years: bool = False,
    include_reference_like: bool = False,
) -> list[NumericToken]:
    tokens: list[NumericToken] = []

    for match in NUMBER_PATTERN.finditer(text):
        raw = match.group(0)
        value = safe_float(raw)

        if value is None:
            continue

        context = context_window(text, match.start(), match.end())

        if not include_years and looks_like_year(value):
            continue

        if not include_reference_like and looks_like_reference_context(context):
            continue

        tokens.append(
            NumericToken(
                raw=raw,
                value=value,
                start=match.start(),
                end=match.end(),
                context=context,
            )
        )

    return tokens


def first_digits_from_tokens(tokens: Iterable[NumericToken]) -> list[int]:
    return [
        digit
        for token in tokens
        if (digit := token.first_digit) is not None
    ]


def first_digits_from_values(values: Iterable[str | int | float]) -> list[int]:
    return [
        digit
        for value in values
        if (digit := first_significant_digit(value)) is not None
    ]


def benford_probabilities() -> dict[int, float]:
    return {digit: math.log10(1 + 1 / digit) for digit in DIGITS}


def uniform_probabilities() -> dict[int, float]:
    return {digit: 1 / 9 for digit in DIGITS}


def count_digits(digits: Iterable[int]) -> dict[int, int]:
    counter = Counter(digit for digit in digits if digit in DIGITS)
    return {digit: counter.get(digit, 0) for digit in DIGITS}


def normalize_counts(counts: Mapping[int, int]) -> dict[int, float]:
    total = sum(counts.get(digit, 0) for digit in DIGITS)

    if total == 0:
        return {digit: 0.0 for digit in DIGITS}

    return {digit: counts.get(digit, 0) / total for digit in DIGITS}


def chi_square_statistic(
    observed: Mapping[int, float],
    expected: Mapping[int, float],
    sample_size: int,
) -> float:
    if sample_size <= 0:
        return 0.0

    statistic = 0.0

    for digit in DIGITS:
        observed_count = observed.get(digit, 0.0) * sample_size
        expected_count = expected[digit] * sample_size

        if expected_count > 0:
            statistic += ((observed_count - expected_count) ** 2) / expected_count

    return statistic


def mean_absolute_deviation(
    observed: Mapping[int, float],
    expected: Mapping[int, float],
) -> float:
    return sum(
        abs(observed.get(digit, 0.0) - expected[digit])
        for digit in DIGITS
    ) / 9


def root_mean_squared_error(
    observed: Mapping[int, float],
    expected: Mapping[int, float],
) -> float:
    squared_error = sum(
        (observed.get(digit, 0.0) - expected[digit]) ** 2
        for digit in DIGITS
    )
    return math.sqrt(squared_error / 9)


def kl_divergence(
    observed: Mapping[int, float],
    expected: Mapping[int, float],
    *,
    epsilon: float = 1e-12,
) -> float:
    total = 0.0

    for digit in DIGITS:
        p = max(observed.get(digit, 0.0), epsilon)
        q = max(expected[digit], epsilon)
        total += p * math.log(p / q)

    return total


def max_absolute_deviation(
    observed: Mapping[int, float],
    expected: Mapping[int, float],
) -> float:
    return max(
        abs(observed.get(digit, 0.0) - expected[digit])
        for digit in DIGITS
    )


def compute_metrics(
    observed: Mapping[int, float],
    expected: Mapping[int, float],
    sample_size: int,
) -> BenfordMetrics:
    return BenfordMetrics(
        sample_size=sample_size,
        chi_square=chi_square_statistic(observed, expected, sample_size),
        mean_absolute_deviation=mean_absolute_deviation(observed, expected),
        rmse=root_mean_squared_error(observed, expected),
        kl_divergence=kl_divergence(observed, expected),
        max_absolute_deviation=max_absolute_deviation(observed, expected),
    )


def analyze_digits(
    digits: Iterable[int],
    *,
    label: str = "analysis",
    metadata: dict[str, str] | None = None,
) -> BenfordAnalysis:
    counts = count_digits(digits)
    observed = normalize_counts(counts)
    expected = benford_probabilities()
    sample_size = sum(counts.values())

    return BenfordAnalysis(
        label=label,
        sample_size=sample_size,
        counts=counts,
        observed=observed,
        expected=expected,
        metrics=compute_metrics(observed, expected, sample_size),
        metadata=metadata or {},
    )


def analyze_text(
    text: str,
    *,
    label: str = "text",
    metadata: dict[str, str] | None = None,
    include_years: bool = False,
    include_reference_like: bool = False,
) -> BenfordAnalysis:
    tokens = extract_numeric_tokens(
        text,
        include_years=include_years,
        include_reference_like=include_reference_like,
    )
    return analyze_digits(
        first_digits_from_tokens(tokens),
        label=label,
        metadata=metadata,
    )


def analyze_document(document: Document) -> BenfordAnalysis:
    return analyze_text(
        document.text,
        label=document.label,
        metadata=document.metadata,
    )


def analyze_dataset(
    documents: Sequence[Document],
    *,
    label: str = "physics-dataset",
) -> DatasetAnalysis:
    document_results = [analyze_document(document) for document in documents]
    combined_text = "\n\n".join(document.text for document in documents)

    combined = analyze_text(
        combined_text,
        label=label,
        metadata={"document_count": str(len(documents))},
    )

    return DatasetAnalysis(
        label=label,
        combined=combined,
        documents=document_results,
    )


def analyze_text_file(path: str | Path) -> BenfordAnalysis:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    return analyze_text(
        text,
        label=file_path.name,
        metadata={"path": str(file_path)},
    )


def load_text_files(folder: str | Path, pattern: str = "*.txt") -> list[Document]:
    folder_path = Path(folder)

    return [
        Document(
            label=path.name,
            text=path.read_text(encoding="utf-8"),
            metadata={"path": str(path)},
        )
        for path in sorted(folder_path.glob(pattern))
    ]


def format_digit_table(result: BenfordAnalysis) -> str:
    lines = [
        "| Digit | Count | Observed | Expected | Difference |",
        "|---:|---:|---:|---:|---:|",
    ]

    for digit in DIGITS:
        observed = result.observed[digit]
        expected = result.expected[digit]
        difference = observed - expected
        lines.append(
            f"| {digit} | {result.counts[digit]} | "
            f"{observed:.4f} | {expected:.4f} | {difference:+.4f} |"
        )

    return "\n".join(lines)


def format_metrics(result: BenfordAnalysis) -> str:
    metrics = result.metrics

    return "\n".join(
        [
            f"- Sample size: {metrics.sample_size}",
            f"- Chi-square statistic: {metrics.chi_square:.4f}",
            f"- Mean absolute deviation: {metrics.mean_absolute_deviation:.4f}",
            f"- RMSE: {metrics.rmse:.4f}",
            f"- KL divergence: {metrics.kl_divergence:.4f}",
            f"- Maximum absolute deviation: {metrics.max_absolute_deviation:.4f}",
        ]
    )


def format_report(result: BenfordAnalysis) -> str:
    parts = [
        f"# Benford analysis: {result.label}",
        "",
        "## First-digit table",
        "",
        format_digit_table(result),
        "",
        "## Metrics",
        "",
        format_metrics(result),
        "",
    ]

    if result.metadata:
        parts.extend(
            ["## Metadata", "", json.dumps(result.metadata, indent=2), ""]
        )

    return "\n".join(parts)


def format_dataset_report(result: DatasetAnalysis) -> str:
    parts = [
        f"# Dataset Benford analysis: {result.label}",
        "",
        "## Combined dataset",
        "",
        format_digit_table(result.combined),
        "",
        "## Combined metrics",
        "",
        format_metrics(result.combined),
        "",
        "## Document-level metrics",
        "",
        "| Document | Sample size | Chi-square | MAD | RMSE | KL divergence |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for document in result.documents:
        metrics = document.metrics
        parts.append(
            f"| {document.label} | {metrics.sample_size} | "
            f"{metrics.chi_square:.4f} | "
            f"{metrics.mean_absolute_deviation:.4f} | "
            f"{metrics.rmse:.4f} | "
            f"{metrics.kl_divergence:.4f} |"
        )

    parts.append("")
    return "\n".join(parts)


def save_report(report: str, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def export_digit_table_csv(result: BenfordAnalysis, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "digit",
                "count",
                "observed",
                "expected",
                "difference",
            ],
        )
        writer.writeheader()

        for digit in DIGITS:
            writer.writerow(
                {
                    "digit": digit,
                    "count": result.counts[digit],
                    "observed": result.observed[digit],
                    "expected": result.expected[digit],
                    "difference": (
                        result.observed[digit] - result.expected[digit]
                    ),
                }
            )


def split_by_section_markers(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = defaultdict(list)
    current = "unknown"

    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()

        if lowered in {
            "abstract",
            "introduction",
            "methods",
            "results",
            "discussion",
        }:
            current = lowered
            continue

        sections[current].append(line)

    return {
        section: "\n".join(values).strip()
        for section, values in sections.items()
    }


def analyze_sections(
    text: str,
    *,
    label_prefix: str = "section",
) -> list[BenfordAnalysis]:
    sections = split_by_section_markers(text)

    return [
        analyze_text(
            section_text,
            label=f"{label_prefix}:{section_name}",
            metadata={"section": section_name},
        )
        for section_name, section_text in sections.items()
        if section_text
    ]


def summarize_sample_sizes(
    results: Sequence[BenfordAnalysis],
) -> dict[str, float]:
    sizes = [result.sample_size for result in results]

    if not sizes:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
        }

    return {
        "count": len(sizes),
        "min": min(sizes),
        "max": max(sizes),
        "mean": statistics.mean(sizes),
        "median": statistics.median(sizes),
    }


def roadmap_by_priority(priority: int) -> list[dict]:
    return [
        item
        for item in PIPELINE_ROADMAP
        if item["priority"] == priority
    ]


def roadmap_status_counts() -> dict[str, int]:
    return dict(Counter(item["status"] for item in PIPELINE_ROADMAP))


def roadmap_component_names() -> list[str]:
    return [item["component"] for item in PIPELINE_ROADMAP]


def roadmap_to_json() -> str:
    return json.dumps(PIPELINE_ROADMAP, indent=2)


def demo_documents() -> list[Document]:
    return [
        Document(
            label="detector-calibration.txt",
            text="""
            Abstract
            The detector recorded 123 events, 47 calibration measurements,
            0.0031 seconds of timing uncertainty, 9.81 m/s^2 acceleration,
            6.022e23 particles per mole, and 1.602e-19 coulombs.

            Results
            The simulation produced 214 trajectories, 389 successful runs,
            144 boundary conditions, 55 rejected parameter sets, and 34
            low-energy states.
            """,
            metadata={"subfield": "instrumentation"},
        ),
        Document(
            label="cosmology-parameters.txt",
            text="""
            Abstract
            We evaluated 72 candidate models using 12 priors and 8 posterior
            diagnostics. The estimated parameter values included 0.315 for
            matter density, 67.4 for the Hubble constant, and 2.725 for the
            cosmic microwave background temperature.

            Discussion
            The Markov chain used 10000 samples, 2500 warmup steps, 4 chains,
            and convergence threshold 1.01.
            """,
            metadata={"subfield": "cosmology"},
        ),
        Document(
            label="particle-constants.txt",
            text="""
            Introduction
            We compare constants including 511 keV electron rest energy,
            938.27 MeV proton mass, 1.054e-34 J*s reduced Planck constant,
            and 2.99792458e8 m/s speed of light.

            Methods
            The table contains 18 constants, 6 unit conversions, 42 derived
            values, and 128 bootstrap resamples.
            """,
            metadata={"subfield": "particle physics"},
        ),
    ]


def run_demo() -> None:
    dataset = analyze_dataset(
        demo_documents(),
        label="demo-physics-dataset",
    )
    print(format_dataset_report(dataset))


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
