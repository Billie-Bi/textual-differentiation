import argparse
import os
import warnings

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

DEFAULT_BASE_OUTPUT_DIRECTORY = "Textual_Indicators_output"
DEFAULT_PUBLICATION_DIRECTORY_NAME = "publication_tables"

RELATIVE_PATH_NARRATIVE_RAW = "narrative-semantic_indicators/01_Narrative-Semantic_Indicators_Raw_Paragraph_Indicators.csv"
RELATIVE_PATH_LEXICAL_RAW = "lexicogrammatical_indicators/01a_Lexicogrammatical_Indicators_Raw_Paragraph_Indicators.csv"
RELATIVE_PATH_REFERENCE_CORPUS = "reference_corpus/reference_corpus_textual_indicators.csv"
RELATIVE_PATH_NARRATIVE_DESC = "narrative-semantic_indicators/02_Narrative-Semantic_Indicators_Descriptive_Statistics.csv"
RELATIVE_PATH_LEXICAL_DESC = "lexicogrammatical_indicators/02_Lexicogrammatical_Indicators_Descriptive_Statistics.csv"
RELATIVE_PATH_NARRATIVE_CORR = "narrative-semantic_indicators/03_Narrative-Semantic_Indicators_Spearman_Correlations.csv"
RELATIVE_PATH_LEXICAL_CORR = "lexicogrammatical_indicators/03_Lexicogrammatical_Indicators_Spearman_Correlations.csv"
RELATIVE_PATH_NARRATIVE_KRUSKAL = "narrative-semantic_indicators/04_Narrative-Semantic_Indicators_Kruskal_Wallis_Tests.csv"
RELATIVE_PATH_LEXICAL_KRUSKAL = "lexicogrammatical_indicators/04_Lexicogrammatical_Indicators_Kruskal_Wallis_Tests.csv"
RELATIVE_PATH_NARRATIVE_MANN_WHITNEY = "narrative-semantic_indicators/05_Narrative-Semantic_Indicators_Mann_Whitney_Tests.csv"
RELATIVE_PATH_LEXICAL_MANN_WHITNEY = "lexicogrammatical_indicators/05_Lexicogrammatical_Indicators_Mann_Whitney_Tests.csv"
RELATIVE_PATH_CHAPTER_DESC = "chapter_level_robustness/02_Chapter_Level_Descriptive_Statistics.csv"
RELATIVE_PATH_CHAPTER_CORR = "chapter_level_robustness/03_Chapter_Level_Spearman_Correlations.csv"
RELATIVE_PATH_CHAPTER_KRUSKAL = "chapter_level_robustness/04_Chapter_Level_Kruskal_Wallis_Tests.csv"
RELATIVE_PATH_CHAPTER_MANN_WHITNEY = "chapter_level_robustness/05_Chapter_Level_Mann_Whitney_Tests.csv"

INPUT_RELATIVE_PATHS = {
    "narrative_raw": RELATIVE_PATH_NARRATIVE_RAW,
    "lexical_raw": RELATIVE_PATH_LEXICAL_RAW,
    "reference_corpus": RELATIVE_PATH_REFERENCE_CORPUS,
    "narrative_descriptive": RELATIVE_PATH_NARRATIVE_DESC,
    "lexical_descriptive": RELATIVE_PATH_LEXICAL_DESC,
    "narrative_correlations": RELATIVE_PATH_NARRATIVE_CORR,
    "lexical_correlations": RELATIVE_PATH_LEXICAL_CORR,
    "narrative_kruskal": RELATIVE_PATH_NARRATIVE_KRUSKAL,
    "lexical_kruskal": RELATIVE_PATH_LEXICAL_KRUSKAL,
    "narrative_mann_whitney": RELATIVE_PATH_NARRATIVE_MANN_WHITNEY,
    "lexical_mann_whitney": RELATIVE_PATH_LEXICAL_MANN_WHITNEY,
    "chapter_descriptive": RELATIVE_PATH_CHAPTER_DESC,
    "chapter_correlations": RELATIVE_PATH_CHAPTER_CORR,
    "chapter_kruskal": RELATIVE_PATH_CHAPTER_KRUSKAL,
    "chapter_mann_whitney": RELATIVE_PATH_CHAPTER_MANN_WHITNEY,
}

OUTPUT_FILENAME_TABLE_1 = "Table 1. Summary of narrative-semantic indicators and reference comparisons.csv"
OUTPUT_FILENAME_TABLE_2 = "Table 2. Summary of lexicogrammatical indicators and reference comparisons.csv"
OUTPUT_FILENAME_S1 = "S1 File. Comprehensive descriptive statistics for narrative-semantic and lexicogrammatical indicators.csv"
OUTPUT_FILENAME_S2 = "S2 File. Kruskal-Wallis test statistics for the eight indicators across the three parts.csv"
OUTPUT_FILENAME_S3 = "S3 File. Pairwise Mann-Whitney U tests for the eight indicators with Benjamini-Hochberg FDR correction.csv"
OUTPUT_FILENAME_S4 = "S4 File. Spearman correlations among the computational textual indicators.csv"

OUTPUT_FILENAMES = {
    "table_1": OUTPUT_FILENAME_TABLE_1,
    "table_2": OUTPUT_FILENAME_TABLE_2,
    "s1": OUTPUT_FILENAME_S1,
    "s2": OUTPUT_FILENAME_S2,
    "s3": OUTPUT_FILENAME_S3,
    "s4": OUTPUT_FILENAME_S4,
}

GROUP_NAME_NARRATIVE = "Narrative-semantic"
GROUP_NAME_LEXICAL = "Lexicogrammatical"
ANALYSIS_LEVEL_PARAGRAPH = "Paragraph"
ANALYSIS_LEVEL_CHAPTER = "Chapter"

NOVEL_PART_ORDER = [
    "The Window",
    "Time Passes",
    "The Lighthouse",
]
ANALYSIS_GROUP_ORDER = [
    GROUP_NAME_NARRATIVE,
    GROUP_NAME_LEXICAL,
]
ANALYSIS_LEVEL_ORDER = [
    ANALYSIS_LEVEL_PARAGRAPH,
    ANALYSIS_LEVEL_CHAPTER,
]
SCOPE_ORDER = [
    "Overall",
    "The Window",
    "Time Passes",
    "The Lighthouse",
]
COMPARISON_ORDER = [
    "The Window vs Time Passes",
    "The Window vs The Lighthouse",
    "Time Passes vs The Lighthouse",
]

TARGET_NOVEL_NAME = "To the Lighthouse"
REFERENCE_NOVEL_BASELINE = "Mrs. Dalloway"
REFERENCE_BAND_LABEL = "19th-Century Reference Band"
REFERENCE_TEXTS_HEADER = "Reference texts"
OVERALL_SCOPE_LABEL = "Overall"

FORMAT_DECIMAL_PLACES_STANDARD = 4
FORMAT_DECIMAL_PLACES_P_VALUE = 3
FORMAT_P_VALUE_MINIMUM = 0.001
STATISTICAL_SIGNIFICANCE_ALPHA = 0.05
FDR_CORRECTION_METHOD = "fdr_bh"
CSV_ENCODING = "utf-8-sig"

NARRATIVE_SEMANTIC_METRICS = [
    "event_density",
    "perspective_shift",
    "psychological_density",
    "temporal_density",
]
LEXICOGRAMMATICAL_METRICS = [
    "pos_ratio_adj_adv_to_verb",
    "avg_dependency_depth",
    "repeated_words_per_100_tokens",
    "parallel_ratio",
]
ALL_TARGET_METRICS = NARRATIVE_SEMANTIC_METRICS + LEXICOGRAMMATICAL_METRICS

METRIC_DISPLAY_LABELS = {
    "event_density": "Event Density",
    "perspective_shift": "Perspective Shift",
    "psychological_density": "Psychological Density",
    "temporal_density": "Temporal Density",
    "pos_ratio_adj_adv_to_verb": "POS Ratio of Adjectives and Adverbs to Verbs",
    "avg_dependency_depth": "Average Dependency Depth",
    "repeated_words_per_100_tokens": "Repeated Words per 100 Tokens",
    "parallel_ratio": "Parallel Ratio",
}

METRIC_DISPLAY_ALIASES = {
    "event density": "Event Density",
    "event_density": "Event Density",
    "perspective shift": "Perspective Shift",
    "perspective_shift": "Perspective Shift",
    "psychological density": "Psychological Density",
    "psychological_density": "Psychological Density",
    "temporal density": "Temporal Density",
    "temporal_density": "Temporal Density",
    "pos ratio adj adv to verb": "POS Ratio of Adjectives and Adverbs to Verbs",
    "pos_ratio_adj_adv_to_verb": "POS Ratio of Adjectives and Adverbs to Verbs",
    "pos ratio of adjectives and adverbs to verbs": "POS Ratio of Adjectives and Adverbs to Verbs",
    "avg dependency depth": "Average Dependency Depth",
    "average dependency depth": "Average Dependency Depth",
    "avg_dependency_depth": "Average Dependency Depth",
    "repeated words per 100 tokens": "Repeated Words per 100 Tokens",
    "repeated_words_per_100_tokens": "Repeated Words per 100 Tokens",
    "parallel ratio": "Parallel Ratio",
    "parallel_ratio": "Parallel Ratio",
}

ORDERED_INDICATOR_LABELS = [METRIC_DISPLAY_LABELS[metric] for metric in ALL_TARGET_METRICS]

TABLE_1_HEADERS = [
    "Category / Part",
    "Event Density",
    "Perspective Shift",
    "Psychological Density",
    "Temporal Density",
]
TABLE_2_HEADERS = [
    "Category / Part",
    "POS Ratio (Adj+Adv/Verb)",
    "Average Dependency Depth",
    "Repeated Words (/100 tokens)",
    "Parallel Ratio",
]


def parse_command_line_arguments():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--base-output-directory",
        default=DEFAULT_BASE_OUTPUT_DIRECTORY,
    )
    argument_parser.add_argument(
        "--publication-output-directory",
        default=None,
    )
    return argument_parser.parse_args()


def validate_file_existence(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERROR] Required output file not found: {filepath}")
    return filepath


def load_csv_dataframe(filepath):
    return pd.read_csv(validate_file_existence(filepath))


def resolve_canonical_indicator(raw_value):
    if pd.isna(raw_value):
        return ""
    normalized_text = str(raw_value).strip()
    search_key = " ".join(normalized_text.replace("_", " ").split()).lower()
    return METRIC_DISPLAY_ALIASES.get(search_key, normalized_text)


def resolve_canonical_group(raw_value, fallback_group):
    if pd.isna(raw_value) or not str(raw_value).strip():
        return fallback_group
    normalized_text = str(raw_value).strip().lower()
    if "narrative" in normalized_text:
        return GROUP_NAME_NARRATIVE
    if "lexico" in normalized_text:
        return GROUP_NAME_LEXICAL
    return str(raw_value).strip()


def format_p_value_for_publication(p_value_numeric):
    if pd.isna(p_value_numeric):
        return ""
    p_value_numeric = float(p_value_numeric)
    if p_value_numeric < 0:
        raise ValueError(f"[ERROR] Invalid negative p-value: {p_value_numeric}")
    if p_value_numeric < FORMAT_P_VALUE_MINIMUM:
        return f"<{FORMAT_P_VALUE_MINIMUM}"
    return f"{p_value_numeric:.{FORMAT_DECIMAL_PLACES_P_VALUE}f}"


def format_fixed_decimal(
    numeric_value,
    decimal_places=FORMAT_DECIMAL_PLACES_STANDARD,
):
    if pd.isna(numeric_value):
        return ""
    numeric_value = float(numeric_value)
    if numeric_value == 0:
        return f"{0:.{decimal_places}f}"

    small_value_threshold = 10 ** (-decimal_places)
    if 0 < numeric_value < small_value_threshold:
        return f"<{small_value_threshold:.{decimal_places}f}"
    if -small_value_threshold < numeric_value < 0:
        return f">-{small_value_threshold:.{decimal_places}f}"
    return f"{numeric_value:.{decimal_places}f}"


def format_signed_estimate_value(
    numeric_value,
    decimal_places=FORMAT_DECIMAL_PLACES_P_VALUE,
):
    if pd.isna(numeric_value):
        return ""
    numeric_value = float(numeric_value)
    if numeric_value == 0:
        return f"{0:.{decimal_places}f}"
    rounding_zero_limit = 0.5 * (10 ** (-decimal_places))
    if abs(numeric_value) < rounding_zero_limit:
        return f"{numeric_value:.2e}"
    return f"{numeric_value:.{decimal_places}f}"


def format_test_statistic(
    numeric_value,
    decimal_places=FORMAT_DECIMAL_PLACES_P_VALUE,
):
    if pd.isna(numeric_value):
        return ""
    return f"{float(numeric_value):.{decimal_places}f}"


def format_mann_whitney_u_statistic(numeric_value):
    if pd.isna(numeric_value):
        return ""
    numeric_value = float(numeric_value)
    if np.isclose(numeric_value, round(numeric_value)):
        return str(int(round(numeric_value)))
    return f"{numeric_value:.1f}"


def format_boolean_to_yes_no(boolean_value):
    if pd.isna(boolean_value):
        return ""
    if isinstance(boolean_value, str):
        normalized_value = boolean_value.strip().lower()
        return "Yes" if normalized_value in {"true", "yes", "1"} else "No"
    return "Yes" if bool(boolean_value) else "No"


def validate_and_filter_raw_data(
    dataframe,
    filepath,
    target_metrics,
):
    required_columns = [
        "part",
        "chapter_id",
        "para_idx",
    ] + target_metrics
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"[ERROR] Missing columns in {filepath}: {missing_columns}")
    return dataframe[required_columns].copy()


def build_main_publication_table(
    raw_metrics_dataframe,
    reference_corpus_dataframe,
    target_metrics,
    table_headers,
):
    table_rows = []
    blank_metric_values = dict.fromkeys(table_headers[1:], "")

    table_rows.append(
        {
            table_headers[0]: TARGET_NOVEL_NAME,
            **blank_metric_values,
        }
    )

    analysis_scopes = [(OVERALL_SCOPE_LABEL, raw_metrics_dataframe)]
    analysis_scopes.extend(
        (
            f"\u201c{part}\u201d",
            raw_metrics_dataframe[raw_metrics_dataframe["part"] == part],
        )
        for part in NOVEL_PART_ORDER
    )

    for scope_name, scope_dataframe in analysis_scopes:
        current_row = {table_headers[0]: scope_name}
        for metric, header in zip(
            target_metrics,
            table_headers[1:],
        ):
            current_row[header] = format_fixed_decimal(scope_dataframe[metric].mean())
        table_rows.append(current_row)

    table_rows.append(
        {
            table_headers[0]: REFERENCE_TEXTS_HEADER,
            **blank_metric_values,
        }
    )

    mrs_dalloway_records = reference_corpus_dataframe[reference_corpus_dataframe["Novel"] == REFERENCE_NOVEL_BASELINE]
    if mrs_dalloway_records.empty:
        raise ValueError("[ERROR] Mrs. Dalloway is missing from the reference output.")

    mrs_dalloway_target_row = mrs_dalloway_records.iloc[0]
    current_row = {table_headers[0]: REFERENCE_NOVEL_BASELINE}
    for metric, header in zip(
        target_metrics,
        table_headers[1:],
    ):
        current_row[header] = format_fixed_decimal(mrs_dalloway_target_row[metric])
    table_rows.append(current_row)

    nineteenth_century_dataframe = reference_corpus_dataframe[reference_corpus_dataframe["Novel"] != REFERENCE_NOVEL_BASELINE]
    if nineteenth_century_dataframe.empty:
        raise ValueError("[ERROR] No 19th-century reference texts were found.")

    current_row = {table_headers[0]: REFERENCE_BAND_LABEL}
    for metric, header in zip(
        target_metrics,
        table_headers[1:],
    ):
        band_minimum = format_fixed_decimal(nineteenth_century_dataframe[metric].min())
        band_maximum = format_fixed_decimal(nineteenth_century_dataframe[metric].max())
        current_row[header] = f"[{band_minimum}\u2013{band_maximum}]"
    table_rows.append(current_row)

    return pd.DataFrame(
        table_rows,
        columns=table_headers,
    )


def normalize_descriptive_statistics_dataframe(
    dataframe,
    fallback_group_name,
    analysis_level,
):
    dataframe = dataframe.rename(
        columns={
            "Feature": "Indicator",
            "Std": "SD",
            "Chapter Count": "Count",
        }
    ).copy()

    if "Indicator Group" not in dataframe.columns:
        dataframe.insert(
            0,
            "Indicator Group",
            fallback_group_name,
        )
    dataframe.insert(
        0,
        "Analysis Level",
        analysis_level,
    )

    expected_columns = [
        "Analysis Level",
        "Indicator Group",
        "Indicator",
        "Scope",
        "Count",
        "Mean",
        "SD",
        "Median",
        "IQR",
        "Skewness",
        "Kurtosis",
    ]
    missing_columns = [column for column in expected_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"[ERROR] Descriptive statistics missing columns: {missing_columns}")

    indicator_text_series = dataframe["Indicator"].fillna("").astype(str).str.strip()
    scope_text_series = dataframe["Scope"].fillna("").astype(str).str.strip()

    dataframe = dataframe[
        indicator_text_series.ne("")
        & scope_text_series.ne("")
        & indicator_text_series.str.lower().ne("indicator")
        & indicator_text_series.str.lower().ne("feature")
        & scope_text_series.str.lower().ne("scope")
    ].copy()

    dataframe["Indicator Group"] = dataframe["Indicator Group"].apply(
        lambda value: resolve_canonical_group(
            value,
            fallback_group_name,
        )
    )
    dataframe["Indicator"] = dataframe["Indicator"].apply(resolve_canonical_indicator)

    numeric_columns = [
        "Count",
        "Mean",
        "SD",
        "Median",
        "IQR",
        "Skewness",
        "Kurtosis",
    ]
    for column_name in numeric_columns:
        dataframe[column_name] = pd.to_numeric(
            dataframe[column_name],
            errors="coerce",
        )

    dataframe = dataframe.dropna(
        subset=[
            "Count",
            "Indicator",
            "Scope",
        ]
    ).copy()
    return dataframe[expected_columns]


def calculate_epsilon_squared(
    h_statistic,
    total_sample_size,
    group_count,
):
    denominator = total_sample_size - group_count
    if denominator <= 0:
        return np.nan
    epsilon_squared = (float(h_statistic) - group_count + 1) / denominator
    return max(epsilon_squared, 0.0)


def normalize_kruskal_wallis_dataframe(
    dataframe,
    fallback_group_name,
    raw_metrics_dataframe,
    analysis_level,
):
    dataframe = dataframe.rename(
        columns={
            "Feature": "Indicator",
            "Statistic": "H Statistic",
            "p-value": "Unadjusted p-value",
        }
    ).copy()

    if "Indicator Group" not in dataframe.columns:
        dataframe.insert(
            0,
            "Indicator Group",
            fallback_group_name,
        )
    dataframe.insert(
        0,
        "Analysis Level",
        analysis_level,
    )

    if "df" not in dataframe.columns:
        dataframe["df"] = len(NOVEL_PART_ORDER) - 1

    for novel_part in NOVEL_PART_ORDER:
        sample_size_column = f"n ({novel_part})"
        if sample_size_column not in dataframe.columns:
            if raw_metrics_dataframe is None:
                raise ValueError("[ERROR] Missing sample sizes in Kruskal-Wallis output.")
            dataframe[sample_size_column] = int((raw_metrics_dataframe["part"] == novel_part).sum())

    required_columns = [
        "Analysis Level",
        "Indicator Group",
        "Indicator",
        "H Statistic",
        "df",
        "n (The Window)",
        "n (Time Passes)",
        "n (The Lighthouse)",
        "Unadjusted p-value",
    ]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"[ERROR] Kruskal-Wallis output missing columns: {missing_columns}")

    indicator_text_series = dataframe["Indicator"].fillna("").astype(str).str.strip()
    dataframe = dataframe[
        indicator_text_series.ne("") & indicator_text_series.str.lower().ne("indicator") & indicator_text_series.str.lower().ne("feature")
    ].copy()

    dataframe["Indicator Group"] = dataframe["Indicator Group"].apply(
        lambda value: resolve_canonical_group(
            value,
            fallback_group_name,
        )
    )
    dataframe["Indicator"] = dataframe["Indicator"].apply(resolve_canonical_indicator)

    numeric_columns = [
        "H Statistic",
        "df",
        "n (The Window)",
        "n (Time Passes)",
        "n (The Lighthouse)",
        "Unadjusted p-value",
    ]
    for column_name in numeric_columns:
        dataframe[column_name] = pd.to_numeric(
            dataframe[column_name],
            errors="coerce",
        )

    dataframe = dataframe.dropna(
        subset=[
            "H Statistic",
            "df",
            "Unadjusted p-value",
        ]
    ).copy()

    total_sample_size = dataframe["n (The Window)"] + dataframe["n (Time Passes)"] + dataframe["n (The Lighthouse)"]
    group_count = len(NOVEL_PART_ORDER)

    if "Epsilon Squared" not in dataframe.columns:
        dataframe["Epsilon Squared"] = [
            calculate_epsilon_squared(
                h_statistic,
                sample_size,
                group_count,
            )
            for h_statistic, sample_size in zip(
                dataframe["H Statistic"],
                total_sample_size,
            )
        ]
    else:
        dataframe["Epsilon Squared"] = pd.to_numeric(
            dataframe["Epsilon Squared"],
            errors="coerce",
        )

    dataframe["Adjusted p-value (BH FDR)"] = np.nan
    for indicator_group, group_indices in dataframe.groupby(
        "Indicator Group",
        sort=False,
    ).groups.items():
        group_index_list = list(group_indices)
        group_p_values = dataframe.loc[
            group_index_list,
            "Unadjusted p-value",
        ]
        finite_p_value_mask = np.isfinite(group_p_values)
        finite_group_indices = group_p_values.index[finite_p_value_mask]
        if len(finite_group_indices) > 0:
            dataframe.loc[
                finite_group_indices,
                "Adjusted p-value (BH FDR)",
            ] = multipletests(
                dataframe.loc[
                    finite_group_indices,
                    "Unadjusted p-value",
                ],
                method=FDR_CORRECTION_METHOD,
            )[1]

    dataframe["Significant after FDR correction"] = dataframe["Adjusted p-value (BH FDR)"] < STATISTICAL_SIGNIFICANCE_ALPHA

    expected_columns = required_columns + [
        "Adjusted p-value (BH FDR)",
        "Epsilon Squared",
        "Significant after FDR correction",
    ]
    return dataframe[expected_columns]


def extract_comparison_sample_sizes(
    comparison_string,
    raw_metrics_dataframe,
):
    if not isinstance(comparison_string, str) or " vs " not in comparison_string:
        return np.nan, np.nan
    if raw_metrics_dataframe is None:
        return np.nan, np.nan

    part_1_name, part_2_name = comparison_string.split(
        " vs ",
        1,
    )
    sample_size_1 = int((raw_metrics_dataframe["part"] == part_1_name).sum())
    sample_size_2 = int((raw_metrics_dataframe["part"] == part_2_name).sum())
    return sample_size_1, sample_size_2


def apply_grouped_fdr_correction(
    dataframe,
    p_value_column,
):
    adjusted_p_values = pd.Series(
        np.nan,
        index=dataframe.index,
        dtype=float,
    )

    for _, group_dataframe in dataframe.groupby(
        "Indicator Group",
        sort=False,
    ):
        finite_mask = np.isfinite(group_dataframe[p_value_column])
        if not finite_mask.any():
            continue
        valid_indices = group_dataframe.index[finite_mask]
        adjusted_p_values.loc[valid_indices] = multipletests(
            group_dataframe.loc[
                valid_indices,
                p_value_column,
            ].to_numpy(dtype=float),
            method=FDR_CORRECTION_METHOD,
        )[1]

    return adjusted_p_values


def normalize_mann_whitney_dataframe(
    dataframe,
    fallback_group_name,
    raw_metrics_dataframe,
    analysis_level,
):
    dataframe = dataframe.rename(
        columns={
            "Feature": "Indicator",
            "Statistic": "Mann-Whitney U",
            "p-value": "Unadjusted p-value",
            "Corrected p-value": ("Adjusted p-value (BH FDR)"),
            "Effect Size": "Cohen's d",
            "Effect Size (Cohen's d)": "Cohen's d",
        }
    ).copy()

    if "Indicator Group" not in dataframe.columns:
        dataframe.insert(
            0,
            "Indicator Group",
            fallback_group_name,
        )
    dataframe.insert(
        0,
        "Analysis Level",
        analysis_level,
    )

    indicator_text_series = (
        dataframe.get(
            "Indicator",
            pd.Series(index=dataframe.index, dtype=object),
        )
        .fillna("")
        .astype(str)
        .str.strip()
    )
    comparison_text_series = (
        dataframe.get(
            "Comparison",
            pd.Series(index=dataframe.index, dtype=object),
        )
        .fillna("")
        .astype(str)
        .str.strip()
    )

    dataframe = dataframe[
        indicator_text_series.ne("")
        & comparison_text_series.str.contains(
            " vs ",
            regex=False,
        )
        & indicator_text_series.str.lower().ne("indicator")
        & indicator_text_series.str.lower().ne("feature")
    ].copy()

    dataframe["Indicator Group"] = dataframe["Indicator Group"].apply(
        lambda value: resolve_canonical_group(
            value,
            fallback_group_name,
        )
    )
    dataframe["Indicator"] = dataframe["Indicator"].apply(resolve_canonical_indicator)

    if "n1" not in dataframe.columns or "n2" not in dataframe.columns:
        sample_sizes = dataframe["Comparison"].apply(
            lambda value: extract_comparison_sample_sizes(
                value,
                raw_metrics_dataframe,
            )
        )
        dataframe["n1"] = [pair[0] for pair in sample_sizes]
        dataframe["n2"] = [pair[1] for pair in sample_sizes]

    numeric_columns = [
        "n1",
        "n2",
        "Mann-Whitney U",
        "Unadjusted p-value",
        "Cohen's d",
    ]
    for column_name in numeric_columns:
        if column_name in dataframe.columns:
            dataframe[column_name] = pd.to_numeric(
                dataframe[column_name],
                errors="coerce",
            )

    dataframe = dataframe.dropna(
        subset=[
            "n1",
            "n2",
            "Mann-Whitney U",
            "Unadjusted p-value",
        ]
    ).copy()

    if "Rank-biserial correlation" not in dataframe.columns:
        denominator = (dataframe["n1"] * dataframe["n2"]).replace(0, np.nan)
        dataframe["Rank-biserial correlation"] = 2.0 * dataframe["Mann-Whitney U"] / denominator - 1.0
    else:
        dataframe["Rank-biserial correlation"] = pd.to_numeric(
            dataframe["Rank-biserial correlation"],
            errors="coerce",
        )

    if "Adjusted p-value (BH FDR)" not in dataframe.columns:
        dataframe["Adjusted p-value (BH FDR)"] = apply_grouped_fdr_correction(
            dataframe,
            "Unadjusted p-value",
        )
    else:
        dataframe["Adjusted p-value (BH FDR)"] = pd.to_numeric(
            dataframe["Adjusted p-value (BH FDR)"],
            errors="coerce",
        )

    if "Significant after FDR correction" not in dataframe.columns:
        dataframe["Significant after FDR correction"] = dataframe["Adjusted p-value (BH FDR)"] < STATISTICAL_SIGNIFICANCE_ALPHA

    expected_columns = [
        "Analysis Level",
        "Indicator Group",
        "Indicator",
        "Comparison",
        "n1",
        "n2",
        "Mann-Whitney U",
        "Unadjusted p-value",
        "Adjusted p-value (BH FDR)",
        "Significant after FDR correction",
        "Rank-biserial correlation",
        "Cohen's d",
    ]
    missing_columns = [column for column in expected_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"[ERROR] Mann-Whitney output missing columns: {missing_columns}")
    return dataframe[expected_columns]


def normalize_correlations_dataframe(
    dataframe,
    fallback_group_name,
    raw_metrics_dataframe,
    analysis_level,
):
    dataframe = dataframe.rename(
        columns={
            "Metric 1": "Indicator 1",
            "Metric 2": "Indicator 2",
            "Correlation (r)": "Spearman's rho",
            "P-value": "Unadjusted p-value",
            "Chapter Count": "N",
        }
    ).copy()

    if "Indicator Group" not in dataframe.columns:
        dataframe.insert(
            0,
            "Indicator Group",
            fallback_group_name,
        )
    dataframe.insert(
        0,
        "Analysis Level",
        analysis_level,
    )

    indicator_1_text_series = (
        dataframe.get(
            "Indicator 1",
            pd.Series(index=dataframe.index, dtype=object),
        )
        .fillna("")
        .astype(str)
        .str.strip()
    )
    indicator_2_text_series = (
        dataframe.get(
            "Indicator 2",
            pd.Series(index=dataframe.index, dtype=object),
        )
        .fillna("")
        .astype(str)
        .str.strip()
    )

    dataframe = dataframe[
        indicator_1_text_series.ne("")
        & indicator_2_text_series.ne("")
        & indicator_1_text_series.str.lower().ne("indicator 1")
        & indicator_1_text_series.str.lower().ne("metric 1")
    ].copy()

    dataframe["Indicator Group"] = dataframe["Indicator Group"].apply(
        lambda value: resolve_canonical_group(
            value,
            fallback_group_name,
        )
    )
    dataframe["Indicator 1"] = dataframe["Indicator 1"].apply(resolve_canonical_indicator)
    dataframe["Indicator 2"] = dataframe["Indicator 2"].apply(resolve_canonical_indicator)

    if "N" not in dataframe.columns:
        if raw_metrics_dataframe is None:
            raise ValueError("[ERROR] Correlation sample size is missing.")
        dataframe["N"] = len(raw_metrics_dataframe)

    numeric_columns = [
        "N",
        "Spearman's rho",
        "Unadjusted p-value",
    ]
    for column_name in numeric_columns:
        dataframe[column_name] = pd.to_numeric(
            dataframe[column_name],
            errors="coerce",
        )

    dataframe = dataframe.dropna(
        subset=[
            "N",
            "Spearman's rho",
            "Unadjusted p-value",
        ]
    ).copy()

    if "Adjusted p-value (BH FDR)" not in dataframe.columns:
        dataframe["Adjusted p-value (BH FDR)"] = apply_grouped_fdr_correction(
            dataframe,
            "Unadjusted p-value",
        )
    else:
        dataframe["Adjusted p-value (BH FDR)"] = pd.to_numeric(
            dataframe["Adjusted p-value (BH FDR)"],
            errors="coerce",
        )

    if "Significant after FDR correction" not in dataframe.columns:
        dataframe["Significant after FDR correction"] = dataframe["Adjusted p-value (BH FDR)"] < STATISTICAL_SIGNIFICANCE_ALPHA

    expected_columns = [
        "Analysis Level",
        "Indicator Group",
        "Indicator 1",
        "Indicator 2",
        "N",
        "Spearman's rho",
        "Unadjusted p-value",
        "Adjusted p-value (BH FDR)",
        "Significant after FDR correction",
    ]
    missing_columns = [column for column in expected_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"[ERROR] Correlation output missing columns: {missing_columns}")
    return dataframe[expected_columns]


def sort_dataframe_by_publication_order(
    dataframe,
    indicator_column_name="Indicator",
):
    group_rank_mapping = {group_name: index for index, group_name in enumerate(ANALYSIS_GROUP_ORDER)}
    indicator_rank_mapping = {indicator_name: index for index, indicator_name in enumerate(ORDERED_INDICATOR_LABELS)}
    analysis_level_rank_mapping = {analysis_level: index for index, analysis_level in enumerate(ANALYSIS_LEVEL_ORDER)}
    scope_rank_mapping = {scope_name: index for index, scope_name in enumerate(SCOPE_ORDER)}
    comparison_rank_mapping = {comparison_name: index for index, comparison_name in enumerate(COMPARISON_ORDER)}

    sorted_dataframe = dataframe.copy()
    sorted_dataframe["_group_rank"] = sorted_dataframe["Indicator Group"].map(group_rank_mapping)
    sorted_dataframe["_indicator_rank"] = sorted_dataframe[indicator_column_name].map(indicator_rank_mapping)
    sorted_dataframe["_analysis_level_rank"] = sorted_dataframe["Analysis Level"].map(analysis_level_rank_mapping)

    sort_columns = [
        "_group_rank",
        "_indicator_rank",
        "_analysis_level_rank",
    ]

    if "Scope" in sorted_dataframe.columns:
        sorted_dataframe["_scope_rank"] = sorted_dataframe["Scope"].map(scope_rank_mapping)
        sort_columns.append("_scope_rank")

    if "Comparison" in sorted_dataframe.columns:
        sorted_dataframe["_comparison_rank"] = sorted_dataframe["Comparison"].map(comparison_rank_mapping)
        sort_columns.append("_comparison_rank")

    sorted_dataframe = sorted_dataframe.sort_values(
        sort_columns,
        kind="stable",
    )
    helper_columns = [column for column in sorted_dataframe.columns if column.startswith("_")]
    return sorted_dataframe.drop(columns=helper_columns)


def generate_publication_supplementary_file_1(
    paragraph_narrative_dataframe,
    paragraph_lexical_dataframe,
    chapter_dataframe,
):
    combined_dataframe = pd.concat(
        [
            paragraph_narrative_dataframe,
            paragraph_lexical_dataframe,
            chapter_dataframe,
        ],
        ignore_index=True,
    )
    combined_dataframe = sort_dataframe_by_publication_order(combined_dataframe)
    output_dataframe = combined_dataframe.copy()

    output_dataframe["Count"] = output_dataframe["Count"].astype(int).astype(str)
    for column_name in [
        "Mean",
        "SD",
        "Median",
        "IQR",
    ]:
        output_dataframe[column_name] = output_dataframe[column_name].apply(format_fixed_decimal)
    for column_name in [
        "Skewness",
        "Kurtosis",
    ]:
        output_dataframe[column_name] = output_dataframe[column_name].apply(
            lambda value: format_signed_estimate_value(
                value,
                decimal_places=(FORMAT_DECIMAL_PLACES_STANDARD),
            )
        )
    return output_dataframe


def generate_publication_supplementary_file_2(
    paragraph_narrative_dataframe,
    paragraph_lexical_dataframe,
    chapter_dataframe,
):
    combined_dataframe = pd.concat(
        [
            paragraph_narrative_dataframe,
            paragraph_lexical_dataframe,
            chapter_dataframe,
        ],
        ignore_index=True,
    )
    combined_dataframe = sort_dataframe_by_publication_order(combined_dataframe)
    output_dataframe = combined_dataframe.copy()

    output_dataframe["H Statistic"] = output_dataframe["H Statistic"].apply(format_test_statistic)
    output_dataframe["df"] = output_dataframe["df"].astype(int).astype(str)

    sample_size_columns = [
        "n (The Window)",
        "n (Time Passes)",
        "n (The Lighthouse)",
    ]
    for column_name in sample_size_columns:
        output_dataframe[column_name] = output_dataframe[column_name].astype(int).astype(str)

    output_dataframe["Unadjusted p-value"] = output_dataframe["Unadjusted p-value"].apply(format_p_value_for_publication)
    output_dataframe["Adjusted p-value (BH FDR)"] = output_dataframe["Adjusted p-value (BH FDR)"].apply(format_p_value_for_publication)
    output_dataframe["Epsilon Squared"] = output_dataframe["Epsilon Squared"].apply(format_signed_estimate_value)
    output_dataframe["Significant after FDR correction"] = output_dataframe["Significant after FDR correction"].apply(
        format_boolean_to_yes_no
    )
    return output_dataframe


def generate_publication_supplementary_file_3(
    paragraph_narrative_dataframe,
    paragraph_lexical_dataframe,
    chapter_dataframe,
):
    combined_dataframe = pd.concat(
        [
            paragraph_narrative_dataframe,
            paragraph_lexical_dataframe,
            chapter_dataframe,
        ],
        ignore_index=True,
    )
    combined_dataframe = sort_dataframe_by_publication_order(combined_dataframe)
    output_dataframe = combined_dataframe.copy()

    output_dataframe["n1"] = output_dataframe["n1"].astype(int).astype(str)
    output_dataframe["n2"] = output_dataframe["n2"].astype(int).astype(str)
    output_dataframe["Mann-Whitney U"] = output_dataframe["Mann-Whitney U"].apply(format_mann_whitney_u_statistic)
    output_dataframe["Unadjusted p-value"] = output_dataframe["Unadjusted p-value"].apply(format_p_value_for_publication)
    output_dataframe["Adjusted p-value (BH FDR)"] = output_dataframe["Adjusted p-value (BH FDR)"].apply(format_p_value_for_publication)
    output_dataframe["Significant after FDR correction"] = output_dataframe["Significant after FDR correction"].apply(
        format_boolean_to_yes_no
    )
    output_dataframe["Rank-biserial correlation"] = output_dataframe["Rank-biserial correlation"].apply(format_signed_estimate_value)
    output_dataframe["Cohen's d"] = output_dataframe["Cohen's d"].apply(format_signed_estimate_value)
    return output_dataframe


def generate_publication_supplementary_file_4(
    paragraph_narrative_dataframe,
    paragraph_lexical_dataframe,
    chapter_dataframe,
):
    combined_dataframe = pd.concat(
        [
            paragraph_narrative_dataframe,
            paragraph_lexical_dataframe,
            chapter_dataframe,
        ],
        ignore_index=True,
    )
    combined_dataframe = sort_dataframe_by_publication_order(
        combined_dataframe,
        indicator_column_name="Indicator 1",
    )
    output_dataframe = combined_dataframe.copy()

    output_dataframe["N"] = output_dataframe["N"].astype(int).astype(str)
    output_dataframe["Spearman's rho"] = output_dataframe["Spearman's rho"].apply(format_signed_estimate_value)
    output_dataframe["Unadjusted p-value"] = output_dataframe["Unadjusted p-value"].apply(format_p_value_for_publication)
    output_dataframe["Adjusted p-value (BH FDR)"] = output_dataframe["Adjusted p-value (BH FDR)"].apply(format_p_value_for_publication)
    output_dataframe["Significant after FDR correction"] = output_dataframe["Significant after FDR correction"].apply(
        format_boolean_to_yes_no
    )
    return output_dataframe


def build_input_filepaths(base_output_directory):
    return {name: os.path.join(base_output_directory, relative_path) for name, relative_path in INPUT_RELATIVE_PATHS.items()}


def build_output_filepaths(publication_output_directory):
    return {name: os.path.join(publication_output_directory, filename) for name, filename in OUTPUT_FILENAMES.items()}


def run_pipeline():
    command_line_arguments = parse_command_line_arguments()
    base_output_directory = command_line_arguments.base_output_directory
    publication_output_directory = command_line_arguments.publication_output_directory or os.path.join(
        base_output_directory,
        DEFAULT_PUBLICATION_DIRECTORY_NAME,
    )
    os.makedirs(
        publication_output_directory,
        exist_ok=True,
    )

    input_filepaths = build_input_filepaths(base_output_directory)
    output_filepaths = build_output_filepaths(publication_output_directory)

    print("[PROCESS] Loading and validating raw metrics...")
    narrative_raw_dataframe = validate_and_filter_raw_data(
        load_csv_dataframe(input_filepaths["narrative_raw"]),
        input_filepaths["narrative_raw"],
        NARRATIVE_SEMANTIC_METRICS,
    )
    lexicogrammatical_raw_dataframe = validate_and_filter_raw_data(
        load_csv_dataframe(input_filepaths["lexical_raw"]),
        input_filepaths["lexical_raw"],
        LEXICOGRAMMATICAL_METRICS,
    )

    reference_corpus_dataframe = load_csv_dataframe(input_filepaths["reference_corpus"])
    reference_required_columns = ["Novel"] + ALL_TARGET_METRICS
    missing_reference_columns = [column for column in reference_required_columns if column not in reference_corpus_dataframe.columns]
    if missing_reference_columns:
        raise ValueError(f"[ERROR] Reference output missing columns: {missing_reference_columns}")

    print("[PROCESS] Generating main publication tables...")
    table_1_dataframe = build_main_publication_table(
        narrative_raw_dataframe,
        reference_corpus_dataframe,
        NARRATIVE_SEMANTIC_METRICS,
        TABLE_1_HEADERS,
    )
    table_2_dataframe = build_main_publication_table(
        lexicogrammatical_raw_dataframe,
        reference_corpus_dataframe,
        LEXICOGRAMMATICAL_METRICS,
        TABLE_2_HEADERS,
    )

    print("[PROCESS] Normalizing descriptive statistics...")
    paragraph_narrative_descriptive_dataframe = normalize_descriptive_statistics_dataframe(
        load_csv_dataframe(input_filepaths["narrative_descriptive"]),
        GROUP_NAME_NARRATIVE,
        ANALYSIS_LEVEL_PARAGRAPH,
    )
    paragraph_lexical_descriptive_dataframe = normalize_descriptive_statistics_dataframe(
        load_csv_dataframe(input_filepaths["lexical_descriptive"]),
        GROUP_NAME_LEXICAL,
        ANALYSIS_LEVEL_PARAGRAPH,
    )
    chapter_descriptive_dataframe = normalize_descriptive_statistics_dataframe(
        load_csv_dataframe(input_filepaths["chapter_descriptive"]),
        "",
        ANALYSIS_LEVEL_CHAPTER,
    )

    print("[PROCESS] Normalizing Kruskal-Wallis results...")
    paragraph_narrative_kruskal_dataframe = normalize_kruskal_wallis_dataframe(
        load_csv_dataframe(input_filepaths["narrative_kruskal"]),
        GROUP_NAME_NARRATIVE,
        narrative_raw_dataframe,
        ANALYSIS_LEVEL_PARAGRAPH,
    )
    paragraph_lexical_kruskal_dataframe = normalize_kruskal_wallis_dataframe(
        load_csv_dataframe(input_filepaths["lexical_kruskal"]),
        GROUP_NAME_LEXICAL,
        lexicogrammatical_raw_dataframe,
        ANALYSIS_LEVEL_PARAGRAPH,
    )
    chapter_kruskal_dataframe = normalize_kruskal_wallis_dataframe(
        load_csv_dataframe(input_filepaths["chapter_kruskal"]),
        "",
        None,
        ANALYSIS_LEVEL_CHAPTER,
    )

    print("[PROCESS] Normalizing Mann-Whitney results...")
    paragraph_narrative_mann_whitney_dataframe = normalize_mann_whitney_dataframe(
        load_csv_dataframe(input_filepaths["narrative_mann_whitney"]),
        GROUP_NAME_NARRATIVE,
        narrative_raw_dataframe,
        ANALYSIS_LEVEL_PARAGRAPH,
    )
    paragraph_lexical_mann_whitney_dataframe = normalize_mann_whitney_dataframe(
        load_csv_dataframe(input_filepaths["lexical_mann_whitney"]),
        GROUP_NAME_LEXICAL,
        lexicogrammatical_raw_dataframe,
        ANALYSIS_LEVEL_PARAGRAPH,
    )
    chapter_mann_whitney_dataframe = normalize_mann_whitney_dataframe(
        load_csv_dataframe(input_filepaths["chapter_mann_whitney"]),
        "",
        None,
        ANALYSIS_LEVEL_CHAPTER,
    )

    print("[PROCESS] Normalizing correlation results...")
    paragraph_narrative_correlations_dataframe = normalize_correlations_dataframe(
        load_csv_dataframe(input_filepaths["narrative_correlations"]),
        GROUP_NAME_NARRATIVE,
        narrative_raw_dataframe,
        ANALYSIS_LEVEL_PARAGRAPH,
    )
    paragraph_lexical_correlations_dataframe = normalize_correlations_dataframe(
        load_csv_dataframe(input_filepaths["lexical_correlations"]),
        GROUP_NAME_LEXICAL,
        lexicogrammatical_raw_dataframe,
        ANALYSIS_LEVEL_PARAGRAPH,
    )
    chapter_correlations_dataframe = normalize_correlations_dataframe(
        load_csv_dataframe(input_filepaths["chapter_correlations"]),
        "",
        None,
        ANALYSIS_LEVEL_CHAPTER,
    )

    print("[PROCESS] Generating supplementary files...")
    supplementary_file_1_dataframe = generate_publication_supplementary_file_1(
        paragraph_narrative_descriptive_dataframe,
        paragraph_lexical_descriptive_dataframe,
        chapter_descriptive_dataframe,
    )
    supplementary_file_2_dataframe = generate_publication_supplementary_file_2(
        paragraph_narrative_kruskal_dataframe,
        paragraph_lexical_kruskal_dataframe,
        chapter_kruskal_dataframe,
    )
    supplementary_file_3_dataframe = generate_publication_supplementary_file_3(
        paragraph_narrative_mann_whitney_dataframe,
        paragraph_lexical_mann_whitney_dataframe,
        chapter_mann_whitney_dataframe,
    )
    supplementary_file_4_dataframe = generate_publication_supplementary_file_4(
        paragraph_narrative_correlations_dataframe,
        paragraph_lexical_correlations_dataframe,
        chapter_correlations_dataframe,
    )

    print("[PROCESS] Exporting publication files...")
    output_dataframes = {
        "table_1": table_1_dataframe,
        "table_2": table_2_dataframe,
        "s1": supplementary_file_1_dataframe,
        "s2": supplementary_file_2_dataframe,
        "s3": supplementary_file_3_dataframe,
        "s4": supplementary_file_4_dataframe,
    }
    for output_name, output_dataframe in output_dataframes.items():
        output_dataframe.to_csv(output_filepaths[output_name], index=False, encoding=CSV_ENCODING)

    print("[RESULT] Publication tables generated successfully.")
    for exported_path in output_filepaths.values():
        print(f"[RESULT] {exported_path}")


if __name__ == "__main__":
    run_pipeline()
