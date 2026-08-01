import os
import warnings
from itertools import combinations

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal, kurtosis, mannwhitneyu, skew, spearmanr
from statsmodels.stats.multitest import multipletests

PROJECT_DIRECTORY = os.path.abspath(os.environ.get("TEXTUAL_INDICATORS_PROJECT_DIRECTORY", "."))
INPUT_DIRECTORY = os.path.abspath(
    os.environ.get("TEXTUAL_INDICATORS_INPUT_DIRECTORY", os.path.join(PROJECT_DIRECTORY, "Textual_Indicators_output"))
)
OUTPUT_DIRECTORY = os.path.abspath(
    os.environ.get(
        "TEXTUAL_INDICATORS_ROBUSTNESS_OUTPUT_DIRECTORY",
        os.path.join(PROJECT_DIRECTORY, "Textual_Indicators_output", "chapter_level_robustness"),
    )
)

INPUT_NARRATIVE_SEMANTIC_FILEPATH = os.path.join(
    INPUT_DIRECTORY, "narrative-semantic_indicators", "01_Narrative-Semantic_Indicators_Raw_Paragraph_Indicators.csv"
)
INPUT_LEXICOGRAMMATICAL_FILEPATH = os.path.join(
    INPUT_DIRECTORY, "lexicogrammatical_indicators", "01a_Lexicogrammatical_Indicators_Raw_Paragraph_Indicators.csv"
)

OUTPUT_CHAPTER_INDICATORS_CSV = os.path.join(OUTPUT_DIRECTORY, "01_Chapter_Level_Indicator_Means.csv")
OUTPUT_DESCRIPTIVE_STATISTICS_CSV = os.path.join(OUTPUT_DIRECTORY, "02_Chapter_Level_Descriptive_Statistics.csv")
OUTPUT_CORRELATIONS_CSV = os.path.join(OUTPUT_DIRECTORY, "03_Chapter_Level_Spearman_Correlations.csv")
OUTPUT_KRUSKAL_WALLIS_CSV = os.path.join(OUTPUT_DIRECTORY, "04_Chapter_Level_Kruskal_Wallis_Tests.csv")
OUTPUT_MANN_WHITNEY_CSV = os.path.join(OUTPUT_DIRECTORY, "05_Chapter_Level_Mann_Whitney_Tests.csv")
OUTPUT_BOXPLOT_FIGURE = os.path.join(OUTPUT_DIRECTORY, "Fig1_Chapter_Level_Standardized_Indicator_Boxplots.png")
OUTPUT_BOXPLOT_SOURCE_DATA = os.path.join(OUTPUT_DIRECTORY, "Fig1_Chapter_Level_Standardized_Indicator_Boxplots_Source_Data.csv")

RANDOM_SEED = 42
SIGNIFICANCE_LEVEL = 0.05
FDR_CORRECTION_METHOD = "fdr_bh"
CSV_ENCODING = "utf-8"
CSV_FLOAT_FORMAT = "%.12g"
FIGURE_DPI = 300
FIGURE_SIZE = (13, 10)

NOVEL_PART_ORDER = ["The Window", "Time Passes", "The Lighthouse"]
MERGE_KEY_COLUMNS = ["part", "chapter_id", "para_idx"]
CHAPTER_KEY_COLUMNS = ["part", "chapter_id"]
NARRATIVE_SEMANTIC_METRICS = ["event_density", "perspective_shift", "psychological_density", "temporal_density"]
LEXICOGRAMMATICAL_METRICS = ["pos_ratio_adj_adv_to_verb", "avg_dependency_depth", "repeated_words_per_100_tokens", "parallel_ratio"]
TARGET_METRICS = NARRATIVE_SEMANTIC_METRICS + LEXICOGRAMMATICAL_METRICS
ANALYSIS_FAMILIES = {"Narrative-semantic": NARRATIVE_SEMANTIC_METRICS, "Lexicogrammatical": LEXICOGRAMMATICAL_METRICS}
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
PART_COLOR_PALETTE = {"The Window": "#3B6FB6", "Time Passes": "#D98C3F", "The Lighthouse": "#4A9B75"}

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 13
sns.set_theme(style="whitegrid")
np.random.seed(RANDOM_SEED)


def load_dataframe_with_required_columns(filepath, required_columns):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERROR] Input file not found: {filepath}")

    dataframe = pd.read_csv(filepath)
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"[ERROR] Missing required columns in {filepath}: {missing_columns}")

    return dataframe[required_columns].copy()


def validate_key_columns(dataframe, dataframe_name):
    if dataframe[MERGE_KEY_COLUMNS].isna().any().any():
        raise ValueError(f"[ERROR] Missing merge keys found in {dataframe_name}.")

    duplicate_mask = dataframe.duplicated(MERGE_KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        raise ValueError(f"[ERROR] Duplicate merge keys found in {dataframe_name}: {duplicate_count} rows.")

    invalid_parts = sorted(set(dataframe["part"].astype(str).unique()) - set(NOVEL_PART_ORDER))
    if invalid_parts:
        raise ValueError(f"[ERROR] Unexpected part labels found in {dataframe_name}: {invalid_parts}")


def load_and_merge_paragraph_indicators():
    narrative_semantic_columns = MERGE_KEY_COLUMNS + NARRATIVE_SEMANTIC_METRICS
    lexicogrammatical_columns = MERGE_KEY_COLUMNS + LEXICOGRAMMATICAL_METRICS

    narrative_semantic_dataframe = load_dataframe_with_required_columns(INPUT_NARRATIVE_SEMANTIC_FILEPATH, narrative_semantic_columns)
    lexicogrammatical_dataframe = load_dataframe_with_required_columns(INPUT_LEXICOGRAMMATICAL_FILEPATH, lexicogrammatical_columns)

    validate_key_columns(narrative_semantic_dataframe, "narrative-semantic paragraph indicators")
    validate_key_columns(lexicogrammatical_dataframe, "lexicogrammatical paragraph indicators")

    merged_dataframe = pd.merge(
        narrative_semantic_dataframe, lexicogrammatical_dataframe, on=MERGE_KEY_COLUMNS, how="outer", validate="one_to_one", indicator=True
    )

    unmatched_dataframe = merged_dataframe[merged_dataframe["_merge"] != "both"]
    if not unmatched_dataframe.empty:
        unmatched_counts = unmatched_dataframe["_merge"].value_counts().to_dict()
        raise ValueError(f"[ERROR] Paragraph rows do not match one-to-one between input files: {unmatched_counts}")

    merged_dataframe = merged_dataframe.drop(columns="_merge")
    merged_dataframe[TARGET_METRICS] = merged_dataframe[TARGET_METRICS].apply(pd.to_numeric, errors="coerce")

    missing_values = merged_dataframe[TARGET_METRICS].isna().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        raise ValueError(f"[ERROR] Missing or nonnumeric indicator values found: {missing_values.to_dict()}")

    finite_values_mask = np.isfinite(merged_dataframe[TARGET_METRICS].to_numpy(dtype=float))
    if not finite_values_mask.all():
        raise ValueError("[ERROR] Nonfinite indicator values found.")

    merged_dataframe["part"] = pd.Categorical(merged_dataframe["part"], categories=NOVEL_PART_ORDER, ordered=True)
    merged_dataframe = merged_dataframe.sort_values(MERGE_KEY_COLUMNS).reset_index(drop=True)

    return merged_dataframe


def aggregate_chapter_level_indicators(paragraph_dataframe):
    chapter_dataframe = paragraph_dataframe.groupby(CHAPTER_KEY_COLUMNS, observed=True)[TARGET_METRICS].mean().reset_index()

    chapter_dataframe["paragraph_count"] = paragraph_dataframe.groupby(CHAPTER_KEY_COLUMNS, observed=True).size().to_numpy()
    chapter_dataframe["part"] = pd.Categorical(chapter_dataframe["part"], categories=NOVEL_PART_ORDER, ordered=True)
    chapter_dataframe = chapter_dataframe.sort_values(CHAPTER_KEY_COLUMNS).reset_index(drop=True)
    chapter_dataframe["chapter_index"] = np.arange(1, len(chapter_dataframe) + 1)

    chapter_counts = chapter_dataframe.groupby("part", observed=True).size()
    missing_parts = [part for part in NOVEL_PART_ORDER if int(chapter_counts.get(part, 0)) == 0]
    if missing_parts:
        raise ValueError(f"[ERROR] No chapter-level observations found for: {missing_parts}")

    ordered_columns = ["chapter_index", "part", "chapter_id", "paragraph_count"] + TARGET_METRICS
    return chapter_dataframe[ordered_columns]


def calculate_rank_biserial_correlation(u_statistic, sample_size_1, sample_size_2):
    denominator = sample_size_1 * sample_size_2
    if denominator <= 0:
        return np.nan
    return (2.0 * float(u_statistic) / denominator) - 1.0


def calculate_cohens_d(sample_1, sample_2):
    array_1 = np.asarray(sample_1, dtype=float)
    array_2 = np.asarray(sample_2, dtype=float)
    array_1 = array_1[np.isfinite(array_1)]
    array_2 = array_2[np.isfinite(array_2)]

    sample_size_1 = len(array_1)
    sample_size_2 = len(array_2)
    if sample_size_1 < 2 or sample_size_2 < 2:
        return np.nan

    pooled_variance_denominator = sample_size_1 + sample_size_2 - 2
    if pooled_variance_denominator <= 0:
        return np.nan

    pooled_variance = (
        (sample_size_1 - 1) * np.var(array_1, ddof=1) + (sample_size_2 - 1) * np.var(array_2, ddof=1)
    ) / pooled_variance_denominator

    if not np.isfinite(pooled_variance) or pooled_variance <= 0:
        return np.nan

    return (np.mean(array_1) - np.mean(array_2)) / np.sqrt(pooled_variance)


def calculate_epsilon_squared(h_statistic, total_sample_size, group_count):
    denominator = total_sample_size - group_count
    if denominator <= 0:
        return np.nan
    epsilon_squared = (float(h_statistic) - group_count + 1) / denominator
    return max(epsilon_squared, 0.0)


def build_descriptive_statistics_dataframe(chapter_dataframe):
    statistics_rows = []

    for analysis_family, metrics in ANALYSIS_FAMILIES.items():
        for metric in metrics:
            analysis_scopes = [("Overall", chapter_dataframe[metric].dropna())]
            analysis_scopes.extend(
                (part, chapter_dataframe.loc[chapter_dataframe["part"] == part, metric].dropna()) for part in NOVEL_PART_ORDER
            )

            for scope_name, metric_values in analysis_scopes:
                sample_size = len(metric_values)
                statistics_rows.append(
                    {
                        "Indicator Group": analysis_family,
                        "Indicator": METRIC_DISPLAY_LABELS[metric],
                        "Scope": scope_name,
                        "Chapter Count": sample_size,
                        "Mean": metric_values.mean() if sample_size > 0 else np.nan,
                        "SD": metric_values.std(ddof=1) if sample_size > 1 else np.nan,
                        "Median": metric_values.median() if sample_size > 0 else np.nan,
                        "IQR": (metric_values.quantile(0.75) - metric_values.quantile(0.25)) if sample_size > 0 else np.nan,
                        "Skewness": skew(metric_values, nan_policy="omit") if sample_size > 2 else np.nan,
                        "Kurtosis": kurtosis(metric_values, nan_policy="omit") if sample_size > 3 else np.nan,
                    }
                )

    return pd.DataFrame(statistics_rows)


def build_correlation_results_dataframe(chapter_dataframe):
    correlation_rows = []

    for analysis_family, metrics in ANALYSIS_FAMILIES.items():
        family_rows = []

        for metric_1, metric_2 in combinations(metrics, 2):
            paired_dataframe = chapter_dataframe[[metric_1, metric_2]].dropna()

            if len(paired_dataframe) < 3:
                correlation_coefficient = np.nan
                unadjusted_p_value = np.nan
            else:
                correlation_result = spearmanr(paired_dataframe[metric_1], paired_dataframe[metric_2])
                correlation_coefficient = correlation_result.statistic
                unadjusted_p_value = correlation_result.pvalue

            family_rows.append(
                {
                    "Indicator Group": analysis_family,
                    "Indicator 1": METRIC_DISPLAY_LABELS[metric_1],
                    "Indicator 2": METRIC_DISPLAY_LABELS[metric_2],
                    "Chapter Count": len(paired_dataframe),
                    "Spearman's rho": correlation_coefficient,
                    "Unadjusted p-value": unadjusted_p_value,
                }
            )

        family_dataframe = pd.DataFrame(family_rows)
        family_dataframe["Adjusted p-value (BH FDR)"] = np.nan
        finite_p_value_mask = np.isfinite(family_dataframe["Unadjusted p-value"])

        if finite_p_value_mask.any():
            adjusted_p_values = multipletests(
                family_dataframe.loc[finite_p_value_mask, "Unadjusted p-value"], method=FDR_CORRECTION_METHOD
            )[1]
            family_dataframe.loc[finite_p_value_mask, "Adjusted p-value (BH FDR)"] = adjusted_p_values

        family_dataframe["Significant after FDR correction"] = family_dataframe["Adjusted p-value (BH FDR)"] < SIGNIFICANCE_LEVEL
        correlation_rows.append(family_dataframe)

    return pd.concat(correlation_rows, ignore_index=True)


def build_kruskal_wallis_results_dataframe(chapter_dataframe):
    family_results = []
    group_count = len(NOVEL_PART_ORDER)

    for analysis_family, metrics in ANALYSIS_FAMILIES.items():
        test_rows = []
        for metric in metrics:
            metric_groups = [chapter_dataframe.loc[chapter_dataframe["part"] == part, metric].dropna() for part in NOVEL_PART_ORDER]
            group_sizes = [len(metric_group) for metric_group in metric_groups]

            if any(group_size == 0 for group_size in group_sizes):
                h_statistic = np.nan
                p_value = np.nan
                epsilon_squared = np.nan
            else:
                test_result = kruskal(*metric_groups)
                h_statistic = test_result.statistic
                p_value = test_result.pvalue
                epsilon_squared = calculate_epsilon_squared(h_statistic, sum(group_sizes), group_count)

            test_rows.append(
                {
                    "Indicator Group": analysis_family,
                    "Indicator": METRIC_DISPLAY_LABELS[metric],
                    "H Statistic": h_statistic,
                    "df": group_count - 1,
                    "n (The Window)": group_sizes[0],
                    "n (Time Passes)": group_sizes[1],
                    "n (The Lighthouse)": group_sizes[2],
                    "Unadjusted p-value": p_value,
                    "Epsilon Squared": epsilon_squared,
                }
            )

        family_dataframe = pd.DataFrame(test_rows)
        family_dataframe["Adjusted p-value (BH FDR)"] = np.nan
        finite_p_value_mask = np.isfinite(family_dataframe["Unadjusted p-value"])
        if finite_p_value_mask.any():
            adjusted_p_values = multipletests(
                family_dataframe.loc[finite_p_value_mask, "Unadjusted p-value"], method=FDR_CORRECTION_METHOD
            )[1]
            family_dataframe.loc[finite_p_value_mask, "Adjusted p-value (BH FDR)"] = adjusted_p_values
        family_dataframe["Significant after FDR correction"] = family_dataframe["Adjusted p-value (BH FDR)"] < SIGNIFICANCE_LEVEL
        family_results.append(family_dataframe)

    combined_dataframe = pd.concat(family_results, ignore_index=True)
    ordered_columns = [
        "Indicator Group",
        "Indicator",
        "H Statistic",
        "df",
        "n (The Window)",
        "n (Time Passes)",
        "n (The Lighthouse)",
        "Unadjusted p-value",
        "Adjusted p-value (BH FDR)",
        "Significant after FDR correction",
        "Epsilon Squared",
    ]
    return combined_dataframe[ordered_columns]


def build_mann_whitney_results_dataframe(chapter_dataframe):
    family_results = []

    for analysis_family, metrics in ANALYSIS_FAMILIES.items():
        comparison_rows = []

        for metric in metrics:
            for part_1, part_2 in combinations(NOVEL_PART_ORDER, 2):
                sample_1 = chapter_dataframe.loc[chapter_dataframe["part"] == part_1, metric].dropna()
                sample_2 = chapter_dataframe.loc[chapter_dataframe["part"] == part_2, metric].dropna()

                if len(sample_1) == 0 or len(sample_2) == 0:
                    u_statistic = np.nan
                    unadjusted_p_value = np.nan
                else:
                    test_result = mannwhitneyu(sample_1, sample_2, alternative="two-sided", method="asymptotic")
                    u_statistic = test_result.statistic
                    unadjusted_p_value = test_result.pvalue

                comparison_rows.append(
                    {
                        "Indicator Group": analysis_family,
                        "Indicator": METRIC_DISPLAY_LABELS[metric],
                        "Comparison": f"{part_1} vs {part_2}",
                        "n1": len(sample_1),
                        "n2": len(sample_2),
                        "Mann-Whitney U": u_statistic,
                        "Unadjusted p-value": unadjusted_p_value,
                        "Rank-biserial correlation": (
                            calculate_rank_biserial_correlation(u_statistic, len(sample_1), len(sample_2))
                            if np.isfinite(u_statistic)
                            else np.nan
                        ),
                        "Cohen's d": calculate_cohens_d(sample_1, sample_2),
                    }
                )

        family_dataframe = pd.DataFrame(comparison_rows)
        family_dataframe["Adjusted p-value (BH FDR)"] = np.nan
        finite_p_value_mask = np.isfinite(family_dataframe["Unadjusted p-value"])

        if finite_p_value_mask.any():
            adjusted_p_values = multipletests(
                family_dataframe.loc[finite_p_value_mask, "Unadjusted p-value"], method=FDR_CORRECTION_METHOD
            )[1]
            family_dataframe.loc[finite_p_value_mask, "Adjusted p-value (BH FDR)"] = adjusted_p_values

        family_dataframe["Significant after FDR correction"] = family_dataframe["Adjusted p-value (BH FDR)"] < SIGNIFICANCE_LEVEL
        family_results.append(family_dataframe)

    combined_dataframe = pd.concat(family_results, ignore_index=True)
    ordered_columns = [
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
    return combined_dataframe[ordered_columns]


def build_standardized_boxplot_source_dataframe(chapter_dataframe):
    standardized_dataframe = chapter_dataframe[["chapter_index", "part", "chapter_id"] + TARGET_METRICS].copy()

    for metric in TARGET_METRICS:
        metric_standard_deviation = standardized_dataframe[metric].std(ddof=1)
        if not np.isfinite(metric_standard_deviation) or metric_standard_deviation == 0:
            standardized_dataframe[metric] = 0.0
        else:
            standardized_dataframe[metric] = (
                standardized_dataframe[metric] - standardized_dataframe[metric].mean()
            ) / metric_standard_deviation

    source_dataframe = standardized_dataframe.melt(
        id_vars=["chapter_index", "part", "chapter_id"],
        value_vars=TARGET_METRICS,
        var_name="indicator_name",
        value_name="standardized_chapter_mean",
    )
    source_dataframe["Indicator"] = source_dataframe["indicator_name"].map(METRIC_DISPLAY_LABELS)
    source_dataframe["part"] = pd.Categorical(source_dataframe["part"], categories=NOVEL_PART_ORDER, ordered=True)

    indicator_order = [METRIC_DISPLAY_LABELS[metric] for metric in TARGET_METRICS]
    source_dataframe["Indicator"] = pd.Categorical(source_dataframe["Indicator"], categories=indicator_order, ordered=True)
    source_dataframe = source_dataframe.sort_values(["Indicator", "part", "chapter_id"]).reset_index(drop=True)

    return source_dataframe[["chapter_index", "part", "chapter_id", "indicator_name", "Indicator", "standardized_chapter_mean"]]


def render_standardized_indicator_boxplots(source_dataframe, output_filepath):
    indicator_order = [METRIC_DISPLAY_LABELS[metric] for metric in TARGET_METRICS]

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    sns.boxplot(
        data=source_dataframe,
        x="standardized_chapter_mean",
        y="Indicator",
        hue="part",
        order=indicator_order,
        hue_order=NOVEL_PART_ORDER,
        palette=PART_COLOR_PALETTE,
        showfliers=False,
        linewidth=1.1,
        ax=axis,
    )
    sns.stripplot(
        data=source_dataframe,
        x="standardized_chapter_mean",
        y="Indicator",
        hue="part",
        order=indicator_order,
        hue_order=NOVEL_PART_ORDER,
        palette=PART_COLOR_PALETTE,
        dodge=True,
        jitter=0.14,
        size=4,
        alpha=0.65,
        linewidth=0,
        legend=False,
        ax=axis,
    )
    axis.axvline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Standardized Chapter Mean")
    axis.set_ylabel("")
    axis.legend(title="Novel Part", loc="lower right", frameon=True)
    figure.tight_layout()
    figure.savefig(output_filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def save_dataframe_to_numeric_csv(dataframe, filepath):
    dataframe.to_csv(filepath, index=False, encoding=CSV_ENCODING, float_format=CSV_FLOAT_FORMAT)


def run_pipeline():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    print("[PROCESS] Loading and validating paragraph-level indicators...")
    paragraph_dataframe = load_and_merge_paragraph_indicators()

    print("[PROCESS] Aggregating chapter-level indicator means...")
    chapter_dataframe = aggregate_chapter_level_indicators(paragraph_dataframe)
    save_dataframe_to_numeric_csv(chapter_dataframe, OUTPUT_CHAPTER_INDICATORS_CSV)

    print("[PROCESS] Computing chapter-level statistical analyses...")
    descriptive_statistics_dataframe = build_descriptive_statistics_dataframe(chapter_dataframe)
    correlation_results_dataframe = build_correlation_results_dataframe(chapter_dataframe)
    kruskal_wallis_results_dataframe = build_kruskal_wallis_results_dataframe(chapter_dataframe)
    mann_whitney_results_dataframe = build_mann_whitney_results_dataframe(chapter_dataframe)

    save_dataframe_to_numeric_csv(descriptive_statistics_dataframe, OUTPUT_DESCRIPTIVE_STATISTICS_CSV)
    save_dataframe_to_numeric_csv(correlation_results_dataframe, OUTPUT_CORRELATIONS_CSV)
    save_dataframe_to_numeric_csv(kruskal_wallis_results_dataframe, OUTPUT_KRUSKAL_WALLIS_CSV)
    save_dataframe_to_numeric_csv(mann_whitney_results_dataframe, OUTPUT_MANN_WHITNEY_CSV)

    print("[PROCESS] Rendering chapter-level visualization...")
    boxplot_source_dataframe = build_standardized_boxplot_source_dataframe(chapter_dataframe)
    save_dataframe_to_numeric_csv(boxplot_source_dataframe, OUTPUT_BOXPLOT_SOURCE_DATA)
    render_standardized_indicator_boxplots(boxplot_source_dataframe, OUTPUT_BOXPLOT_FIGURE)

    significant_omnibus_count = int(kruskal_wallis_results_dataframe["Significant after FDR correction"].sum())
    significant_pairwise_count = int(mann_whitney_results_dataframe["Significant after FDR correction"].sum())
    print(
        "[RESULT] Analysis completed successfully. "
        f"Significant FDR-adjusted omnibus tests: "
        f"{significant_omnibus_count}. "
        f"Significant FDR-adjusted pairwise tests: "
        f"{significant_pairwise_count}."
    )
    print(f"[RESULT] Outputs saved to: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    run_pipeline()
