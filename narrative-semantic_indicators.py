import os
import re
import unicodedata
import warnings
from itertools import combinations

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from nltk.corpus import wordnet as nltk_wordnet
from scipy.spatial.distance import cosine
from scipy.stats import kruskal, kurtosis, mannwhitneyu, skew, spearmanr
from sentence_transformers import SentenceTransformer
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 14
sns.set_theme(style="whitegrid", palette="viridis")

INPUT_FILEPATH = "Textual_Indicators_output/text_segmentation/to_the_lighthouse_paragraphs.csv"
REFERENCE_CORPUS_FILEPATH = "Textual_Indicators_output/reference_corpus/reference_corpus_textual_indicators.csv"
OUTPUT_DIRECTORY = "Textual_Indicators_output/narrative-semantic_indicators"

OUTPUT_RAW_INDICATORS_CSV = os.path.join(OUTPUT_DIRECTORY, "01_Narrative-Semantic_Indicators_Raw_Paragraph_Indicators.csv")
OUTPUT_DESCRIPTIVE_STATS_CSV = os.path.join(OUTPUT_DIRECTORY, "02_Narrative-Semantic_Indicators_Descriptive_Statistics.csv")
OUTPUT_CORRELATIONS_CSV = os.path.join(OUTPUT_DIRECTORY, "03_Narrative-Semantic_Indicators_Spearman_Correlations.csv")
OUTPUT_KRUSKAL_WALLIS_CSV = os.path.join(OUTPUT_DIRECTORY, "04_Narrative-Semantic_Indicators_Kruskal_Wallis_Tests.csv")
OUTPUT_MANN_WHITNEY_CSV = os.path.join(OUTPUT_DIRECTORY, "05_Narrative-Semantic_Indicators_Mann_Whitney_Tests.csv")
OUTPUT_BOXPLOT_FIGURE = os.path.join(OUTPUT_DIRECTORY, "Fig1_Narrative-Semantic_Indicators_Boxplots.png")
OUTPUT_BOXPLOT_DATA = os.path.join(OUTPUT_DIRECTORY, "Fig1_Narrative-Semantic_Indicators_Source_Data_Boxplots.csv")
OUTPUT_TRENDS_FIGURE = os.path.join(OUTPUT_DIRECTORY, "Fig2_Narrative-Semantic_Indicators_Chapter_Trends.png")
OUTPUT_TRENDS_DATA = os.path.join(OUTPUT_DIRECTORY, "Fig2_Narrative-Semantic_Indicators_Source_Data_Chapter_Trends.csv")

SPACY_MODEL_NAME = "en_core_web_sm"
SPACY_TAGGER_DISABLED_COMPONENTS = ["ner", "parser"]
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
CSV_ENCODING = "utf-8"
CSV_FLOAT_FORMAT = "%.12g"
SIGNIFICANCE_LEVEL = 0.05
FDR_CORRECTION_METHOD = "fdr_bh"
SMOOTHING_WINDOW = 3
FIGURE_DPI = 300
BOXPLOT_FIGURE_SIZE = (14, 10)
TRENDS_FIGURE_SIZE = (16, 12)

ANALYSIS_GROUP_NAME = "Narrative-semantic"
NOVEL_PART_ORDER = ["The Window", "Time Passes", "The Lighthouse"]
TARGET_METRICS = [
    "event_density",
    "perspective_shift",
    "psychological_density",
    "temporal_density",
]
METRIC_DISPLAY_LABELS = {
    "event_density": "Event Density",
    "perspective_shift": "Perspective Shift",
    "psychological_density": "Psychological Density",
    "temporal_density": "Temporal Density",
}

PSYCHOLOGICAL_BASE_WORDS = [
    "mind",
    "thought",
    "feeling",
    "memory",
    "introspection",
    "consciousness",
    "awareness",
    "soul",
    "emotion",
    "perception",
    "imagination",
    "reflection",
    "contemplation",
    "intellect",
    "mood",
]
TEMPORAL_BASE_WORDS = [
    "time",
    "year",
    "season",
    "past",
    "future",
    "day",
    "night",
    "moment",
    "hour",
    "age",
    "minute",
    "century",
    "eternity",
    "duration",
    "instant",
    "clock",
    "afternoon",
    "morning",
]


def build_synonym_lexicon(base_words):
    synonyms = set()
    for word in base_words:
        synsets = nltk_wordnet.synsets(word)
        if synsets:
            for lemma in synsets[0].lemma_names():
                synonyms.add(lemma.lower())
    return synonyms


def clean_paragraph_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\n\r\t]", " ", text)
    text = re.sub(r"[;—]+", "; ", text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def save_dataframe_to_numeric_csv(dataframe, filepath):
    dataframe.to_csv(filepath, index=False, encoding=CSV_ENCODING, float_format=CSV_FLOAT_FORMAT)


def extract_narrative_features(
    raw_text,
    previous_embedding,
    spacy_full_model,
    spacy_tagger_model,
    embedding_model,
    psychological_synonyms,
    temporal_synonyms,
):
    cleaned_text = clean_paragraph_text(raw_text)
    if not cleaned_text:
        return 0.0, 0.0, 0.0, 0.0, 0.0, previous_embedding

    spacy_document = spacy_full_model(cleaned_text)
    sentences = list(spacy_document.sents)
    num_sentences = max(len(sentences), 1)
    num_words = max(sum(1 for token in spacy_document if token.is_alpha), 1)

    identified_events = [
        sent.text
        for sent in sentences
        if any(token.pos_ == "VERB" and token.dep_ in ["ROOT", "xcomp", "ccomp", "advcl"] for token in sent)
        and any(ent.label_ in ["PERSON", "ORG", "GPE", "LOC"] for ent in sent.ents)
    ]
    event_density = len(identified_events) / num_sentences

    semantic_distance = 0.0
    pronoun_transition_score = 0.0
    current_text_embedding = embedding_model.encode(cleaned_text)

    if previous_embedding is not None:
        distance_metric = cosine(current_text_embedding, previous_embedding)
        semantic_distance = 0.0 if pd.isna(distance_metric) else float(distance_metric)

    for sent in sentences:
        text_without_quotes = re.sub(r'["“”].*?["“”]|(?<!\w)[\'‘].*?[\'’](?!\w)', "", sent.text)
        sentence_doc = spacy_tagger_model(text_without_quotes)
        pronouns = [token.text.lower() for token in sentence_doc if token.pos_ == "PRON"]
        if len(pronouns) > 1:
            pronoun_transition_score += (
                sum(1 for index in range(1, len(pronouns)) if pronouns[index] != pronouns[index - 1]) / num_sentences
            )

    psychological_density = sum(1 for token in spacy_document if token.lemma_.lower() in psychological_synonyms) / num_words
    temporal_density = sum(1 for token in spacy_document if token.lemma_.lower() in temporal_synonyms) / num_words

    return (
        event_density,
        semantic_distance,
        pronoun_transition_score,
        psychological_density,
        temporal_density,
        current_text_embedding,
    )


def calculate_rank_biserial_correlation(u_statistic, sample_size_1, sample_size_2):
    if sample_size_1 <= 0 or sample_size_2 <= 0:
        return np.nan
    return (2.0 * float(u_statistic) / (sample_size_1 * sample_size_2)) - 1.0


def calculate_cohens_d(array_1, array_2):
    array_1 = np.asarray(array_1, dtype=float)
    array_2 = np.asarray(array_2, dtype=float)
    array_1 = array_1[np.isfinite(array_1)]
    array_2 = array_2[np.isfinite(array_2)]

    length_1, length_2 = len(array_1), len(array_2)
    if length_1 < 2 or length_2 < 2:
        return np.nan

    pooled_variance = ((length_1 - 1) * np.var(array_1, ddof=1) + (length_2 - 1) * np.var(array_2, ddof=1)) / (length_1 + length_2 - 2)
    if not np.isfinite(pooled_variance) or pooled_variance <= 0:
        return np.nan

    return (np.mean(array_1) - np.mean(array_2)) / np.sqrt(pooled_variance)


def build_descriptive_statistics_dataframe(segmentation_dataframe):
    statistics_rows = []
    for metric in TARGET_METRICS:
        analysis_scopes = [("Overall", segmentation_dataframe[metric].dropna())]
        analysis_scopes.extend(
            (part, segmentation_dataframe.loc[segmentation_dataframe["part"] == part, metric].dropna()) for part in NOVEL_PART_ORDER
        )

        for scope_name, metric_values in analysis_scopes:
            statistics_rows.append(
                {
                    "Indicator Group": ANALYSIS_GROUP_NAME,
                    "Indicator": METRIC_DISPLAY_LABELS[metric],
                    "Scope": scope_name,
                    "Count": len(metric_values),
                    "Mean": metric_values.mean(),
                    "SD": metric_values.std(ddof=1),
                    "Median": metric_values.median(),
                    "IQR": metric_values.quantile(0.75) - metric_values.quantile(0.25),
                    "Skewness": skew(metric_values, nan_policy="omit"),
                    "Kurtosis": kurtosis(metric_values, nan_policy="omit"),
                }
            )
    return pd.DataFrame(statistics_rows)


def build_correlation_results_dataframe(segmentation_dataframe):
    correlation_rows = []
    for metric_1, metric_2 in combinations(TARGET_METRICS, 2):
        paired_data = segmentation_dataframe[[metric_1, metric_2]].dropna()
        spearman_rho, unadjusted_p_value = spearmanr(paired_data[metric_1], paired_data[metric_2])
        correlation_rows.append(
            {
                "Indicator Group": ANALYSIS_GROUP_NAME,
                "Indicator 1": METRIC_DISPLAY_LABELS[metric_1],
                "Indicator 2": METRIC_DISPLAY_LABELS[metric_2],
                "N": len(paired_data),
                "Spearman's rho": spearman_rho,
                "Unadjusted p-value": unadjusted_p_value,
            }
        )

    correlation_dataframe = pd.DataFrame(correlation_rows)
    correlation_dataframe["Adjusted p-value (BH FDR)"] = np.nan
    finite_p_values_mask = np.isfinite(correlation_dataframe["Unadjusted p-value"])

    if finite_p_values_mask.any():
        correlation_dataframe.loc[finite_p_values_mask, "Adjusted p-value (BH FDR)"] = multipletests(
            correlation_dataframe.loc[finite_p_values_mask, "Unadjusted p-value"], method=FDR_CORRECTION_METHOD
        )[1]

    correlation_dataframe["Significant after FDR correction"] = correlation_dataframe["Adjusted p-value (BH FDR)"] < SIGNIFICANCE_LEVEL
    return correlation_dataframe


def build_kruskal_wallis_results_dataframe(segmentation_dataframe):
    kruskal_rows = []
    group_sizes = {part: int(segmentation_dataframe.loc[segmentation_dataframe["part"] == part].shape[0]) for part in NOVEL_PART_ORDER}

    for metric in TARGET_METRICS:
        metric_groups = [segmentation_dataframe.loc[segmentation_dataframe["part"] == part, metric].dropna() for part in NOVEL_PART_ORDER]
        kruskal_result = kruskal(*metric_groups)
        kruskal_rows.append(
            {
                "Indicator Group": ANALYSIS_GROUP_NAME,
                "Indicator": METRIC_DISPLAY_LABELS[metric],
                "H Statistic": kruskal_result.statistic,
                "df": len(NOVEL_PART_ORDER) - 1,
                "n (The Window)": group_sizes["The Window"],
                "n (Time Passes)": group_sizes["Time Passes"],
                "n (The Lighthouse)": group_sizes["The Lighthouse"],
                "Unadjusted p-value": kruskal_result.pvalue,
            }
        )
    kruskal_dataframe = pd.DataFrame(kruskal_rows)
    kruskal_dataframe["Adjusted p-value (BH FDR)"] = multipletests(
        kruskal_dataframe["Unadjusted p-value"],
        method=FDR_CORRECTION_METHOD,
    )[1]
    kruskal_dataframe["Significant after FDR correction"] = kruskal_dataframe["Adjusted p-value (BH FDR)"] < SIGNIFICANCE_LEVEL
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
    ]
    return kruskal_dataframe[ordered_columns]


def build_mann_whitney_results_dataframe(segmentation_dataframe):
    mann_whitney_rows = []
    for metric in TARGET_METRICS:
        for part_1, part_2 in combinations(NOVEL_PART_ORDER, 2):
            sample_1 = segmentation_dataframe.loc[segmentation_dataframe["part"] == part_1, metric].dropna()
            sample_2 = segmentation_dataframe.loc[segmentation_dataframe["part"] == part_2, metric].dropna()
            mann_whitney_result = mannwhitneyu(
                sample_1,
                sample_2,
                alternative="two-sided",
                method="asymptotic",
            )
            mann_whitney_rows.append(
                {
                    "Indicator Group": ANALYSIS_GROUP_NAME,
                    "Indicator": METRIC_DISPLAY_LABELS[metric],
                    "Comparison": f"{part_1} vs {part_2}",
                    "n1": len(sample_1),
                    "n2": len(sample_2),
                    "Mann-Whitney U": mann_whitney_result.statistic,
                    "Unadjusted p-value": mann_whitney_result.pvalue,
                    "Rank-biserial correlation": calculate_rank_biserial_correlation(
                        mann_whitney_result.statistic, len(sample_1), len(sample_2)
                    ),
                    "Cohen's d": calculate_cohens_d(sample_1, sample_2),
                }
            )

    mann_whitney_dataframe = pd.DataFrame(mann_whitney_rows)
    mann_whitney_dataframe["Adjusted p-value (BH FDR)"] = multipletests(
        mann_whitney_dataframe["Unadjusted p-value"], method=FDR_CORRECTION_METHOD
    )[1]
    mann_whitney_dataframe["Significant after FDR correction"] = mann_whitney_dataframe["Adjusted p-value (BH FDR)"] < SIGNIFICANCE_LEVEL

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
    return mann_whitney_dataframe[ordered_columns]


def apply_standardization_and_smoothing(chapter_aggregated_dataframe):
    for metric in TARGET_METRICS:
        standard_deviation = chapter_aggregated_dataframe[metric].std(ddof=1)
        standardized_column_name = f"standardized_{metric}"
        smoothed_column_name = f"smoothed_{metric}"

        if not np.isfinite(standard_deviation) or standard_deviation == 0:
            chapter_aggregated_dataframe[standardized_column_name] = 0.0
        else:
            chapter_aggregated_dataframe[standardized_column_name] = (
                chapter_aggregated_dataframe[metric] - chapter_aggregated_dataframe[metric].mean()
            ) / standard_deviation

        chapter_aggregated_dataframe[smoothed_column_name] = chapter_aggregated_dataframe.groupby("part", observed=True)[
            standardized_column_name
        ].transform(lambda grouped_series: grouped_series.rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1).mean())
    return chapter_aggregated_dataframe


def run_pipeline():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    print("[PROCESS] Initializing models...")

    spacy_full_model = spacy.load(SPACY_MODEL_NAME)
    spacy_tagger_model = spacy.load(SPACY_MODEL_NAME, disable=SPACY_TAGGER_DISABLED_COMPONENTS)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    psychological_synonyms = build_synonym_lexicon(PSYCHOLOGICAL_BASE_WORDS)
    temporal_synonyms = build_synonym_lexicon(TEMPORAL_BASE_WORDS)

    print("[PROCESS] Formatting dataset...")
    segmentation_dataframe = pd.read_csv(INPUT_FILEPATH)
    required_data_columns = ["part", "chapter_id", "para_idx", "text"]
    missing_data_columns = [col for col in required_data_columns if col not in segmentation_dataframe.columns]

    if missing_data_columns:
        raise ValueError(f"[ERROR] Missing required columns in {INPUT_FILEPATH}: {missing_data_columns}")

    segmentation_dataframe = segmentation_dataframe[required_data_columns].copy()
    if segmentation_dataframe["text"].isna().any() or segmentation_dataframe["text"].astype(str).str.strip().eq("").any():
        raise ValueError("[ERROR] Empty or missing paragraph text found.")

    segmentation_dataframe["part"] = pd.Categorical(segmentation_dataframe["part"], categories=NOVEL_PART_ORDER, ordered=True)
    if segmentation_dataframe["part"].isna().any():
        invalid_part_labels = sorted(set(segmentation_dataframe["part"].dropna()) - set(NOVEL_PART_ORDER))
        raise ValueError(f"[ERROR] Unexpected part labels: {invalid_part_labels}")

    segmentation_dataframe = segmentation_dataframe.sort_values(["part", "chapter_id", "para_idx"]).reset_index(drop=True)

    print("[PROCESS] Extracting narrative-semantic indicators...")
    extracted_features_records = []
    previous_paragraph_embedding = None

    for _, paragraph_row in tqdm(segmentation_dataframe.iterrows(), total=len(segmentation_dataframe), desc="Extracting"):
        (
            computed_event_density,
            computed_semantic_distance,
            computed_pronoun_transition,
            computed_psychological_density,
            computed_temporal_density,
            current_paragraph_embedding,
        ) = extract_narrative_features(
            raw_text=str(paragraph_row["text"]),
            previous_embedding=previous_paragraph_embedding,
            spacy_full_model=spacy_full_model,
            spacy_tagger_model=spacy_tagger_model,
            embedding_model=embedding_model,
            psychological_synonyms=psychological_synonyms,
            temporal_synonyms=temporal_synonyms,
        )
        extracted_features_records.append(
            [
                computed_event_density,
                computed_semantic_distance,
                computed_pronoun_transition,
                computed_psychological_density,
                computed_temporal_density,
            ]
        )
        previous_paragraph_embedding = current_paragraph_embedding

    raw_metric_names = [
        "event_density",
        "semantic_distance",
        "pronoun_transition",
        "psychological_density",
        "temporal_density",
    ]
    features_dataframe = pd.DataFrame(extracted_features_records, columns=raw_metric_names)
    segmentation_dataframe = pd.concat([segmentation_dataframe, features_dataframe], axis=1)
    segmentation_dataframe["perspective_shift"] = segmentation_dataframe["semantic_distance"] + segmentation_dataframe["pronoun_transition"]

    save_dataframe_to_numeric_csv(segmentation_dataframe, OUTPUT_RAW_INDICATORS_CSV)

    print("[PROCESS] Computing statistical outputs...")
    save_dataframe_to_numeric_csv(build_descriptive_statistics_dataframe(segmentation_dataframe), OUTPUT_DESCRIPTIVE_STATS_CSV)
    save_dataframe_to_numeric_csv(build_correlation_results_dataframe(segmentation_dataframe), OUTPUT_CORRELATIONS_CSV)
    save_dataframe_to_numeric_csv(build_kruskal_wallis_results_dataframe(segmentation_dataframe), OUTPUT_KRUSKAL_WALLIS_CSV)
    save_dataframe_to_numeric_csv(build_mann_whitney_results_dataframe(segmentation_dataframe), OUTPUT_MANN_WHITNEY_CSV)

    print("[PROCESS] Aggregating chapter data for visualizations...")
    chapter_aggregated_dataframe = (
        segmentation_dataframe.groupby(["part", "chapter_id"], observed=True)[TARGET_METRICS].mean().reset_index()
    )
    chapter_aggregated_dataframe["part"] = pd.Categorical(chapter_aggregated_dataframe["part"], categories=NOVEL_PART_ORDER, ordered=True)
    chapter_aggregated_dataframe = chapter_aggregated_dataframe.sort_values(["part", "chapter_id"]).reset_index(drop=True)
    chapter_aggregated_dataframe["chapter_index"] = range(1, len(chapter_aggregated_dataframe) + 1)
    chapter_aggregated_dataframe = apply_standardization_and_smoothing(chapter_aggregated_dataframe)

    print("[PROCESS] Rendering visualizations...")
    if not os.path.exists(REFERENCE_CORPUS_FILEPATH):
        raise FileNotFoundError(f"[ERROR] Reference corpus file missing: {REFERENCE_CORPUS_FILEPATH}")

    reference_corpus_dataframe = pd.read_csv(REFERENCE_CORPUS_FILEPATH)
    nineteenth_century_dataframe = reference_corpus_dataframe[reference_corpus_dataframe["Novel"] != "Mrs. Dalloway"]
    mrs_dalloway_records = reference_corpus_dataframe[reference_corpus_dataframe["Novel"] == "Mrs. Dalloway"]

    if mrs_dalloway_records.empty:
        raise ValueError("[ERROR] Mrs. Dalloway is missing from the reference corpus output.")

    mrs_dalloway_target_row = mrs_dalloway_records.iloc[0]

    reference_nineteenth_century_means = {metric: nineteenth_century_dataframe[metric].dropna().tolist() for metric in TARGET_METRICS}
    mrs_dalloway_means = {metric: mrs_dalloway_target_row[metric] for metric in TARGET_METRICS}

    boxplot_figure, boxplot_axes = plt.subplots(2, 2, figsize=BOXPLOT_FIGURE_SIZE)
    boxplot_custom_palette = ["#fae9ae", "#b9dafa", "#f1c8c8"]

    for plot_axis, target_metric in zip(boxplot_axes.flatten(), TARGET_METRICS):
        sns.boxplot(
            x="part",
            y=target_metric,
            data=chapter_aggregated_dataframe,
            ax=plot_axis,
            palette=boxplot_custom_palette,
            order=NOVEL_PART_ORDER,
            zorder=2,
        )

        reference_band_minimum = min(reference_nineteenth_century_means[target_metric])
        reference_band_maximum = max(reference_nineteenth_century_means[target_metric])
        nineteenth_century_legend_label = "19th-Century Reference Band" if target_metric == TARGET_METRICS[0] else ""

        plot_axis.axhspan(
            ymin=reference_band_minimum,
            ymax=reference_band_maximum,
            color="gray",
            alpha=0.25,
            zorder=0,
            label=nineteenth_century_legend_label,
        )

        mrs_dalloway_legend_label = "Mrs. Dalloway" if target_metric == TARGET_METRICS[0] else ""
        plot_axis.axhline(
            y=mrs_dalloway_means[target_metric],
            color="#2c3e50",
            linestyle="--",
            linewidth=2,
            zorder=1,
            label=mrs_dalloway_legend_label,
        )

        plot_axis.set_title(METRIC_DISPLAY_LABELS[target_metric], fontsize=15, fontweight="bold")
        plot_axis.set_ylabel("Chapter-Level Mean", fontsize=14)
        plot_axis.set_xlabel("")
        if target_metric == TARGET_METRICS[0]:
            plot_axis.legend(title="References", fontsize=12, title_fontsize=13, loc="best")
        plot_axis.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_BOXPLOT_FIGURE, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()

    save_dataframe_to_numeric_csv(
        chapter_aggregated_dataframe[["part", "chapter_id"] + TARGET_METRICS],
        OUTPUT_BOXPLOT_DATA,
    )

    print("[PROCESS] Generating chapter-level trends visualization...")
    trends_figure, trends_axes = plt.subplots(len(TARGET_METRICS), 1, figsize=TRENDS_FIGURE_SIZE, sharex=True)
    trends_figure.subplots_adjust(hspace=0.1, top=0.98)
    trends_plot_colors = ["#E88D67", "#8B8EC7", "#76C28F", "#6ABBE0"]

    for plot_index, (trend_axis, target_metric, plot_color) in enumerate(zip(trends_axes, TARGET_METRICS, trends_plot_colors)):
        target_smoothed_column = f"smoothed_{target_metric}"
        sns.lineplot(
            x="chapter_index",
            y=target_smoothed_column,
            data=chapter_aggregated_dataframe,
            color=plot_color,
            linewidth=2.5,
            ax=trend_axis,
        )
        trend_axis.fill_between(
            chapter_aggregated_dataframe["chapter_index"],
            chapter_aggregated_dataframe[target_smoothed_column],
            0,
            color=plot_color,
            alpha=0.3,
        )
        trend_axis.axhline(0, color="gray", linewidth=0.8)

        part_boundary_indices = chapter_aggregated_dataframe[
            chapter_aggregated_dataframe["part"] != chapter_aggregated_dataframe["part"].shift()
        ].index

        for boundary_index in part_boundary_indices:
            if boundary_index != 0:
                trend_axis.axvline(
                    x=chapter_aggregated_dataframe.loc[boundary_index, "chapter_index"] - 0.5,
                    color="#d62728",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                )

        trend_axis.set_ylabel(
            METRIC_DISPLAY_LABELS[target_metric],
            fontsize=12,
            rotation=0,
            ha="right",
            va="center",
        )
        trend_axis.yaxis.set_label_coords(-0.06, 0.5)
        trend_axis.tick_params(axis="y", labelsize=9, colors="dimgray")
        trend_axis.spines["top"].set_visible(False)
        trend_axis.spines["right"].set_visible(False)

        if plot_index < len(TARGET_METRICS) - 1:
            trend_axis.spines["bottom"].set_visible(False)
            trend_axis.tick_params(axis="x", length=0)

    plt.xlabel("Chapter Index", fontsize=14)
    plt.xticks(np.arange(1, len(chapter_aggregated_dataframe) + 1, 1), fontsize=11)
    plt.xlim(0, len(chapter_aggregated_dataframe) + 1)
    plt.savefig(OUTPUT_TRENDS_FIGURE, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()

    trends_data_columns = ["chapter_index", "part", "chapter_id"]
    trends_data_columns.extend(f"standardized_{metric}" for metric in TARGET_METRICS)
    trends_data_columns.extend(f"smoothed_{metric}" for metric in TARGET_METRICS)
    save_dataframe_to_numeric_csv(
        chapter_aggregated_dataframe[trends_data_columns],
        OUTPUT_TRENDS_DATA,
    )

    print(f"[PROCESS] Analysis completed successfully. Output saved to: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    run_pipeline()
