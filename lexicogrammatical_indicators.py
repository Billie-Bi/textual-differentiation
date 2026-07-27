import os
import re
import unicodedata
import warnings
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal, kurtosis, mannwhitneyu, skew, spearmanr
from sentence_transformers import SentenceTransformer
import spacy
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import umap

os.environ["NUMBA_CACHE_DIR"] = r"E:\BaiduSyncdisk\programe\Textual_Indicators_2"
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 14
sns.set_theme(style="whitegrid", palette="viridis")

INPUT_FILEPATH = "Textual_Indicators_output/text_segmentation/to_the_lighthouse_paragraphs.csv"
REFERENCE_CORPUS_FILEPATH = "Textual_Indicators_output/reference_corpus/reference_corpus_textual_indicators.csv"
OUTPUT_DIRECTORY = "Textual_Indicators_output/lexicogrammatical_indicators"

OUTPUT_RAW_PARAGRAPH_CSV = os.path.join(OUTPUT_DIRECTORY, "01a_Lexicogrammatical_Indicators_Raw_Paragraph_Indicators.csv")
OUTPUT_RAW_CHAPTER_CSV = os.path.join(OUTPUT_DIRECTORY, "01b_Lexicogrammatical_Indicators_Raw_Chapter_Indicators.csv")
OUTPUT_DESCRIPTIVE_STATS_CSV = os.path.join(OUTPUT_DIRECTORY, "02_Lexicogrammatical_Indicators_Descriptive_Statistics.csv")
OUTPUT_CORRELATIONS_CSV = os.path.join(OUTPUT_DIRECTORY, "03_Lexicogrammatical_Indicators_Spearman_Correlations.csv")
OUTPUT_KRUSKAL_WALLIS_CSV = os.path.join(OUTPUT_DIRECTORY, "04_Lexicogrammatical_Indicators_Kruskal_Wallis_Tests.csv")
OUTPUT_MANN_WHITNEY_CSV = os.path.join(OUTPUT_DIRECTORY, "05_Lexicogrammatical_Indicators_Mann_Whitney_Tests.csv")

OUTPUT_BOXPLOT_FIGURE = os.path.join(OUTPUT_DIRECTORY, "Fig1_Lexicogrammatical_Indicators_Boxplots.png")
OUTPUT_BOXPLOT_DATA = os.path.join(OUTPUT_DIRECTORY, "Fig1_Lexicogrammatical_Indicators_Source_Data_Boxplots.csv")
OUTPUT_TRENDS_FIGURE = os.path.join(OUTPUT_DIRECTORY, "Fig2_Lexicogrammatical_Indicators_Chapter_Trends.png")
OUTPUT_TRENDS_DATA = os.path.join(OUTPUT_DIRECTORY, "Fig2_Lexicogrammatical_Indicators_Source_Data_Chapter_Trends.csv")
OUTPUT_UMAP_FIGURE = os.path.join(OUTPUT_DIRECTORY, "Fig3_Chapter-Level_UMAP_Projection_of_Text_Embeddings.png")
OUTPUT_UMAP_DATA = os.path.join(OUTPUT_DIRECTORY, "Fig3_Chapter-Level_UMAP_Projection_of_Text_Embeddings.csv")

ANALYSIS_GROUP_NAME = "Lexicogrammatical"
NOVEL_PART_ORDER = ["The Window", "Time Passes", "The Lighthouse"]
TARGET_METRICS = [
    "pos_ratio_adj_adv_to_verb",
    "avg_dependency_depth",
    "repeated_words_per_100_tokens",
    "parallel_ratio",
]
METRIC_DISPLAY_LABELS = {
    "pos_ratio_adj_adv_to_verb": "POS Ratio of Adjectives and Adverbs to Verbs",
    "avg_dependency_depth": "Average Dependency Depth",
    "repeated_words_per_100_tokens": "Repeated Words per 100 Tokens",
    "parallel_ratio": "Parallel Ratio",
}


def clean_paragraph_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\n\r\t]", " ", text)
    text = re.sub(r"[;—]+", "; ", text)
    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    return re.sub(r"\s+", " ", text).strip()


def encode_long_text_for_umap(text, embedding_model):
    tokenizer = embedding_model.tokenizer

    max_content_tokens = (
        embedding_model.max_seq_length
        - tokenizer.num_special_tokens_to_add(pair=False)
    )

    if max_content_tokens <= 0:
        raise ValueError(
            "[ERROR] The embedding model has an invalid maximum sequence length."
        )

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False,
    )

    if not token_ids:
        return np.zeros(
            embedding_model.get_sentence_embedding_dimension(),
            dtype=np.float32,
        )

    token_chunks = [
        token_ids[start_index:start_index + max_content_tokens]
        for start_index in range(0, len(token_ids), max_content_tokens)
    ]

    chunk_texts = [
        tokenizer.decode(
            token_chunk,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        for token_chunk in token_chunks
    ]

    chunk_embeddings = embedding_model.encode(
        chunk_texts,
        batch_size=8,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    chunk_weights = np.asarray(
        [len(token_chunk) for token_chunk in token_chunks],
        dtype=np.float64,
    )

    chapter_embedding = np.average(
        chunk_embeddings,
        axis=0,
        weights=chunk_weights,
    )

    return chapter_embedding.astype(np.float32)


def save_dataframe_to_numeric_csv(dataframe, filepath):
    dataframe.to_csv(filepath, index=False, encoding="utf-8", float_format="%.12g")


def calculate_dependency_tree_depth(token):
    return 1 + max((calculate_dependency_tree_depth(child) for child in token.children), default=0)


def extract_lexicogrammatical_features(spacy_document):
    sentences = list(spacy_document.sents)
    num_sentences = max(len(sentences), 1)
    num_words = max(sum(1 for token in spacy_document if token.is_alpha), 1)

    num_adj_adv = sum(1 for token in spacy_document if token.pos_ in ["ADJ", "ADV"])
    num_verbs = max(sum(1 for token in spacy_document if token.pos_ == "VERB"), 1)
    pos_ratio = num_adj_adv / num_verbs

    dependency_depths = [
        calculate_dependency_tree_depth(sent.root)
        / max(sum(1 for token in sent if token.is_alpha), 1)
        for sent in sentences
    ]
    avg_dependency_depth = sum(dependency_depths) / max(len(dependency_depths), 1)

    content_words = [
        token.text.lower()
        for token in spacy_document
        if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"] and token.is_alpha
    ]
    word_counts = pd.Series(content_words, dtype="object").value_counts()
    repeated_words = sum(count for count in word_counts if count > 1)
    repeated_words_per_100_tokens = (repeated_words / num_words) * 100

    parallel_conjunctions = sum(
        1 for token in spacy_document if token.dep_ == "cc" and token.head.pos_ == "VERB"
    )
    parallel_ratio = parallel_conjunctions / num_sentences

    return [
        pos_ratio,
        avg_dependency_depth,
        repeated_words_per_100_tokens,
        parallel_ratio,
    ]


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

    pooled_variance = (
        ((length_1 - 1) * np.var(array_1, ddof=1) + (length_2 - 1) * np.var(array_2, ddof=1))
        / (length_1 + length_2 - 2)
    )
    if not np.isfinite(pooled_variance) or pooled_variance <= 0:
        return np.nan

    return (np.mean(array_1) - np.mean(array_2)) / np.sqrt(pooled_variance)


def build_descriptive_statistics_dataframe(segmentation_dataframe):
    statistics_rows = []
    for metric in TARGET_METRICS:
        analysis_scopes = [("Overall", segmentation_dataframe[metric].dropna())]
        analysis_scopes.extend(
            (part, segmentation_dataframe.loc[segmentation_dataframe["part"] == part, metric].dropna())
            for part in NOVEL_PART_ORDER
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
            correlation_dataframe.loc[finite_p_values_mask, "Unadjusted p-value"], method="fdr_bh"
        )[1]
        
    correlation_dataframe["Significant after FDR correction"] = (
        correlation_dataframe["Adjusted p-value (BH FDR)"] < 0.05
    )
    return correlation_dataframe


def build_kruskal_wallis_results_dataframe(segmentation_dataframe):
    kruskal_rows = []
    group_sizes = {
        part: int(segmentation_dataframe.loc[segmentation_dataframe["part"] == part].shape[0]) 
        for part in NOVEL_PART_ORDER
    }

    for metric in TARGET_METRICS:
        metric_groups = [
            segmentation_dataframe.loc[segmentation_dataframe["part"] == part, metric].dropna() 
            for part in NOVEL_PART_ORDER
        ]
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
                "p-value": kruskal_result.pvalue,
            }
        )
    return pd.DataFrame(kruskal_rows)


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
        mann_whitney_dataframe["Unadjusted p-value"], method="fdr_bh"
    )[1]
    mann_whitney_dataframe["Significant after FDR correction"] = (
        mann_whitney_dataframe["Adjusted p-value (BH FDR)"] < 0.05
    )

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

        chapter_aggregated_dataframe[smoothed_column_name] = chapter_aggregated_dataframe.groupby(
            "part", observed=True
        )[standardized_column_name].transform(
            lambda grouped_series: grouped_series.rolling(
                window=3, center=True, min_periods=1
            ).mean()
        )
    return chapter_aggregated_dataframe


def run_pipeline():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    print("[PROCESS] Initializing models...")
    spacy_full_model = spacy.load("en_core_web_sm")
    embedding_model = SentenceTransformer("all-mpnet-base-v2")

    print("[PROCESS] Formatting dataset...")
    segmentation_dataframe = pd.read_csv(INPUT_FILEPATH)
    required_data_columns = ["part", "chapter_id", "para_idx", "text"]
    missing_data_columns = [col for col in required_data_columns if col not in segmentation_dataframe.columns]
    
    if missing_data_columns:
        raise ValueError(f"[ERROR] Missing required columns in {INPUT_FILEPATH}: {missing_data_columns}")

    segmentation_dataframe = segmentation_dataframe[required_data_columns].copy()
    if segmentation_dataframe["text"].isna().any() or segmentation_dataframe["text"].astype(str).str.strip().eq("").any():
        raise ValueError("[ERROR] Empty or missing paragraph text found.")
        
    segmentation_dataframe["part"] = pd.Categorical(
        segmentation_dataframe["part"], categories=NOVEL_PART_ORDER, ordered=True
    )
    if segmentation_dataframe["part"].isna().any():
        invalid_part_labels = sorted(set(segmentation_dataframe["part"].dropna()) - set(NOVEL_PART_ORDER))
        raise ValueError(f"[ERROR] Unexpected part labels: {invalid_part_labels}")

    segmentation_dataframe = segmentation_dataframe.sort_values(["part", "chapter_id", "para_idx"]).reset_index(drop=True)
    segmentation_dataframe["text"] = segmentation_dataframe["text"].astype(str).apply(clean_paragraph_text)

    print("[PROCESS] Extracting lexicogrammatical indicators...")
    extracted_features_records = []
    for spacy_document in tqdm(
        spacy_full_model.pipe(segmentation_dataframe["text"].tolist(), batch_size=32),
        total=len(segmentation_dataframe),
        desc="Parsing",
    ):
        extracted_features_records.append(extract_lexicogrammatical_features(spacy_document))

    features_dataframe = pd.DataFrame(extracted_features_records, columns=TARGET_METRICS)
    segmentation_dataframe = pd.concat([segmentation_dataframe, features_dataframe], axis=1)
    save_dataframe_to_numeric_csv(segmentation_dataframe, OUTPUT_RAW_PARAGRAPH_CSV)

    print("[PROCESS] Aggregating chapter data...")
    chapter_aggregated_dataframe = (
        segmentation_dataframe.groupby(["part", "chapter_id"], observed=True)
        .agg(
            {
                "text": " ".join,
                "pos_ratio_adj_adv_to_verb": "mean",
                "avg_dependency_depth": "mean",
                "repeated_words_per_100_tokens": "mean",
                "parallel_ratio": "mean",
            }
        )
        .reset_index()
    )
    chapter_aggregated_dataframe["part"] = pd.Categorical(
        chapter_aggregated_dataframe["part"], categories=NOVEL_PART_ORDER, ordered=True
    )
    chapter_aggregated_dataframe = chapter_aggregated_dataframe.sort_values(["part", "chapter_id"]).reset_index(drop=True)
    chapter_aggregated_dataframe["chapter_index"] = range(1, len(chapter_aggregated_dataframe) + 1)
    save_dataframe_to_numeric_csv(chapter_aggregated_dataframe, OUTPUT_RAW_CHAPTER_CSV)

    print("[PROCESS] Generating chapter embeddings and UMAP projection...")

    chapter_embeddings = np.vstack(
        [
            encode_long_text_for_umap(
                text=chapter_text,
                embedding_model=embedding_model,
            )
            for chapter_text in tqdm(
                chapter_aggregated_dataframe["text"].tolist(),
                total=len(chapter_aggregated_dataframe),
                desc="Encoding chapters",
            )
        ]
    )

    umap_reducer = umap.UMAP(
        n_neighbors=10,
        min_dist=0.3,
        n_components=2,
        random_state=42,
    )

    chapter_aggregated_dataframe[["umap_x", "umap_y"]] = (
        umap_reducer.fit_transform(chapter_embeddings)
    )

    print("[PROCESS] Computing statistical outputs...")
    save_dataframe_to_numeric_csv(build_descriptive_statistics_dataframe(segmentation_dataframe), OUTPUT_DESCRIPTIVE_STATS_CSV)
    save_dataframe_to_numeric_csv(build_correlation_results_dataframe(segmentation_dataframe), OUTPUT_CORRELATIONS_CSV)
    save_dataframe_to_numeric_csv(build_kruskal_wallis_results_dataframe(segmentation_dataframe), OUTPUT_KRUSKAL_WALLIS_CSV)
    save_dataframe_to_numeric_csv(build_mann_whitney_results_dataframe(segmentation_dataframe), OUTPUT_MANN_WHITNEY_CSV)

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

    reference_nineteenth_century_means = {
        metric: nineteenth_century_dataframe[metric].dropna().tolist() for metric in TARGET_METRICS
    }
    mrs_dalloway_means = {metric: mrs_dalloway_target_row[metric] for metric in TARGET_METRICS}

    boxplot_custom_palette = ["#fae9ae", "#b9dafa", "#f1c8c8"]
    boxplot_figure, boxplot_axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for plot_axis, target_metric in zip(boxplot_axes.flatten(), TARGET_METRICS):
        sns.boxplot(
            x="part",
            y=target_metric,
            data=chapter_aggregated_dataframe,
            order=NOVEL_PART_ORDER,
            palette=boxplot_custom_palette,
            ax=plot_axis,
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

        plot_axis.set_title(METRIC_DISPLAY_LABELS[target_metric], fontsize=14, fontweight="bold")
        plot_axis.set_ylabel("Raw Values")
        plot_axis.set_xlabel("")
        if target_metric == TARGET_METRICS[0]:
            plot_axis.legend(title="References", fontsize=10, loc="best")

    plt.tight_layout()
    plt.savefig(OUTPUT_BOXPLOT_FIGURE, dpi=300, bbox_inches="tight")
    plt.close()

    save_dataframe_to_numeric_csv(
        chapter_aggregated_dataframe[["part", "chapter_id"] + TARGET_METRICS],
        OUTPUT_BOXPLOT_DATA,
    )

    trends_figure, trends_axes = plt.subplots(len(TARGET_METRICS), 1, figsize=(16, 12), sharex=True)
    trends_figure.subplots_adjust(hspace=0.1,top=0.98)
    trends_plot_colors = ["#e5aba5", "#fcd7af", "#DDBA6B", "#969bc7"]

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
            alpha=0.6,
        )
        trend_axis.axhline(0, color="gray", linewidth=0.8)

        part_boundary_indices = chapter_aggregated_dataframe[
            chapter_aggregated_dataframe["part"] != chapter_aggregated_dataframe["part"].shift()
        ].index

        for boundary_index in part_boundary_indices:
            if boundary_index > 0:
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
    plt.savefig(OUTPUT_TRENDS_FIGURE, dpi=300, bbox_inches="tight")
    plt.close()

    trends_data_columns = ["chapter_index", "part", "chapter_id"]
    trends_data_columns.extend(f"standardized_{metric}" for metric in TARGET_METRICS)
    trends_data_columns.extend(f"smoothed_{metric}" for metric in TARGET_METRICS)
    save_dataframe_to_numeric_csv(
        chapter_aggregated_dataframe[trends_data_columns],
        OUTPUT_TRENDS_DATA,
    )

    chapter_aggregated_dataframe_sorted = (
        chapter_aggregated_dataframe
        .sort_values("chapter_index")
        .reset_index(drop=True)
    )

    plt.figure(figsize=(10, 8))

    umap_deeper_palette = ["#e8c874", "#8cb8db", "#e09a9a"]

    sns.scatterplot(
        x="umap_x",
        y="umap_y",
        hue="part",
        hue_order=NOVEL_PART_ORDER,
        palette=umap_deeper_palette,
        s=260,
        alpha=0.95,
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
        data=chapter_aggregated_dataframe_sorted,
    )

    for _, chapter_row in chapter_aggregated_dataframe_sorted.iterrows():
        plt.text(
            chapter_row["umap_x"],
            chapter_row["umap_y"],
            str(int(chapter_row["chapter_index"])),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="black",
            zorder=3,
        )

    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])

    plt.legend(
        title="Part",
        loc="best",
        frameon=True,
        framealpha=0.9,
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_UMAP_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    save_dataframe_to_numeric_csv(
        chapter_aggregated_dataframe_sorted[
            ["chapter_index", "part", "chapter_id", "umap_x", "umap_y"]
        ],
        OUTPUT_UMAP_DATA,
    )

    print(f"[RESULT] Analysis completed successfully. Output saved to: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    run_pipeline()